"""Deep-model benchmark on the harmonised corpus.

Seven reference architectures plus this project's graph network, trained under one protocol on one
corpus, and scored the same way as the classical models in `128_benchmark_harmonised.py` so the
two tables can sit side by side.

**Reference implementations, not reimplementations.** The architectures come from braindecode,
which is the field's maintained source for them. The previous benchmark used hand-written EEGNet,
ResNet and transformer baselines living inside `24_baseline_comparisons.py`; a reviewer has no way
to tell a faithful reimplementation from a weakened one, and a deep baseline that loses because it
was rebuilt badly proves nothing. Using the canonical implementations removes that objection
entirely.

    ShallowFBCSPNet   Schirrmeister 2017, the standard shallow filter-bank CSP network
    Deep4Net          Schirrmeister 2017, its deep counterpart
    EEGNetv4          Lawhern 2018, the most-cited compact EEG CNN
    EEGConformer      Song 2023, convolution plus self-attention
    ATCNet            Altaheri 2023, attention over temporal convolutions
    EEGNeX            Chen 2024
    EEGITNet          Salami 2022, inception-style temporal convolutions

Protocol is identical to the classical arm: train on CAUEEG only, five folds grouped by subject,
three classes, subject score = mean of window posteriors, blind sites scored with the posterior
restricted to the two classes that site contains. Every architecture gets the same optimiser,
schedule, budget, batch size and early-stopping rule, so the comparison is between architectures
rather than between training recipes.

The raw-window tensor is built once per scaling arm and memory-mapped (~4.8 GB), using exactly the
window selection and rejection rule of `127_extract_harmonised.py`, so the deep and classical arms
see the same windows.

    python experiments/131_deep_benchmark.py --mode channel
Out: outputs/results/audit/deep_benchmark_<mode>.json
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
import time
import warnings
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "experiments"))

from _protocol_v2 import CN, DEM, MCI, TARGETS, auc_ci, pairwise_score, safe_auc, target_subset  # noqa: E402
from src.utils.provenance import write_result  # noqa: E402

CACHE = ROOT / "outputs" / "harmonised"
FEAT = ROOT / "outputs" / "harmonised_features"
WIN = ROOT / "outputs" / "harmonised_windows"
OUT = ROOT / "outputs" / "results" / "audit"

WINDOW, STRIDE, MAX_WINDOWS, REJECT_V = 1024, 1024, 50, 200e-6
SEED, FOLDS = 42, 5
LABEL3 = {"CN": CN, "C": CN, "CONTROL": CN, "MCI": MCI,
          "AD": DEM, "A": DEM, "FTD": DEM, "F": DEM}
EXPECTED = {("CAUEEG", CN): 459, ("CAUEEG", MCI): 417, ("CAUEEG", DEM): 244,
            ("ds004504_raw", CN): 29, ("ds004504_raw", DEM): 59,
            ("GENEEG", CN): 35, ("GENEEG", MCI): 28}

#: Reference values from the classical arm on the same corpus, for the comparison that matters.
CLASSICAL = {"OpenNeuro|CN_vs_dementia": {"random forest": 0.863, "concept bottleneck": 0.816},
             "GENEEG|CN_vs_MCI": {"random forest": 0.581, "concept bottleneck": 0.582}}

ALL_MODELS = ("shallow", "deep4", "eegnet", "conformer", "atcnet", "eegnex",
              "eegitnet")


def window_starts(n: int, sid: str):
    if n < WINDOW:
        return []
    starts = list(range(0, n - WINDOW + 1, STRIDE))
    if len(starts) > MAX_WINDOWS:
        seed = int(hashlib.md5(str(sid).encode("utf-8")).hexdigest()[:8], 16)
        starts = sorted(np.random.default_rng(seed).choice(starts, MAX_WINDOWS, replace=False))
    return list(starts)


def build_windows(mode: str, tag: str | None = None):
    """Materialise [n_windows, 19, 1024] float32 aligned to the feature meta, or reuse it.

    `tag` selects the feature table; `mode` remains the scaling mode. If that table was built
    with per-cohort truncation, the same truncation is applied here, read from the table's own
    extraction summary so the two cannot drift apart. The alignment assertion below then
    checks the result window by window.
    """
    from src.data.scaling import robust_scale

    tag = tag or mode
    d = WIN / tag
    d.mkdir(parents=True, exist_ok=True)
    xpath = d / "windows.npy"
    meta = np.load(FEAT / tag / "meta.npz", allow_pickle=False)
    trunc = {}
    summ = OUT / f"extract_harmonised_{tag}.json"
    if summ.exists():
        trunc = json.loads(summ.read_text()).get("truncated_cohorts") or {}
        if trunc:
            print(f"applying the table's truncation: {trunc}")
    n_expect = len(meta["groups"])
    if xpath.exists():
        X = np.load(xpath, mmap_mode="r")
        if X.shape == (n_expect, 19, WINDOW):
            print(f"reusing window tensor {X.shape}")
            return X, meta
        print(f"window tensor shape {X.shape} != expected {(n_expect, 19, WINDOW)}; rebuilding")

    print(f"building window tensor for {n_expect:,} windows (~{n_expect*19*WINDOW*4/1e9:.1f} GB)")
    X = np.lib.format.open_memmap(xpath, mode="w+", dtype=np.float32,
                                  shape=(n_expect, 19, WINDOW))
    # Driven by the feature table's own record of which window came from where, rather than by
    # replaying the selection rules. Those rules now include a cohort allowlist, a per-cohort
    # truncation and a fixed per-subject quota, and any one of them drifting between the two
    # scripts silently misaligns the tensor against the features. The meta already stores the
    # subject and the sample offset of every window it kept, so reading them back is exact by
    # construction and needs no rules at all.
    want, starts = meta["groups"], meta["start"]
    if "start" not in meta:
        raise RuntimeError("feature meta has no window offsets; re-extract with 127")
    order = {}
    for i, (sid, s) in enumerate(zip(want, starts)):
        order.setdefault(str(sid), []).append((int(s), i))

    t0, done = time.time(), 0
    for k, (sid, rows) in enumerate(order.items(), 1):
        npy = CACHE / f"{sid}.npy"
        if not npy.exists():
            raise RuntimeError(f"{sid} is in the feature table but has no cached recording")
        data = np.load(npy)
        for s, i in rows:
            w = data[:, s:s + WINDOW]
            if w.shape[1] != WINDOW:
                raise RuntimeError(f"{sid} window at {s} is short; table and cache disagree")
            X[i] = robust_scale(w, mode).astype(np.float32)
            done += 1
        if k % 200 == 0:
            print(f"  {k}/{len(order)} subjects, {done:,}/{n_expect:,} windows, "
                  f"{(time.time()-t0)/60:.1f} min", flush=True)
    if done != n_expect:
        raise RuntimeError(f"wrote {done} windows, expected {n_expect}")
    X.flush()
    print(f"built {xpath} in {(time.time()-t0)/60:.1f} min")
    return np.load(xpath, mmap_mode="r"), meta


def make_model(name: str, device):
    import braindecode.models as M
    kw = dict(n_chans=19, n_outputs=3, n_times=WINDOW)
    if name == "shallow":
        m = M.ShallowFBCSPNet(**kw, final_conv_length="auto")
    elif name == "deep4":
        m = M.Deep4Net(**kw, final_conv_length="auto")
    elif name == "eegnet":
        m = M.EEGNetv4(**kw)
    elif name == "conformer":
        m = M.EEGConformer(**kw)
    elif name == "atcnet":
        m = M.ATCNet(**kw, sfreq=256)
    elif name == "eegnex":
        m = M.EEGNeX(**kw)
    elif name == "eegitnet":
        m = M.EEGITNet(**kw)
    else:
        raise ValueError(name)
    return m.to(device)


#: Extra heads worth persisting when an architecture exposes them. v2 is hierarchical -- it makes a
#: screening decision, a staging decision and a subtype decision, and additionally regresses MMSE
#: and emits an evidential uncertainty. Only the recomposed three-class posterior is used for
#: scoring, so without saving these the intermediate clinical decisions are computed and discarded,
#: which is exactly the kind of silently-dead output this project has found before.
EXTRA_HEADS = ("logits_screen", "logits_stage", "logits_subtype", "mmse_pred", "uncertainty")


def forward_extras(model, x):
    """-> dict of any auxiliary head the architecture exposes, as numpy."""
    out = model(x)
    if not isinstance(out, dict):
        return {}
    return {k: out[k].detach().cpu().numpy() for k in EXTRA_HEADS if k in out}


def forward_logits(model, x):
    """Normalise every architecture to a 3-class logit tensor.

    v2 has a hierarchical head -- a screening decision (control vs impaired) and a staging decision
    (mild cognitive impairment vs dementia) -- rather than a flat three-class output, so its two
    binary heads are recomposed the same way the concept bottleneck's are:

        P(CN) = 1 - p_screen,  P(MCI) = p_screen (1 - p_stage),  P(dementia) = p_screen p_stage

    The result is returned as LOG-probabilities. That is exact rather than approximate for a
    cross-entropy objective: log_softmax of a vector of log-probabilities is the identity, because
    their logsumexp is log(1) = 0, so nothing is rescaled on the way through the loss.
    """
    import torch

    out = model(x)
    if not isinstance(out, dict):
        return out
    if "logits" in out:
        return out["logits"]
    if "logits_screen" in out and "logits_stage" in out:
        eps = 1e-8
        p1 = torch.softmax(out["logits_screen"], dim=1)[:, 1]
        p2 = torch.softmax(out["logits_stage"], dim=1)[:, 1]
        return torch.log(torch.stack([1 - p1, p1 * (1 - p2), p1 * p2], dim=1) + eps)
    raise KeyError(f"cannot find a class head in {list(out)}")


def train_one(name, X, y, tr_idx, va_idx, device, epochs, batch, lr, seed):
    """One fold. Returns the state dict of the best epoch by validation balanced accuracy."""
    import torch
    from torch.utils.data import DataLoader, TensorDataset

    torch.manual_seed(seed)
    np.random.seed(seed)
    model = make_model(name, device)

    cls, cnt = np.unique(y[tr_idx], return_counts=True)
    w = np.zeros(3, dtype=np.float32)
    w[cls] = len(tr_idx) / (len(cls) * cnt)
    crit = torch.nn.CrossEntropyLoss(weight=torch.tensor(w, device=device))
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)

    def loader(idx, shuffle):
        return DataLoader(TensorDataset(torch.from_numpy(np.ascontiguousarray(X[idx])),
                                        torch.from_numpy(y[idx].astype(np.int64))),
                          batch_size=batch, shuffle=shuffle, num_workers=0, pin_memory=True)

    tl, vl = loader(tr_idx, True), loader(va_idx, False)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=lr, epochs=epochs,
                                                steps_per_epoch=max(len(tl), 1))
    best, best_state = -1.0, None
    for ep in range(epochs):
        model.train()
        for xb, yb in tl:
            xb, yb = xb.to(device, non_blocking=True), yb.to(device, non_blocking=True)
            opt.zero_grad(set_to_none=True)
            loss = crit(forward_logits(model, xb), yb)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step(); sched.step()
        model.eval()
        preds, trues = [], []
        with torch.no_grad():
            for xb, yb in vl:
                preds.append(forward_logits(model, xb.to(device)).argmax(1).cpu().numpy())
                trues.append(yb.numpy())
        p, t = np.concatenate(preds), np.concatenate(trues)
        bacc = float(np.mean([(p[t == c] == c).mean() for c in np.unique(t)]))
        if bacc > best:
            best = bacc
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
    return best_state, best


def predict(name, state, X, idx, device, batch, want_extras=False):
    """Window-level 3-class posterior, and optionally the auxiliary clinical heads."""
    import torch
    model = make_model(name, device)
    model.load_state_dict(state)
    model.eval()
    out = np.zeros((len(idx), 3), dtype=np.float64)
    extras: dict = {}
    with torch.no_grad():
        for i in range(0, len(idx), batch):
            sl = idx[i:i + batch]
            xb = torch.from_numpy(np.ascontiguousarray(X[sl])).to(device)
            out[i:i + len(sl)] = torch.softmax(forward_logits(model, xb), dim=1).cpu().numpy()
            if want_extras:
                for k, v in forward_extras(model, xb).items():
                    extras.setdefault(k, []).append(v)
    if want_extras:
        return out, {k: np.concatenate(v) for k, v in extras.items()}
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", choices=["channel", "global"], default="channel")
    ap.add_argument("--tag", default=None,
                    help="feature table to read, if not the one named by --mode. The scaling "
                         "mode is still --mode; the tag only selects which corpus.")
    ap.add_argument("--no-checkpoints", action="store_true",
                    help="skip saving fold weights (they are saved by default)")
    ap.add_argument("--models", default=",".join(ALL_MODELS))
    ap.add_argument("--epochs", type=int, default=25)
    ap.add_argument("--batch", type=int, default=128)
    ap.add_argument("--lr", type=float, default=3e-3)
    ap.add_argument("--seeds", default="",
                    help="comma-separated extra base seeds. Each retrains every fold from a "
                         "different initialisation, so the spread is genuine training variance "
                         "rather than fold-assignment noise. Results are stored per seed under "
                         "the model entry; the primary run stays seed 42.")
    args = ap.parse_args()
    warnings.filterwarnings("ignore")
    OUT.mkdir(parents=True, exist_ok=True)

    import torch
    from sklearn.model_selection import StratifiedGroupKFold
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"device={device} | mode={args.mode}")

    X, meta = build_windows(args.mode, args.tag)
    groups, labels, cohort = meta["groups"], meta["labels"], meta["cohort"]
    subs = np.unique(groups)
    idxs = [np.flatnonzero(groups == s) for s in subs]
    sub_coh = np.array([cohort[i[0]] for i in idxs])
    y3 = np.array([LABEL3[labels[i[0]]] for i in idxs])
    # The label check exists to catch a label-VOCABULARY mismatch, which silently relabels whole
    # classes. It must not also hard-fail on a legitimately different corpus: the de-confounded
    # table drops subjects who cannot supply the fixed window quota, so the counts move by design.
    # An unmapped label is still fatal; a changed count is reported.
    unmapped = sorted({str(x) for x in labels} - set(LABEL3))
    if unmapped:
        raise SystemExit(f"labels with no class mapping: {unmapped}")
    for (c, cls), n_exp in EXPECTED.items():
        got = int(((sub_coh == c) & (y3 == cls)).sum())
        if got != n_exp:
            print(f"  [note] {c}/class{cls}: {got} subjects, reference corpus had {n_exp}")
    print(f"{len(subs)} subjects, {len(groups):,} windows | label mapping verified")

    y_win = np.zeros(len(groups), dtype=np.int64)
    for i, s in enumerate(idxs):
        y_win[s] = y3[i]
    tr_sub = subs[sub_coh == "CAUEEG"]
    w_tr = np.flatnonzero(np.isin(groups, tr_sub))
    sgkf = StratifiedGroupKFold(FOLDS, shuffle=True, random_state=SEED)
    splits = list(sgkf.split(np.zeros(len(w_tr)), y_win[w_tr], groups[w_tr]))

    # Every cohort other than the training one that carries a usable contrast. A cohort with a
    # single diagnostic class cannot rank anything and is skipped rather than scored at chance.
    def usable(c):
        s = sub_coh == c
        return any(((s & np.isin(y3, pair)).sum() >= 16
                    and len(np.unique(y3[s & np.isin(y3, pair)])) == 2)
                   for pair in ([CN, DEM], [CN, MCI]))

    eval_sites = [c for c in sorted(set(sub_coh)) if c != "CAUEEG" and usable(c)]
    print(f"evaluation sites: {eval_sites}")

    chosen = [m.strip() for m in args.models.split(",") if m.strip()]
    report = {"mode": args.mode, "epochs": args.epochs, "batch": args.batch, "lr": args.lr,
              "n_windows": int(len(groups)), "n_subjects": int(len(subs)), "models": {}}
    # Merge into any existing table rather than replacing it, so a later architecture can be added
    # without retraining the ones already measured. Guarded on the settings that would make the
    # rows incomparable: a merged table must describe one protocol.
    prev_path = OUT / f"deep_benchmark_{args.tag or args.mode}.json"
    if prev_path.exists():
        prev = json.loads(prev_path.read_text())
        same = all(prev.get(k) == report[k] for k in ("mode", "epochs", "batch", "lr",
                                                      "n_windows", "n_subjects"))
        if same:
            # Keep every previous row, including those about to be re-measured: each is replaced
            # in place as its new result lands. Dropping them up front means an interrupted run
            # leaves the table permanently short of the models it was in the middle of redoing.
            report["models"] = dict(prev.get("models", {}))
            stale = [k for k in chosen if k in report["models"]]
            print(f"merging with {len(report['models'])} previously measured model(s)"
                  + (f"; {', '.join(stale)} will be replaced as they complete" if stale else ""))
        else:
            print("existing table used different settings; starting a fresh one")

    seed_list = [SEED] + [int(s) for s in args.seeds.split(",") if s.strip()]
    SEED_CACHE_PATH = OUT / f"seed_cache_{args.mode}.json"
    seed_cache = json.loads(SEED_CACHE_PATH.read_text()) if SEED_CACHE_PATH.exists() else {}
    if seed_cache:
        print(f"seed cache holds {len(seed_cache)} completed extra-seed run(s)")
    for name in chosen:
        t0 = time.time()
        n_par = sum(p.numel() for p in make_model(name, "cpu").parameters())
        print(f"\n{'='*74}\n{name}  ({n_par/1e3:.0f}k parameters)", flush=True)
        # Seed `seed_list[0]` is the primary run: its fold ensemble is the one whose predictions
        # are saved and reported. Any further seeds retrain every fold from a different
        # initialisation and contribute only their site AUCs, so the spread is genuine training
        # variance rather than fold-assignment noise.
        per_seed_auc: dict = {}
        states, vbaccs = [], []
        for si, base in enumerate(seed_list):
            # Extra seeds are cached as they finish. Each costs two to three hours and contributes
            # only its site AUCs, so there is no reason for an interrupted run to repeat one that
            # already completed. The primary seed is not cached because its fold weights are needed
            # in memory to write the subject-level predictions.
            ckey = f"{name}|{base}"
            if si > 0 and ckey in seed_cache:
                for k, v in seed_cache[ckey].items():
                    per_seed_auc.setdefault(k, []).append(v)
                print(f"  seed {base}: reusing cached result", flush=True)
                continue
            this_seed: dict = {}
            s_states, s_vb = [], []
            for f, (a, b) in enumerate(splits):
                st, vb = train_one(name, X, y_win, w_tr[a], w_tr[b], device,
                                   args.epochs, args.batch, args.lr, base + f)
                s_states.append(st); s_vb.append(vb)
                if not args.no_checkpoints:
                    # Saved so that scoring these models at a new site never again requires
                    # retraining them. Weights are CPU tensors already.
                    cdir = ROOT / "outputs" / "checkpoints" / "deep" / (args.tag or args.mode)
                    cdir.mkdir(parents=True, exist_ok=True)
                    torch.save({"model": name, "seed": base + f, "fold": f,
                                "val_bacc": float(vb), "state": st},
                               cdir / f"{name}_seed{base}_fold{f}.pt")
                if si == 0:
                    print(f"  fold {f}: val balanced acc {vb:.3f}  "
                          f"({(time.time()-t0)/60:.1f} min)", flush=True)
            if si == 0:
                states, vbaccs = s_states, s_vb
            else:
                for site in eval_sites:
                    sel = np.flatnonzero(sub_coh == site)
                    widx = np.concatenate([idxs[i] for i in sel])
                    Pw = np.mean([predict(name, st, X, widx, device, args.batch)
                                  for st in s_states], axis=0)
                    off, cur = [], 0
                    for i in sel:
                        off.append(Pw[cur:cur + len(idxs[i])].mean(axis=0)); cur += len(idxs[i])
                    Ps = np.stack(off)
                    for t in TARGETS:
                        mt, yt = target_subset(y3[sel], t)
                        if mt.sum() < 16 or len(np.unique(yt)) < 2:
                            continue
                        key = (f"{'OpenNeuro' if site.startswith('ds004504') else site}|{t}")
                        a_seed = safe_auc(yt, pairwise_score(Ps[mt], t))
                        per_seed_auc.setdefault(key, []).append(a_seed)
                        this_seed[key] = a_seed
                seed_cache[ckey] = this_seed
                SEED_CACHE_PATH.write_text(json.dumps(seed_cache, indent=1))
                print(f"  seed {base}: extra run complete "
                      f"({(time.time()-t0)/60:.1f} min)", flush=True)

        entry = {"n_parameters": int(n_par), "val_bacc_folds": vbaccs,
                 "val_bacc_mean": float(np.mean(vbaccs)), "seeds": seed_list, "sites": {}}
        for site in eval_sites:
            m = sub_coh == site
            sel = np.flatnonzero(m)
            widx = np.concatenate([idxs[i] for i in sel])
            want_extra = False
            fold_out = [predict(name, st, X, widx, device, args.batch, want_extra)
                        for st in states]
            if want_extra:
                P = np.mean([f[0] for f in fold_out], axis=0)
                ex_win = {k: np.mean([f[1][k] for f in fold_out], axis=0)
                          for k in fold_out[0][1]}
            else:
                P = np.mean(fold_out, axis=0)
                ex_win = {}
            offs, cur, ex_sub = [], 0, {k: [] for k in ex_win}
            for i in sel:
                offs.append(P[cur:cur + len(idxs[i])].mean(axis=0))
                for k, v in ex_win.items():
                    ex_sub[k].append(v[cur:cur + len(idxs[i])].mean(axis=0))
                cur += len(idxs[i])
            Psub = np.stack(offs)
            ex_sub = {k: np.stack(v) for k, v in ex_sub.items()}
            # Persist the subject-level posterior. Without it a comparison can only quote two
            # independent AUCs and their overlapping intervals, which is not a test: the models
            # are scored on the SAME subjects, so the comparison must be paired. Saving here costs
            # nothing and is the difference between "0.886 versus 0.863" and a delta with an
            # interval and a p-value.
            # Predictions belong to the corpus they were computed on, not to the scaling arm.
            # Writing them under the mode put clean-corpus deep predictions in the
            # same directory as old-corpus classical ones, where a paired test would
            # have compared models across two different corpora.
            pdir = OUT / "predictions" / (args.tag or args.mode)
            pdir.mkdir(parents=True, exist_ok=True)
            np.savez_compressed(pdir / f"{name}__{site}.npz",
                                subjects=subs[sel], y3=y3[sel], posterior=Psub, **ex_sub)
            for t in TARGETS:
                mt, yt = target_subset(y3[sel], t)
                if mt.sum() < 16 or len(np.unique(yt)) < 2:
                    continue
                key = f"{'OpenNeuro' if site.startswith('ds004504') else site}|{t}"
                a, lo, hi = auc_ci(yt, pairwise_score(Psub[mt], t))
                ref = CLASSICAL.get(key, {})
                entry["sites"][key] = {"n": int(mt.sum()), "auc": a, "ci95": [lo, hi],
                                       "vs_random_forest": (a - ref["random forest"]
                                                            if a and "random forest" in ref
                                                            else None),
                                       "vs_concept_bottleneck": (a - ref["concept bottleneck"]
                                                                 if a and "concept bottleneck"
                                                                 in ref else None)}
                print(f"  {key:28s} AUC {a:.3f} [{lo:.3f}-{hi:.3f}]  "
                      f"vs RF {a - ref.get('random forest', a):+.3f}  "
                      f"vs concepts {a - ref.get('concept bottleneck', a):+.3f}")
        for key, vals in per_seed_auc.items():
            v = np.array([x for x in vals if x is not None])
            if key in entry["sites"] and len(v):
                allv = np.concatenate([[entry["sites"][key]["auc"]], v])
                entry["sites"][key]["seed_aucs"] = allv.tolist()
                entry["sites"][key]["seed_mean"] = float(allv.mean())
                entry["sites"][key]["seed_sd"] = float(allv.std(ddof=1))
                print(f"  {key:28s} across {len(allv)} seeds: "
                      f"{allv.mean():.3f} +/- {allv.std(ddof=1):.3f}")
        entry["minutes"] = (time.time() - t0) / 60
        report["models"][name] = entry
        write_result(OUT / f"deep_benchmark_{args.tag or args.mode}.json", report,
                     script="131_deep_benchmark.py")

    print(f"\n{'='*74}\nSUMMARY  (classical reference: RF 0.863, concept bottleneck 0.816)")
    print(f"  {'model':14s} {'params':>9s} {'OpenNeuro':>10s} {'GENEEG':>9s} {'min':>6s}")
    for name, e in report["models"].items():
        on = e["sites"].get("OpenNeuro|CN_vs_dementia", {}).get("auc")
        ge = e["sites"].get("GENEEG|CN_vs_MCI", {}).get("auc")
        print(f"  {name:14s} {e['n_parameters']/1e3:8.0f}k "
              f"{(f'{on:.3f}' if on else '    -'):>10s} {(f'{ge:.3f}' if ge else '    -'):>9s} "
              f"{e['minutes']:6.1f}")
    print(f"\nwrote {OUT / f'deep_benchmark_{args.mode}.json'}")


if __name__ == "__main__":
    main()
