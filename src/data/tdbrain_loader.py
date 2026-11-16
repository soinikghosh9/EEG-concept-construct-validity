"""TDBRAIN loader -- van Dijk et al. (2022), Brainclinics 'Two Decades of Brainclinics Research'.

Psychiatric resting-state EEG (ADHD / MDD / OCD / ...), BIDS + BrainVision, 1274 subjects, 500 Hz,
eyes-closed (restEC) + eyes-open (restEO), 26 EEG channels (our 19-channel 10-20 montage is a subset).

USED UNSUPERVISED ONLY. TDBRAIN has NO dementia labels, so it is *never* added to the labeled 4-class
training factory. Its value to this project is twofold:
  (a) a large, distribution-diverse SSL pretraining corpus -- testing whether our earlier in-house SSL
      stalled because the pretraining set was small/same-distribution (lever: bigger-corpus MAE);
  (b) a clean non-dementia out-of-distribution (OOD) validation set for the deployment OOD detector.

Each subject is given a placeholder label (CN) that the masked-SSL objective ignores; never trust it as
a class. Preprocessing mirrors caueeg_loader EXACTLY (standardize channels -> resample -> robust-scale)
so the pretraining/OOD distribution matches the dementia cohorts; per-window band-filtering + z-scoring
are applied uniformly downstream by dataset_wrapper.
"""
from pathlib import Path
from typing import Dict, List
import warnings

import numpy as np
import mne
from src.data.scaling import robust_scale   # centralised: see src/data/scaling.py

from .base_loader import BaseDataset
from ..config.config import CHANNEL_NAMES, SAMPLING_RATE

# TDBRAIN uses 10-10 labels; map the four that differ to our old-10-20 CHANNEL_NAMES. The other 15 match.
_RENAME = {'T7': 'T3', 'T8': 'T4', 'P7': 'T5', 'P8': 'T6'}
CLASS_CN = 2  # placeholder index (CL = ['AD','FTD','CN','MCI']); IGNORED by SSL.


class TDBRAINDataset(BaseDataset):
    """BIDS/BrainVision resting-state EEG. Unsupervised use only (SSL pretraining + OOD)."""

    def __init__(self, root_path: Path, task: str = 'restEC', verbose: bool = True):
        super().__init__(root_path)
        self.task = task
        self.verbose = verbose
        self.subject_ids = sorted(p.name for p in self.root_path.glob('sub-*') if p.is_dir())
        if verbose:
            print(f"Loaded TDBRAIN: {len(self.subject_ids)} subjects (task={task}, UNSUPERVISED)")

    def get_subject_ids(self) -> List[str]:
        return self.subject_ids

    def get_subject_info(self, subject_id: str) -> Dict:
        # No dementia label -> placeholder CN (ignored by masked-SSL). 'TDBRAIN_' tag enables OOD/cohort_of.
        return {
            'subject_id': f"TDBRAIN_{subject_id}", 'dataset': 'TDBRAIN', 'group': 'TDBRAIN',
            'label': CLASS_CN, 'age': -1, 'sex': -1, 'mmse': -1.0, 'condition': 'eyes_closed',
            'metadata': np.array([0.0, 0.0, -1.0], dtype=np.float32),
        }

    def _vhdr(self, subject_id: str, task: str) -> Path:
        return self.root_path / subject_id / 'ses-1' / 'eeg' / f"{subject_id}_ses-1_task-{task}_eeg.vhdr"

    def _standardize_channels(self, raw: mne.io.Raw) -> mne.io.Raw:
        raw.rename_channels({k: v for k, v in _RENAME.items() if k in raw.ch_names})
        montage = mne.channels.make_standard_montage('standard_1020')
        raw.set_montage(montage, on_missing='ignore')
        keep = [ch for ch in CHANNEL_NAMES if ch in raw.ch_names]
        raw.pick(keep)
        raw.reorder_channels([ch for ch in CHANNEL_NAMES if ch in raw.ch_names])
        return raw

    def load_raw(self, subject_id: str) -> mne.io.Raw:
        vhdr = self._vhdr(subject_id, self.task)
        if not vhdr.exists():  # fall back to the other rest task
            alt = 'restEO' if self.task == 'restEC' else 'restEC'
            vhdr = self._vhdr(subject_id, alt)
        if not vhdr.exists():
            raise FileNotFoundError(f"No rest EEG for {subject_id}")
        with warnings.catch_warnings():
            warnings.simplefilter('ignore')
            raw = mne.io.read_raw_brainvision(str(vhdr), preload=True, verbose=False)
        raw = self._standardize_channels(raw)
        native = float(raw.info['sfreq'])
        if native != SAMPLING_RATE:
            raw.resample(SAMPLING_RATE)
        try:
            raw.set_channel_types({ch: 'eeg' for ch in raw.ch_names if ch in CHANNEL_NAMES})
        except Exception:
            pass
        raw._data = robust_scale(raw.get_data())
        return raw
