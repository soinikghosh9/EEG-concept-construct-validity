"""
EEG Dataset Wrapper.

Converts MNE Raw buffers into PyTorch sliding windows.
Supports:
- Window limiting per subject (prevents overfitting)
- Minority class augmentation (MCI, FTD)
- Configurable overlap
"""

import os
import hashlib
from pathlib import Path
import torch
from torch.utils.data import Dataset
import numpy as np
from typing import List, Dict, Optional
from scipy.signal import butter, filtfilt

from .augmentation import EEGAugmentation, MinorityClassAugmenter
from .window_features import compute_all_streams
from .alignment import apply_alignment

# Class labels
CLASS_AD = 0
CLASS_FTD = 1
CLASS_CN = 2
CLASS_MCI = 3


class EEGDatasetWrapper(Dataset):
    """
    Wraps a BaseDataset loader.
    Preloads data into RAM (float32) for maximum efficiency and stability.

    Features:
    - Limits windows per subject to prevent overfitting
    - Supports online augmentation for minority classes
    - Configurable overlap ratio
    """

    def __init__(self, loader, config):
        self.loader = loader
        self.config = config
        self.window_size = config['n_times']  # 2000 samples

        # Configurable overlap: 0.0 = no overlap, 0.5 = 50% overlap
        overlap_ratio = config.get('overlap_ratio', 0.0)  # DEFAULT: No overlap (changed from 0.5)
        self.stride = int(self.window_size * (1 - overlap_ratio))
        if self.stride < 1:
            self.stride = 1

        self.sfreq = config.get('sfreq', 200.0)  # Default sampling rate

        # Limit windows per subject to prevent overfitting
        self.max_windows_per_subject = config.get('max_windows_per_subject', 50)

        # Optional hand-crafted biomarker streams returned alongside band features
        # (used by feature_streams= setting in NeuroChronoGraphV3 for ablation).
        self.compute_streams = config.get('compute_feature_streams', False)
        # Stream values are deterministic functions of (orig_sid, start) — augmentation
        # is applied AFTER stream extraction — so caching by that key amortizes the
        # ~60 ms/window cost across epochs (first epoch pays once, rest are free).
        self._stream_cache: dict = {} if self.compute_streams else None

        # Enable/disable augmentation
        self.enable_augmentation = config.get('enable_augmentation', True)
        self.augmenter = None
        self.minority_augmenter = None

        if self.enable_augmentation:
            self.augmenter = EEGAugmentation(
                time_shift_max=100,
                amplitude_scale_range=(0.9, 1.1),
                noise_std=0.02,
                channel_dropout_prob=0.1,
                augment_prob=0.3  # Lower prob for training augmentation
            )
            self.minority_augmenter = MinorityClassAugmenter()

        # Define Bands
        self.bands = {
            'delta': (0.5, 4),
            'theta': (4, 8),
            'alpha': (8, 13),
            'beta': (13, 30),
            'gamma': (30, 45)
        }
        self.filters = self._design_filters()

        self.samples = []  # List of (subject_id, start_idx, is_augmented)
        self.groups = []   # Subject IDs for splitting
        self.labels = []   # 0, 1, 2, 3

        # Data Store: {sid: np.array(channels, times)}
        self.data_store = {}

        # Optional band-feature cache (NCG_CACHE_BANDS=1): filter each FULL recording once (lazily,
        # float16) and slice per window (RAM-cached). Superseded by memmap+num_workers below for the
        # production fast path; mutually exclusive with memmap (which uses exact per-window filtfilt).
        self.cache_bands = bool(config.get('cache_bands', False)) or \
            os.environ.get('NCG_CACHE_BANDS', '0').strip() in ('1', 'true', 'True', 'yes')
        self.band_store = {}

        # MEMMAP mode (NCG_MEMMAP=1): store each recording as an on-disk .npy on the NVMe SSD and
        # read it lazily via memmap in __getitem__, instead of holding every recording in RAM. This
        # (a) frees RAM and (b) lets num_workers>0 SHARE the data via the OS page cache -- no
        # per-worker pickling/duplication, so the idle CPU cores can do parallel data loading
        # (exact per-window filtfilt) and feed the GPU. See __getstate__ for worker pickle-safety.
        self.use_memmap = os.environ.get('NCG_MEMMAP', '0').strip() in ('1', 'true', 'True', 'yes')
        self.memmap_dir = Path(__file__).resolve().parents[2] / 'outputs' / 'memmap_cache'
        if self.use_memmap:
            self.memmap_dir.mkdir(parents=True, exist_ok=True)
        self._mmap_handles = {}   # per-process lazy memmap cache (excluded from pickling)
        self.info_cache = {}      # sid -> {'label','metadata'} so __getitem__ is loader-free

        # Augmented samples storage (for minority classes)
        self.augmented_samples = []  # List of pre-computed augmented samples

        print(f"[{loader.__class__.__name__}] Preloading data...")
        self._prepare_and_load()

        # Print class distribution
        self._print_class_distribution()
        
    def _design_filters(self):
        """Pre-calculate Butterworth filters for each band."""
        filters = {}
        nyq = 0.5 * self.sfreq
        for band, (l, h) in self.bands.items():
            b, a = butter(4, [l / nyq, h / nyq], btype='band')
            filters[band] = (b, a)
        return filters
        
    def _apply_filters(self, data: np.ndarray) -> Dict[str, np.ndarray]:
        """Apply filters to a data window [n_channels, n_times]."""
        filtered = {}
        for band, (b, a) in self.filters.items():
            # Apply filter along time axis (axis=-1)
            # Using float32 for speed, filtfilt handles the rest
            filt_data = filtfilt(b, a, data, axis=-1).astype(np.float32)
            filtered[band] = filt_data
        return filtered

    def _safe_name(self, sid):
        """Filesystem-safe, collision-free memmap filename for a subject id."""
        return hashlib.md5(str(sid).encode('utf-8')).hexdigest()[:12]

    def _get_full(self, sid):
        """Full recording: in-RAM array, or a lazily-opened read-only NVMe memmap (NCG_MEMMAP)."""
        rec = self.data_store[sid]
        if not self.use_memmap:
            return rec
        h = self._mmap_handles.get(sid)
        if h is None:
            h = np.load(rec['path'], mmap_mode='r')
            self._mmap_handles[sid] = h
        return h

    def __getstate__(self):
        """Pickle-safe state for num_workers>0 (Windows spawn): drop per-process memmap handles and
        the (possibly unpicklable, and unused in workers) loader. __getitem__ relies only on
        info_cache + memmap paths, so workers never duplicate the in-RAM recordings."""
        s = self.__dict__.copy()
        s['_mmap_handles'] = {}
        if getattr(self, 'use_memmap', False):
            s['loader'] = None
        return s
        
    def _prepare_and_load(self):
        """
        Pre-calculate indices and load data into RAM.
        Applies window limiting and generates augmented samples for minority classes.
        """
        ids = self.loader.get_subject_ids()

        # Track samples by class for augmentation
        class_samples = {CLASS_AD: [], CLASS_FTD: [], CLASS_CN: [], CLASS_MCI: []}

        for sid in ids:
            try:
                # 1. Get metadata
                info = self.loader.get_subject_info(sid)
                label = info['label']

                # 2. Load Raw Data
                raw = self.loader.load_raw(sid)

                # Convert to float32 numpy array to save RAM (MNE uses float64)
                data = raw.get_data().astype(np.float32)

                # Optional cross-cohort harmonization (Euclidean Alignment), env NCG_ALIGN=ea.
                # Applied per recording BEFORE windowing so every cohort shares a common
                # covariance reference. Confound-safe (uses no labels); no-op when NCG_ALIGN unset.
                data = apply_alignment(data).astype(np.float32)

                n_samples = data.shape[1]

                # 3. Cache subject info so __getitem__ never calls the loader (worker-safe).
                self.info_cache[sid] = {'label': info['label'], 'metadata': info['metadata']}

                # 4. Store recording: on-disk memmap (frees RAM + shareable across workers) or RAM.
                if self.use_memmap:
                    _mmp = self.memmap_dir / (self._safe_name(sid) + '.npy')
                    np.save(_mmp, data)
                    self.data_store[sid] = {'path': str(_mmp), 'shape': tuple(data.shape)}
                else:
                    self.data_store[sid] = data

                # 4. Generate windows with limiting
                # Handle recordings shorter than window_size by padding
                min_required_samples = self.window_size // 2  # At least half window size

                if n_samples >= self.window_size:
                    # Normal case: generate sliding windows
                    possible_starts = list(range(0, n_samples - self.window_size + 1, self.stride))

                    # Limit windows per subject
                    if len(possible_starts) > self.max_windows_per_subject:
                        # AUDIT H2: deterministic per-subject subsample via a LOCAL rng.
                        # Previously reseeded the GLOBAL numpy RNG with hash(sid) (salted by
                        # PYTHONHASHSEED -> non-reproducible) then called np.random.seed(None),
                        # clobbering the global seed for everything downstream.
                        import hashlib
                        _seed = int(hashlib.md5(str(sid).encode("utf-8")).hexdigest()[:8], 16)
                        _rng = np.random.default_rng(_seed)
                        selected_starts = _rng.choice(
                            possible_starts,
                            self.max_windows_per_subject,
                            replace=False
                        )
                    else:
                        selected_starts = possible_starts

                    for start in selected_starts:
                        self.samples.append((sid, start, False))  # False = not augmented
                        self.groups.append(self._person_group(sid))  # AUDIT T2.2: person-level group
                        self.labels.append(label)

                        # Store sample reference for minority class augmentation
                        if label in [CLASS_MCI, CLASS_FTD]:
                            class_samples[label].append((sid, start, info['metadata']))

                elif n_samples >= min_required_samples:
                    # Short recording: create a single padded sample at start=0
                    # The __getitem__ method will handle padding
                    self.samples.append((sid, 0, False))
                    self.groups.append(self._person_group(sid))  # AUDIT T2.2: person-level group
                    self.labels.append(label)

                    # Store sample reference for minority class augmentation
                    if label in [CLASS_MCI, CLASS_FTD]:
                        class_samples[label].append((sid, 0, info['metadata']))
                else:
                    print(f"  Warning: {sid} has only {n_samples} samples (need at least {min_required_samples}), skipping.")

                # Clean up MNE object
                del raw

            except Exception as e:
                print(f"Skipping {sid}: {e}")

        # 5. Generate augmented samples for minority classes
        if self.enable_augmentation and self.minority_augmenter:
            self._generate_minority_augmentations(class_samples)

    def _generate_minority_augmentations(self, class_samples: Dict):
        """
        Generate augmented samples for minority classes.
        """
        # Augmentation multipliers (MINIMAL to prevent synthetic bias)
        # MCI augmentation was causing AD→MCI confusion via staging head
        # Reduced to minimum necessary for class balance
        aug_factors = {
            CLASS_MCI: 2,   # 2x augmentation for MCI (was 3x - still too aggressive)
            CLASS_FTD: 2,   # 2x augmentation for FTD
        }

        for label, factor in aug_factors.items():
            samples = class_samples.get(label, [])
            if not samples:
                continue

            n_original = len(samples)
            n_to_add = n_original * (factor - 1)

            print(f"  Augmenting class {label}: {n_original} -> {n_original + n_to_add} samples")

            for _ in range(factor - 1):
                for sid, start, metadata in samples:
                    # Create augmented sample reference
                    self.samples.append((sid, start, True))  # True = augmented
                    # AUDIT T2.1: inherit the REAL subject group (was f"{sid}_aug", which made the
                    # splitter treat a noised copy as a different subject -> same person leaked across
                    # train/val/test). Augmented windows are also excluded from val/hold-out eval in
                    # 18_train_hierarchical.py so synthetic windows never enter reported metrics.
                    self.groups.append(self._person_group(sid))  # AUDIT T2.1/T2.2: real person-level group
                    self.labels.append(label)

    @staticmethod
    def _person_group(sid):
        """AUDIT T2.2: person-level group id used ONLY for train/val/test splitting.

        Strips paradigm/condition tokens so a patient's eyes-closed and eyes-open recordings
        (AlzEEG, which encoded the condition in the id) -- or any two paradigms of the same
        person -- can never straddle the split. The full `sid` remains the storage/metadata key.
        OpenNeuro ids (bare 'sub-001') are already person-level and pass through unchanged.
        """
        s = str(sid)
        for tok in ("_Eyes_closed", "_Eyes_open", "_Eyes_Closed", "_Eyes_Open",
                    "_eyes_closed", "_eyes_open"):
            s = s.replace(tok, "")
        return s

    def _print_class_distribution(self):
        """Print class distribution after loading."""
        labels_arr = np.array(self.labels)
        class_names = {0: 'AD', 1: 'FTD', 2: 'CN', 3: 'MCI'}
        print("  Class Distribution:")
        for idx in range(4):
            count = (labels_arr == idx).sum()
            print(f"    {class_names[idx]}: {count} samples")
                
    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        sample_info = self.samples[idx]
        sid = sample_info[0]
        start = sample_info[1]
        is_augmented = sample_info[2] if len(sample_info) > 2 else False

        # Handle augmented group ID (e.g., "sub-001_aug")
        orig_sid = sid.replace("_aug", "") if isinstance(sid, str) and "_aug" in sid else sid

        # Retrieve preloaded data (in-RAM array or lazily-opened NVMe memmap)
        full_data = self._get_full(orig_sid)
        end = start + self.window_size

        # Slicing numpy array is extremely fast
        data = full_data[:, start:end].copy()  # Copy for augmentation safety

        # Check integrity
        if data.shape[1] != self.window_size:
            diff = self.window_size - data.shape[1]
            data = np.pad(data, ((0, 0), (0, diff)), mode='constant')

        # ========== PER-SAMPLE NORMALIZATION (mode via NCG_NORM) ==========
        # 'channel' (DEFAULT, = v1..v4): per-channel z-score. Harmonizes multi-cohort amplitude BUT
        #   erases the inter-channel power TOPOGRAPHY (posterior alpha, regional slowing) -- a core
        #   dementia biomarker. Suspected shared upstream cap (all models plateau at within-CAUEEG k~0.37).
        # 'global': single mean/std over channels+time per window. Harmonizes overall amplitude while
        #   PRESERVING inter-channel power ratios (topography). v5 fix; A/B'd in 46_normalization_probe.py.
        eps = 1e-8
        _norm = os.environ.get('NCG_NORM', 'channel')
        if _norm == 'global':
            gstd = float(data.std()) + eps
            data = (data - data.mean()) / gstd
            std = np.full((data.shape[0], 1), gstd, dtype=data.dtype)  # for the cache-bands path below
        else:
            mean = data.mean(axis=1, keepdims=True)
            std = data.std(axis=1, keepdims=True) + eps
            data = (data - mean) / std

        # Optional artifact winsorization (NCG_CLIP, in post-normalization SD units; 0 = off, unchanged).
        # Mitigates the relaxed 200uV epoch rejection by capping extreme residual spikes (eye/muscle)
        # without dropping windows (index-safe). e.g. NCG_CLIP=6 clips to +/-6 SD.
        _clip = float(os.environ.get('NCG_CLIP', '0'))
        if _clip > 0:
            np.clip(data, -_clip, _clip, out=data)

        # Convert to tensor
        x = torch.from_numpy(data.astype(np.float32))

        # Apply augmentation if flagged
        if is_augmented and self.augmenter is not None:
            x = self.augmenter(x)

        # Get metadata (from the loader-free cache built at load -> safe under num_workers>0)
        info = self.info_cache[orig_sid]

        # Generate Band Features.
        # Fast path (cache_bands): filter the FULL recording once (cached, float16) and slice.
        # filter(zscore(window)) == filter((raw-mean)/std) ~= filter(raw)/std because band-pass
        # removes the DC mean term, so slicing the pre-filtered recording and dividing by the
        # window std matches the original per-window filter up to small window-edge effects.
        if self.cache_bands and not self.use_memmap and full_data.shape[1] >= self.window_size:
            rec = self.band_store.get(orig_sid)
            if rec is None:
                rec = {bn: filtfilt(b, a, full_data, axis=-1).astype(np.float16)
                       for bn, (b, a) in self.filters.items()}
                self.band_store[orig_sid] = rec
            band_features_np = {}
            for bn, ffull in rec.items():
                seg = ffull[:, start:end].astype(np.float32)
                if seg.shape[1] != self.window_size:
                    seg = np.pad(seg, ((0, 0), (0, self.window_size - seg.shape[1])), mode='constant')
                band_features_np[bn] = seg / std
        else:
            band_features_np = self._apply_filters(data)
        band_features_torch = {k: torch.from_numpy(np.ascontiguousarray(v))
                               for k, v in band_features_np.items()}

        # Apply augmentation to band features if needed
        if is_augmented and self.augmenter is not None:
            # ONE amplitude scale shared across all bands. Calling augmenter.amplitude_scale()
            # per band draws an independent np.random.uniform each time, which randomises
            # RELATIVE band power -- the gain-invariant quantity BandSpecificEncoder is built
            # on, and the representation that transfers best across sites. Harmless while
            # band_features were unread by v2; destructive the moment they are wired in.
            if np.random.rand() <= self.augmenter.augment_prob:
                shared_scale = float(np.random.uniform(*self.augmenter.amplitude_scale_range))
                for band_name in band_features_torch:
                    band_features_torch[band_name] = band_features_torch[band_name] * shared_scale
            for band_name in band_features_torch:
                band_features_torch[band_name] = self.augmenter.add_noise(
                    band_features_torch[band_name]
                )

        sample = {
            'x': x,
            'band_features': band_features_torch,
            'label': torch.tensor(info['label'], dtype=torch.long),
            'metadata': torch.tensor(info['metadata'], dtype=torch.float32),
            'subject_id': orig_sid,
            'is_augmented': is_augmented,
        }

        if self.compute_streams:
            cache_key = (orig_sid, int(start))
            cached = self._stream_cache.get(cache_key)
            if cached is None:
                cached = compute_all_streams(data, band_features_np, self.sfreq)
                self._stream_cache[cache_key] = cached
            sample['feature_streams'] = {
                k: torch.from_numpy(v) for k, v in cached.items()
            }

        return sample
