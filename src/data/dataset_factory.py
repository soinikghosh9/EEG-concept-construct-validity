"""
Dataset Factory.

Unifies access to all datasets and handles splitting logic.
"""

from pathlib import Path
from typing import List, Tuple, Dict
import torch
from torch.utils.data import Dataset, ConcatDataset
import numpy as np
from sklearn.model_selection import GroupKFold, StratifiedGroupKFold

from ..config.config import DATA_ROOT, PROJECT_ROOT
from .bids_loader import BIDSDataset
from .alz_eeg_loader import AlzEEGDataset
from .mendeley_loader import MendeleyDataset
from .mci_loader import MCIDataset
from .misp_loader import MISPDataset
from .padic_loader import PADICDataset
from .caueeg_loader import CAUEEGDataset
from .tdbrain_loader import TDBRAINDataset  # UNSUPERVISED only (SSL pretraining + OOD); never labeled training
from .geneeg_loader import GENEEGDataset    # external real-world MCI-vs-CN (blind cross-domain test)
from .dataset_wrapper import EEGDatasetWrapper # To be created: wraps MNE raw to Torch Dataset


class DatasetFactory:
    """
    Factory for creating and merging EEG datasets.
    """
    
    def __init__(self):
        self.datasets = []
        self.dataset_names = []
        self.all_subjects = []
        self.all_groups = [] # For GroupKFold (subject IDs)
        self.all_labels = [] # For stratification checks
        
    def add_dataset(self, name: str, root_path: Path):
        """Initialize and add a dataset."""
        print(f"Adding dataset: {name}")
        print(f"Initializing {name}...")
        if name == 'ds004504':
            ds = BIDSDataset(root_path, dataset_name=name)
        elif name == 'ds006036':
            ds = BIDSDataset(root_path, dataset_name=name)
        elif name == 'Alz_EEG':
            ds = AlzEEGDataset(root_path)
        elif name == 'Mendeley':
            ds = MendeleyDataset(root_path)
        elif name == 'MCI_Dataset':
            ds = MCIDataset(root_path)
        elif name == 'MISP':
            ds = MISPDataset(root_path)
        elif name == 'P-ADIC':
            ds = PADICDataset(root_path)
        elif name == 'CAUEEG':
            ds = CAUEEGDataset(root_path)
        elif name == 'TDBRAIN':
            ds = TDBRAINDataset(root_path)  # psychiatric, no dementia labels -> SSL/OOD use only
        elif name == 'GENEEG':
            ds = GENEEGDataset(root_path)   # external real-world MCI-vs-CN blind test set
        else:
            raise ValueError(f"Unknown dataset: {name}")
            
        self.datasets.append(ds)
        self.dataset_names.append(name)
        
    def create_torch_datasets(self, config) -> Tuple[ConcatDataset, np.ndarray, np.ndarray]:
        """
        Convert loaded loaders to Torch Datasets and concatenate.
        
        Returns:
            - Combined Dataset
            - Groups array (Subject IDs)
            - Labels array
        """
        import os
        # AUDIT (single-site / per-cohort): NCG_COHORT restricts training+eval to ONE cohort,
        # e.g. the OpenNeuro-only "clean" benchmark, or to try each dataset individually.
        # OpenNeuro = ds004504 + ds006036 (the same 88 people under two paradigms).
        _cohort = os.environ.get('NCG_COHORT', '').strip()
        _COHORT_TO_NAMES = {
            'OpenNeuro': {'ds004504', 'ds006036'},
            'AlzEEG': {'Alz_EEG'},
            'Mendeley': {'Mendeley'},
            'MCI_Dataset': {'MCI_Dataset'},
            'MISP': {'MISP'},
            'P-ADIC': {'P-ADIC'},
            'CAUEEG': {'CAUEEG'},   # the only all-4-class cohort -> the confound-free within-cohort benchmark (NCG-Bench P2)
        }
        if _cohort and _cohort not in _COHORT_TO_NAMES:
            raise ValueError(f"NCG_COHORT='{_cohort}' must be one of {sorted(_COHORT_TO_NAMES)}")
        if _cohort:
            print(f">>> NCG_COHORT={_cohort}: single-cohort mode "
                  f"(datasets {sorted(_COHORT_TO_NAMES[_cohort])})")

        torch_datasets = []
        groups = []
        labels = []

        for i, ds in enumerate(self.datasets):
            if _cohort and self.dataset_names[i] not in _COHORT_TO_NAMES[_cohort]:
                continue
            # Wrap standard loader into Torch Dataset (epochs/windows)
            # This wrapper handles slicing Raw into windows
            wrapper = EEGDatasetWrapper(ds, config)
            torch_datasets.append(wrapper)
            
            # Collect metadata for splitting
            # Wrapper should expose these
            groups.extend(wrapper.groups)
            labels.extend(wrapper.labels)
            
        combined = ConcatDataset(torch_datasets)
        return combined, np.array(groups), np.array(labels)

    @staticmethod
    def get_kfold_split(groups: np.ndarray, n_splits: int = 5, labels: np.ndarray = None,
                        seed: int = 42):
        """
        Return StratifiedGroupKFold iterator.
        Ensures strict subject separation AND class stratification.

        This prevents fold collapse where some folds have no samples
        from minority classes (MCI, FTD).

        ``seed`` controls the fold partition (StratifiedGroupKFold random_state).
        It defaults to 42 for backward compatibility; vary it across runs to obtain
        REPEATED subject-isolated CV (different fold assignments) for variance-controlled
        estimates -- e.g. 3x5 repeated CV via NCG_FOLD_SEED in 18_train_hierarchical.py.
        """
        if labels is not None:
            # Use Stratified Group KFold for balanced class distribution
            sgkf = StratifiedGroupKFold(n_splits=n_splits, shuffle=True, random_state=seed)
            dummy_X = np.zeros((len(groups), 1))
            return sgkf.split(dummy_X, labels, groups=groups)
        else:
            # Fallback to regular GroupKFold if no labels provided
            gkf = GroupKFold(n_splits=n_splits)
            dummy_X = np.zeros((len(groups), 1))
            return gkf.split(dummy_X, groups=groups)

    @staticmethod
    def get_holdout_split(groups: np.ndarray, labels: np.ndarray, test_size: float = 0.1, seed: int = 42):
        """
        Split groups into Train/Val and Hold-out Test set.
        Manually stratified by subject label to ensure all classes are present.
        """
        from sklearn.model_selection import StratifiedShuffleSplit
        
        # 1. Get unique subjects and their corresponding labels
        unique_subjs, subj_indices = np.unique(groups, return_index=True)
        subj_labels = labels[subj_indices] # Assumes label is constant per subject
        
        # 2. Stratified Split on SUBJECTS
        sss = StratifiedShuffleSplit(n_splits=1, test_size=test_size, random_state=seed)
        train_val_subj_idx, test_subj_idx = next(sss.split(unique_subjs, subj_labels))
        
        # Get the actual subject IDs
        train_val_subjects = unique_subjs[train_val_subj_idx]
        test_subjects = unique_subjs[test_subj_idx]
        
        # 3. Map back to full dataset indices
        # Vectorized check using isin
        train_val_mask = np.isin(groups, train_val_subjects)
        test_mask = np.isin(groups, test_subjects)
        
        train_val_idx = np.where(train_val_mask)[0]
        test_idx = np.where(test_mask)[0]
        
        return train_val_idx, test_idx
