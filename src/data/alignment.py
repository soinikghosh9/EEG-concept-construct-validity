"""
Cross-cohort EEG harmonization by covariance alignment (a generalized common space).

Euclidean Alignment (EA; He & Wu, IEEE TBME 2020) re-centers each recording's spatial
covariance to the identity, so recordings from heterogeneous cohorts/sites become
comparable in a shared space WITHOUT using labels -- which makes it confound-safe for
our setting (acquisition site is partially confounded with diagnosis, so label-driven
domain-invariance such as DANN would erase signal; EA does not).

Per recording with channel x time matrix X (19 x T):
    R = cov(X)                      # 19x19 spatial covariance
    X_aligned = R^{-1/2} X          # centers cov(X_aligned) at the identity

Congruence invariance: aligning by R^{-1/2} preserves the *relative* geometry within a
recording (within-recording distances unchanged) while removing the cross-recording offset.

Riemannian Alignment (RA) is the SPD-manifold analogue (re-center on the geometric mean);
EA is the simple, robust default. Selected via env NCG_ALIGN in {'', 'none', 'ea'}.
"""

import os
import numpy as np


def euclidean_align(data: np.ndarray, eps: float = 1e-6) -> np.ndarray:
    """Per-recording Euclidean Alignment. data: [n_channels, n_samples] -> aligned [n_channels, n_samples].

    Numerically stabilized: channels are mean-centred, the covariance is shrunk toward a
    scaled identity, eigenvalues are floored, and R^{-1/2} is formed from the symmetric
    eigendecomposition. NaN/Inf-safe.
    """
    X = np.asarray(data, dtype=np.float64)
    if X.ndim != 2:
        raise ValueError(f"euclidean_align expects [channels, samples], got {X.shape}")
    C, T = X.shape
    if T < 2:
        return X.astype(np.float32)

    Xc = X - X.mean(axis=1, keepdims=True)
    R = (Xc @ Xc.T) / (T - 1)                      # [C, C] spatial covariance
    tr = np.trace(R) / C
    if not np.isfinite(tr) or tr <= 0:
        return X.astype(np.float32)                 # degenerate recording: leave unchanged
    R = R + eps * tr * np.eye(C)                    # shrinkage regularization

    w, V = np.linalg.eigh(R)                         # R symmetric PSD
    w = np.clip(w, eps * tr, None)                  # floor eigenvalues for stability
    R_inv_sqrt = (V * (1.0 / np.sqrt(w))) @ V.T     # R^{-1/2}

    Xa = R_inv_sqrt @ X
    if not np.isfinite(Xa).all():
        return X.astype(np.float32)                 # fall back rather than propagate NaN/Inf
    return Xa.astype(np.float32)


def apply_alignment(data: np.ndarray, mode: str = None) -> np.ndarray:
    """Dispatch by mode (defaults to env NCG_ALIGN). mode in {None,'','none','ea'}."""
    if mode is None:
        mode = os.environ.get("NCG_ALIGN", "").strip().lower()
    if mode in ("", "none"):
        return data
    if mode == "ea":
        return euclidean_align(data)
    raise ValueError(f"Unknown NCG_ALIGN mode '{mode}' (expected 'ea' or 'none')")
