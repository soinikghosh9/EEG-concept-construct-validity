"""Electrode coordinates for the EasyCog forehead-and-ear wearable array.

The array is four forehead electrodes in a headband plus two six-electrode cEEGrid strips,
one around each ear, on an OpenBCI Cyton+Daisy at 125 Hz. Channel order in the released
``.npz`` files follows ``STD_CH`` below, taken from the EasyCog benchmark repository
(``data_processing/analysis_utils.py``). That order is not stated in the dataset readme; it
is corroborated independently by the repository's own EOG derivation, which subtracts array
indices 3 and 11 and names the result ``FR1 - FR4``, the outermost forehead pair.

Forehead identities are given in the dataset paper (figure 5): FR1=AF8, FR2=Fp2, FR3=Fp1,
FR4=AF7. Only Fp1 and Fp2 coincide with a position in our 19-channel training montage.

**The ear coordinates are a reconstruction, not a measurement.** The repository refers to a
``cEEGrid_sensor.xlsx`` that is not distributed, so the twelve ear positions here are placed
from the repository's own 2-D layout table ``STD_CH_DIST``, anchored on an ear centre
slightly posterior and inferior to T3/T4. ``region_occupancy()`` below reports how sensitive
the resulting region assignment is to that anchor, and it is sensitive: see the module test.

Convention matches ``src.config.config.CHANNEL_COORDS_10_20`` -- a unit sphere with x
left-negative, y posterior-negative, z inferior-negative.
"""
from __future__ import annotations

from typing import Dict, Tuple

import numpy as np

#: Channel order of ``raw_eeg`` in the released files.
STD_CH = ['R1', 'R2', 'R3', 'FR1', 'FR2', 'R8', 'R9', 'R10',
          'L1', 'L2', 'L3', 'FR4', 'FR3', 'L8', 'L9', 'L10']

#: The repository's schematic 2-D layout, unrolled right ear -> forehead -> left ear.
#: (x increases away from the right ear and toward the left; y is superior-positive.)
_LAYOUT: Dict[str, Tuple[int, int]] = {
    'R1': (0, 0), 'R2': (-1, 1), 'R3': (-2, 0), 'R8': (-2, -2), 'R9': (-1, -3), 'R10': (0, -2),
    'FR1': (3, 2), 'FR2': (4, 2), 'FR3': (5, 2), 'FR4': (6, 2),
    'L1': (9, 0), 'L2': (10, 1), 'L3': (11, 0), 'L8': (11, -2), 'L9': (10, -3), 'L10': (9, -2),
}


def _unit(v) -> Tuple[float, float, float]:
    v = np.asarray(v, float)
    return tuple(float(x) for x in v / np.linalg.norm(v))


#: Forehead electrodes at their 10-20 positions. AF7/AF8 are interpolated between Fp and
#: F7/F8 and renormalised onto the sphere; they are real 10-20 sites but are not present in
#: our 19-channel montage, so they have no entry in CHANNEL_COORDS_10_20 to copy.
FOREHEAD: Dict[str, Tuple[float, float, float]] = {
    'FR3': (-0.30, 0.95, -0.03),           # Fp1
    'FR2': (0.30, 0.95, -0.03),            # Fp2
    'FR4': _unit((-0.555, 0.77, -0.03)),   # AF7
    'FR1': _unit((0.555, 0.77, -0.03)),    # AF8
}

#: Anchor of the cEEGrid strips. The ear canal defines y=0 in the 10-20 frame, and the strip
#: sits a little behind and well below it. DY/DZ convert one layout step into sphere units.
EAR_CENTRE_Y, EAR_CENTRE_Z = -0.10, -0.55
DY, DZ = 0.060, 0.070
_LAYOUT_XC = {'L': 10.0, 'R': -1.0}
_LAYOUT_YC = -1.0


def easycog_coords(ear_centre_y: float = EAR_CENTRE_Y,
                   dy: float = DY) -> Dict[str, Tuple[float, float, float]]:
    """Coordinate table for the sixteen electrodes.

    ``ear_centre_y`` and ``dy`` are exposed because the ear positions are reconstructed:
    varying them is how the sensitivity of any regional concept to that reconstruction is
    measured, and they should be varied before a regional result is believed.
    """
    out: Dict[str, Tuple[float, float, float]] = dict(FOREHEAD)
    for ch, (lx, ly) in _LAYOUT.items():
        if ch.startswith('FR'):
            continue
        side = ch[0]
        # On the left strip a larger layout-x is further from the forehead, so more
        # posterior; on the right strip the sign reverses, because the layout unrolls
        # right ear -> front -> left ear.
        sgn = -1.0 if side == 'L' else +1.0
        y = ear_centre_y + sgn * (lx - _LAYOUT_XC[side]) * dy
        z = EAR_CENTRE_Z + (ly - _LAYOUT_YC) * DZ
        out[ch] = ((-0.95 if side == 'L' else 0.95), y, z)
    return out


EASYCOG_COORDS = easycog_coords()


def region_occupancy(coords: Dict[str, Tuple[float, float, float]] = None,
                     channels=None) -> Dict[str, list]:
    """Which electrodes fall in each of the extractor's regions, by name.

    Wraps ``src.models.v6.concepts.region_masks`` so a montage can be checked before it is
    handed to the extractor. A region that comes back empty makes every concept defined over
    it return NaN, which is the safe outcome. A region that comes back *populated by
    electrodes nowhere near the territory the concept names* is the unsafe one, and this
    function is what makes that visible.
    """
    import torch

    from src.models.v6.concepts import region_masks

    coords = coords if coords is not None else EASYCOG_COORDS
    channels = list(channels) if channels is not None else STD_CH
    C = torch.tensor([coords[c] for c in channels], dtype=torch.float32)
    m = region_masks(C)
    return {k: [channels[i] for i in range(len(channels)) if bool(v[i])] for k, v in m.items()}
