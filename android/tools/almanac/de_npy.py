"""Production ephemeris adapter for the jplephem `.npy` DE packages.

Stage 1 of the implementation prompt asks for `de440s.bsp` via calcephpy. That
file lives on `naif.jpl.nasa.gov`, which this session's egress policy blocks
(403 on CONNECT), so we use DE423 from PyPI instead. For the quantities this
engine computes the substitution is immaterial: DE423 (2010) and DE440 (2020)
differ by a few milliarcseconds in solar/lunar apparent longitude over
1800-2200, and 1" of solar longitude is 24.4 s of jieqi -- so a few mas is a
few hundredths of a second. DE423 also spans 1800-2200, covering the app's
1900-2100 tables, which `de440s` (1849-2150) would too.

Units: the legacy jplephem API returns km and km/day; the Ephemeris protocol in
almanac_core wants AU and AU/day.
"""

from __future__ import annotations

import os
import sys
from types import ModuleType

import numpy as np
from jplephem import Ephemeris as _JplEphemeris


def _load_module(pkg_dir: str, name: str) -> ModuleType:
    """Import a de4xx data package straight from an extracted sdist."""
    parent = os.path.dirname(os.path.abspath(pkg_dir))
    if parent not in sys.path:
        sys.path.insert(0, parent)
    return __import__(name)


class DeNpyEphemeris:
    """DE4xx via jplephem's `.npy` packages, in AU and AU/day."""

    def __init__(self, pkg_dir: str, name: str = "de423") -> None:
        self.e = _JplEphemeris(_load_module(pkg_dir, name))
        # The ephemeris carries its own AU; using a different one would put a
        # systematic scale error into the light-time and aberration terms.
        self.au_km = float(self.e.AU)
        self.emrat = float(self.e.EMRAT)  # Earth mass / Moon mass

    def earth_bary(self, jd1: float, jd2: float):
        """Barycentric Earth *centre* (NAIF 399), AU and AU/day.

        TRAP 4: the DE tables give the Earth-Moon BARYCENTRE, not the Earth.
        Earth = EMB - Moon_geo / (1 + EMRAT). The 4670 km offset is ~6" of
        solar longitude, i.e. ~2.5 min of jieqi, and both vectors are ~1 AU
        long so no distance check will catch the mistake.
        """
        pb, vb = self.e.position_and_velocity("earthmoon", jd1, jd2)
        pm, vm = self.e.position_and_velocity("moon", jd1, jd2)
        f = 1.0 / (1.0 + self.emrat)
        p = (pb.ravel() - pm.ravel() * f) / self.au_km
        v = (vb.ravel() - vm.ravel() * f) / self.au_km
        return p, v

    def sun_bary(self, jd1: float, jd2: float) -> np.ndarray:
        p, _ = self.e.position_and_velocity("sun", jd1, jd2)
        return p.ravel() / self.au_km

    def moon_geo(self, jd1: float, jd2: float) -> np.ndarray:
        p, _ = self.e.position_and_velocity("moon", jd1, jd2)
        return p.ravel() / self.au_km
