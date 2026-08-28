"""Lean reference core: new moon, full moon, jieqi, transit, sunrise/sunset.

This is the ~300 lines that decide whether you are right or wrong. Everything
in the long spec is refinement on top of this.

FOUR TRAPS. Get these right and you are at second-level accuracy:

  1. Light-time is ASYMMETRIC: body at emission time t-tau, observer at
     reception time t. Evaluating a geocentric vector at t-tau gives you
     body(t-tau) - Earth(t-tau), which is wrong by the Earth's motion over
     tau -- about 20.5" for the Sun, i.e. the aberration constant. Applying
     aberration on top then DOUBLE-COUNTS it: ~8 minutes of jieqi error,
     and the result still looks plausible.

  2. Aberration is ~20.5" and must be applied exactly once.

  3. TRUE ecliptic of date, not mean. Skipping nutation costs up to 17",
     which is ~35 s on a lunar phase and ~7 min on a jieqi.

  4. Earth is NAIF 399, not 3 (Earth-Moon barycentre). The 4670 km offset is
     ~6" of solar longitude, ~2.5 min of jieqi. A distance check will NOT
     catch this -- both give ~1 AU. The error is angular.

Scale factors worth memorising:
    1" of lunar elongation  ~ 2.0 s
    1" of solar longitude   ~ 24.4 s   (the Sun is 12x slower)

Ephemeris is pluggable. ErfaEphemeris runs anywhere pyerfa is installed and
is good to a few seconds on phases; De440Ephemeris is the production path.

    pip install pyerfa numpy          # runs as-is
    pip install calcephpy             # + de440s.bsp for production
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Protocol

import erfa
import numpy as np

DAY = 86400.0
CLIGHT_AU_D = erfa.DC  # speed of light, AU/day
AU_KM = 149597870.7
R_SUN_KM = 696000.0  # convention; IAU 2015 nominal is 695700
H0_STANDARD = math.radians(-50.0 / 60.0)  # -50', the almanac convention


# --------------------------------------------------------------------------
# Ephemeris
# --------------------------------------------------------------------------

class Ephemeris(Protocol):
    def earth_bary(self, jd1: float, jd2: float) -> tuple[np.ndarray, np.ndarray]:
        """Earth barycentric position (AU) and velocity (AU/day)."""

    def sun_bary(self, jd1: float, jd2: float) -> np.ndarray: ...

    def moon_geo(self, jd1: float, jd2: float) -> np.ndarray: ...


class ErfaEphemeris:
    """Low-precision fallback. Runs with no data file. Few-arcsecond Moon."""

    def earth_bary(self, jd1, jd2):
        pvh, pvb = erfa.epv00(jd1, jd2)
        return np.array(pvb[0]), np.array(pvb[1])

    def sun_bary(self, jd1, jd2):
        pvh, pvb = erfa.epv00(jd1, jd2)
        return np.array(pvb[0]) - np.array(pvh[0])  # SSB->Earth minus Sun->Earth

    def moon_geo(self, jd1, jd2):
        pv = erfa.moon98(jd1, jd2)
        return np.array(pv[0])


class De440Ephemeris:
    """Production path. Requires calcephpy and de440s.bsp."""

    def __init__(self, path: str):
        from calcephpy import CalcephBin
        self.p = CalcephBin.open(path)
        self.KM = None

    def _q(self, jd1, jd2, target, center):
        from calcephpy import Constants, NaifId
        pv = self.p.compute_unit(
            jd1, jd2, target, center,
            Constants.UNIT_AU + Constants.UNIT_DAY + Constants.USE_NAIFID,
        )
        return np.array(pv[:3]), np.array(pv[3:])

    def earth_bary(self, jd1, jd2):
        # TRAP 4: 399 (Earth centre), never 3 (Earth-Moon barycentre).
        return self._q(jd1, jd2, 399, 0)

    def sun_bary(self, jd1, jd2):
        return self._q(jd1, jd2, 10, 0)[0]

    def moon_geo(self, jd1, jd2):
        return self._q(jd1, jd2, 301, 399)[0]


# --------------------------------------------------------------------------
# Apparent place
# --------------------------------------------------------------------------

def _apparent_dir(eph: Ephemeris, body: str, jd1: float, jd2: float) -> np.ndarray:
    """Apparent geocentric unit direction in GCRS."""
    pe, ve = eph.earth_bary(jd1, jd2)

    tau = 0.0
    for _ in range(2):
        # TRAP 1: body retarded, observer at reception time.
        j2 = jd2 - tau
        if body == "sun":
            pb = eph.sun_bary(jd1, j2)
        else:
            pb = eph.moon_geo(jd1, j2) + eph.earth_bary(jd1, j2)[0]
        r = pb - pe
        tau = np.linalg.norm(r) / CLIGHT_AU_D

    pnat = r / np.linalg.norm(r)
    v = ve / CLIGHT_AU_D
    s = np.linalg.norm(pe)  # observer-Sun distance, AU (Sun ~ at SSB)
    bm1 = math.sqrt(max(0.0, 1.0 - float(v @ v)))
    # TRAP 2: aberration, exactly once.
    return np.array(erfa.ab(pnat, v, s, bm1))


def ecliptic_longitude(eph: Ephemeris, body: str, jd1: float, jd2: float) -> float:
    """Apparent longitude, TRUE equinox and ecliptic of date, radians."""
    v = erfa.pnm06a(jd1, jd2) @ _apparent_dir(eph, body, jd1, jd2)
    dpsi, deps = erfa.nut06a(jd1, jd2)
    eps = erfa.obl06(jd1, jd2) + deps  # TRAP 3: TRUE obliquity
    c, s = math.cos(eps), math.sin(eps)
    y = c * v[1] + s * v[2]
    return math.atan2(y, v[0]) % (2 * math.pi)


def _equatorial(eph: Ephemeris, jd1: float, jd2: float) -> tuple[float, float]:
    v = erfa.pnm06a(jd1, jd2) @ _apparent_dir(eph, "sun", jd1, jd2)
    return math.atan2(v[1], v[0]), math.asin(v[2])


def _wrap(a: float) -> float:
    return (a + math.pi) % (2 * math.pi) - math.pi


# --------------------------------------------------------------------------
# Solvers
# --------------------------------------------------------------------------

def _newton(f, t0: float, half: float, max_iter: int = 8) -> float:
    """Newton with numerical derivative. Returns TT JD fraction."""
    t = t0
    for _ in range(max_iter):
        f0 = f(t)
        d = (f(t + 0.01) - f(t - 0.01)) / 0.02
        step = -f0 / d
        step = max(-half, min(half, step))
        t += step
        if abs(step) < 1e-10:  # ~10 us
            break
    return t


def phase(eph: Ephemeris, k: float, jd1: float = 2451545.0) -> float:
    """New moon at integer k, full at k+0.5. Returns TT JD offset from jd1."""
    t0 = 2451550.09766 + 29.530588861 * k - jd1
    target = 0.0 if float(k).is_integer() else math.pi

    def f(t):
        lm = ecliptic_longitude(eph, "moon", jd1, t)
        ls = ecliptic_longitude(eph, "sun", jd1, t)
        return _wrap(lm - ls - target)

    return _newton(f, t0, 1.0)


def jieqi(eph: Ephemeris, lam_deg: float, near_jd2: float,
          jd1: float = 2451545.0) -> float:
    """Solar term at apparent longitude lam_deg, nearest near_jd2 (TT)."""
    target = math.radians(lam_deg % 360.0)

    def f(t):
        return _wrap(ecliptic_longitude(eph, "sun", jd1, t) - target)

    return _newton(f, near_jd2, 3.0)


def jieqi_seed(lam_deg: float, year: int, jd1: float = 2451545.0) -> float:
    """Mean-longitude seed for the term of `lam_deg` falling inside `year`.

    Good to ~2 days: the mean sun leads or lags the true sun by at most the
    equation of centre, ~1.94 d, which is inside the +-3 d step clamp of the
    Newton solver.

    FIX vs the published reference, which read:

        d = (lam_deg - 280.46646) / 0.9856474
        n = math.floor((year - 2000) * 365.2422 / 365.2422 + 0.5)   # unused
        return d + 365.2422 * (year - 2000)

    That measured the longitude from J2000 and then added a whole number of
    tropical years, so for every longitude below 280.47 deg -- which is 22 of
    the 24 terms -- it landed in the PREVIOUS December: 365.7 d off for the
    winter solstice, 363.4 d for the vernal equinox. The solver clamps each
    step to 3 d over 8 iterations, i.e. 24 d of reach, so it cannot recover;
    it silently converges on a different term or on nothing at all. The `n`
    line was dead code and hints at an intended year-disambiguation that was
    never finished.

    Walking forward from 1 January of `year` keeps the seed inside the
    requested year by construction.
    """
    d1, d2 = erfa.cal2jd(year, 1, 1)
    d0 = (d1 - jd1) + d2
    lam0 = (280.46646 + 0.9856474 * d0) % 360.0
    return d0 + ((lam_deg - lam0) % 360.0) / 0.9856474


# --------------------------------------------------------------------------
# Topocentric
# --------------------------------------------------------------------------

@dataclass
class Observer:
    lat_deg: float
    lon_deg_east: float
    height_m: float = 0.0  # ELLIPSOIDAL, not MSL


def _hour_angle(eph, jd1, jd2, obs, dut1=0.0):
    ut = jd2 - 32.184 / DAY + dut1 / DAY  # TT -> UT1 (approx; use real dUT1)
    gast = erfa.gst06a(jd1, ut, jd1, jd2)
    ra, dec = _equatorial(eph, jd1, jd2)
    return _wrap(gast + math.radians(obs.lon_deg_east) - ra), dec


def transit(eph: Ephemeris, obs: Observer, jd2_guess: float,
            jd1: float = 2451545.0, dut1: float = 0.0) -> float:
    """Local apparent noon (TT). Parallax and refraction do not affect it:
    at the meridian both act along the meridian, not across it."""
    t = jd2_guess
    for _ in range(4):
        h, _ = _hour_angle(eph, jd1, t, obs, dut1)
        t -= h / (2 * math.pi * 1.0027379)  # ~sidereal rate, days per radian
    return t


def _altitude(eph, jd1, jd2, obs, dut1):
    h, dec = _hour_angle(eph, jd1, jd2, obs, dut1)
    phi = math.radians(obs.lat_deg)
    return math.asin(math.sin(phi) * math.sin(dec)
                     + math.cos(phi) * math.cos(dec) * math.cos(h))


def rise_set(eph: Ephemeris, obs: Observer, jd2_midnight: float,
             jd1: float = 2451545.0, dut1: float = 0.0, h0: float = H0_STANDARD):
    """Returns (sunrise, sunset) as TT JD offsets, or None where absent.

    Solve GEOMETRIC altitude against h0, which already contains refraction and
    semidiameter. Never refract the altitude AND use a refraction-bearing h0.

    Scan-then-bisect, not Newton: altitude is not monotonic and its derivative
    vanishes near the horizon at high latitude.
    """
    step = 1.0 / 144.0  # 10 min
    ts = [jd2_midnight + i * step for i in range(145)]
    fs = [_altitude(eph, jd1, t, obs, dut1) - h0 for t in ts]

    events = []
    for i in range(len(ts) - 1):
        if fs[i] == 0.0 or fs[i] * fs[i + 1] < 0:
            lo, hi = ts[i], ts[i + 1]
            for _ in range(40):
                mid = 0.5 * (lo + hi)
                if (_altitude(eph, jd1, lo, obs, dut1) - h0) * \
                   (_altitude(eph, jd1, mid, obs, dut1) - h0) <= 0:
                    hi = mid
                else:
                    lo = mid
            events.append((0.5 * (lo + hi), fs[i + 1] > fs[i]))

    rise = next((t for t, up in events if up), None)
    sett = next((t for t, up in events if not up), None)
    return rise, sett


def jd_to_iso(jd1: float, jd2: float) -> str:
    y, m, d, f = erfa.jd2cal(jd1, jd2)
    h = f * 24
    mi = (h - int(h)) * 60
    return f"{y:04d}-{m:02d}-{d:02d} {int(h):02d}:{int(mi):02d}:{(mi - int(mi)) * 60:05.2f}"
