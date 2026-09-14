"""Rigorous equation of time from the DE-backed apparent place.

EoT itself is purely geometric -- apparent minus mean solar time, no Earth
rotation -- so the table needs no UT1 input.

But Chinh Ngo is that quantity expressed in CIVIL CLOCK time, and civil clocks
run on UTC while mean solar time runs on UT1. The formula therefore assumes
UT1 = UTC, which is stage 2's point: the residual is UT1 - UTC, measured over
19,955 IERS records at -0.676 .. +0.808 s. It is NOT baked in here, because it
is unknowable for future dates, undefined before UTC began in 1972, and this app
is offline with no way to fetch EOP. It is also ~60x below the minute at which
the app displays Chinh Ngo.

Meeus 28.3:  E = L0 - 0.0057183 deg - alpha + dpsi*cos(eps)
  L0     sun's mean longitude, mean equinox of date
  alpha  sun's APPARENT right ascension, true equinox of date
  0.0057183 deg = 20.5" , the aberration constant
  dpsi*cos(eps) reconciles the two frames (equation of the equinoxes in RA)
"""
import math
import erfa
from almanac_core import _equatorial

J2000 = 2451545.0
DEG = math.pi / 180.0


def eot_minutes(eph, jd1: float, jd2: float) -> float:
    """Equation of time in MINUTES (apparent minus mean), at TT jd1+jd2."""
    t = ((jd1 - J2000) + jd2) / 36525.0
    # Sun's mean longitude, Meeus 28.2 (same series the app's astro.js uses)
    l0 = (280.4664567 + 360007.6982779 * t / 10 + 0.03032028 * t * t / 100
          + (t ** 3) / 100 / 49931 - (t ** 4) / 10000 / 15300
          - (t ** 5) / 100000 / 2000000) % 360.0
    ra, _ = _equatorial(eph, jd1, jd2)
    dpsi, deps = erfa.nut06a(jd1, jd2)
    eps = erfa.obl06(jd1, jd2) + deps
    e = l0 - 0.0057183 - ra / DEG + (dpsi * math.cos(eps)) / DEG
    e = (e + 180.0) % 360.0 - 180.0
    return e * 4.0
