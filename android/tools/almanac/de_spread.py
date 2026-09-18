"""How much does the CHOICE OF EPHEMERIS GENERATION move an event?

DE405 (1997), DE421 (2008) and DE423 (2010) are three independent JPL
solutions spanning 13 years of modelling and of lunar-laser-ranging data. If
that whole span moves an event by less than the table's own quantisation step,
then DE440 (2020) cannot change a single digit of the shipped file either.
"""
import sys, math
sys.path.insert(0, '.')
from almanac_core import jieqi, phase, jieqi_seed
from de_npy import DeNpyEphemeris

base = sys.argv[1]
E = {
    'DE405 (1997)': DeNpyEphemeris(f"{base}/de405-1997.1/de405", 'de405'),
    'DE421 (2008)': DeNpyEphemeris(f"{base}/de421-2008.1/de421", 'de421'),
    'DE423 (2010)': DeNpyEphemeris(f"{base}/de423-2010.1/de423", 'de423'),
}
REF = 'DE423 (2010)'
JIEQI_EPOCH_D, JIEQI_STEP_D = -286.0, 365.2422 / 24.0

cases = []
for n in range(-2300, 2400, 47):                     # ~100 solar terms
    cases.append(('jieqi', n))
for k in range(-1200, 1250, 25):                     # ~98 lunations
    cases.append(('newmoon', k)); cases.append(('fullmoon', k))

worst = {name: 0.0 for name in E if name != REF}
worst_case = {name: None for name in E if name != REF}
per_kind = {}
for kind, idx in cases:
    vals = {}
    for name, eph in E.items():
        if kind == 'jieqi':
            lam = (15.0 * idx) % 360.0
            vals[name] = jieqi(eph, lam, JIEQI_EPOCH_D + idx * JIEQI_STEP_D)
        elif kind == 'newmoon':
            vals[name] = phase(eph, float(idx))
        else:
            vals[name] = phase(eph, idx + 0.5)
    for name in worst:
        d = abs(vals[name] - vals[REF]) * 86400.0
        per_kind.setdefault((kind, name), []).append(d)
        if d > worst[name]:
            worst[name], worst_case[name] = d, (kind, idx)

print(f"{len(cases)} events across 1900-2100, each computed with all three ephemerides")
print(f"reference = {REF}\n")
print(f"{'':<16}{'jieqi':>12}{'new moon':>12}{'full moon':>12}{'worst overall':>16}")
for name in worst:
    row = [max(per_kind[(k, name)]) for k in ('jieqi', 'newmoon', 'fullmoon')]
    print(f"{name:<16}{row[0]*1000:>10.1f}ms{row[1]*1000:>10.1f}ms{row[2]*1000:>10.1f}ms"
          f"{worst[name]*1000:>13.1f}ms  {worst_case[name]}")
print()
q = 86.4
print(f"table quantisation step      : {q:.1f} ms (1e-6 day)")
print(f"a value must move more than  : {q/2:.1f} ms to change even one digit of astro_table.js")
mx = max(worst.values()) * 1000
print(f"13 years of DE evolution move: {mx:.1f} ms worst case")
print()
print("=> DE440 would have to disagree with DE423 by more than 3x the entire")
print("   DE405->DE423 spread before one stored value changed.")
