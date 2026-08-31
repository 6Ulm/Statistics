# Privacy

SPEC.md §22.3 is normative. §37 rule 10: never commit secrets, API keys, device identifiers or
raw private locations.

## What the app handles

Precise location is the sensitive category. The heading pipeline needs a fresh fix to resolve the
north reference and evaluate the geomagnetic model (§10, §11), so location is intrinsic to
correctness rather than an add-on — which makes minimization, not avoidance, the design question.

Also collected during a measurement: magnetometer and motion samples, device and sensor runtime
descriptors, OS build identity, thermal and charging state, and provider version identities. §24
requires these in the certification key, so they are not optional for a calibrated claim.

## Rules

- **Consent and minimization.** Obtain consent, minimize retention, and keep lab telemetry
  separate from consumer analytics.
- **Production export is opt-in** and MUST redact or quantize coordinates where exact values are
  unnecessary.
- **Two salts, two code paths.** Consumer builds hash device identifiers with a **rotating**
  project salt and MUST NOT log advertising identifiers. Benchmark builds use a **fixed archived**
  salt so per-unit longitudinal analysis remains possible (§28). §22.3 is explicit that these
  "MUST be separate code paths, not one path with a flag that could ship in the wrong position" —
  a flag is one misconfiguration away from shipping a fixed salt to consumers.
- **Integrity.** Sign manifests and hash raw files so analysis can prove inputs were not edited.
- **Blinding.** Keep test labels separate until blind analysis completes where practical.
- **No background location**, ever (§13). Lifecycle is foreground only (§2).
- **No network is required for heading after install** (§2). Space weather is a cached,
  nonblocking advisory; losing it degrades to `UNKNOWN`, never to `QUIET` and never to zero.

## Permission handling

§13 and §12: request coarse and fine per current platform rules; detect approximate-only grants
including runtime downgrades that restart the process; treat one-time grants as expiring and
re-check rather than caching a past "granted" state; detect disabled services and API
unavailability. Reduced authorization is a **provenance label, not automatic failure** — it is
accepted when its declination envelope stays inside the grade budget (§18.5), and the label is
always shown and logged.

## Status in Phase 0

No telemetry is written and no location is read: no provider adapters exist. What exists is the
schema shape that will carry this data (`schemas/telemetry-event-v1.schema.json`,
`schemas/session-manifest-v1.schema.json`) and one example event whose `deviceAnonymousId` is a
SHA-256 of a documented `FIXTURE:` string, not a real identifier. No real coordinate, device
identifier or salt appears anywhere in this repository.

The separate consumer and benchmark salt paths are a **Phase 3** obligation, landing with the
JSONL telemetry sink.
