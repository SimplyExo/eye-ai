# EyeAI ByteTrack binding

This directory vendors `Peanutt42/bytetrack-cpp-rs` v0.1.1 and its pinned
`Peanutt42/ByteTrack-cpp` v1.1.1 source so EyeAI's native timing contract can
be reviewed and built from the application repository. The original licenses
are retained in this directory and in `ByteTrack-cpp/`.

EyeAI's local delta is intentionally limited to variable monotonic elapsed
time, real-time lost-track expiry, and the corresponding variable-time Kalman
prediction/covariance model. Detection thresholds and association heuristics
remain those of the vendored version.
