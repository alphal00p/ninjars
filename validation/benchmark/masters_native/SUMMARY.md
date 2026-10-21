# oneloopmaster native masters vs OneLOop Fortran (identical call streams)

Native: oneloopmaster's direct ahead-of-time Rust backend (`NativeEvaluator<f64>`, feature
`generated-evaluators`; no expression interpreter, no SymJIT), one scalar `evaluate` per call,
NinjaRS release profile (thin LTO, 1 CGU). OneLOop: the same calls (`../ninja/<case>.masters.json`,
recorded from original Ninja) replayed through `ninja::avh_olo` (build `o3-fortranrules`, OneLOop-3.6,
gfortran -O3, OneLOop warnings off). Microseconds per phase-space point (one lane's calls),
median of >=7 blocks >=25 ms. OneLOop replay and olm-264a6ce native pinned to CPU 250 (2026-10-18),
olm-3023fa6 native on CPU 250 (2026-10-19; earlier labels: native on CPU 29, see
../cpu-crosscheck.md); olm-8ec256fc native one call stream per process, each pinned to one CPU of
4-93 whose SMT sibling was at most 5 % busy (2026-10-21; per stream the most representative of three
rounds, see `runs` in its results file). 'OL B' is every bubble call (olo_b11 for
Ninja's rank-2 bubbles: B11, B00, B1, B0); 'nat B0 only' evaluates just B0 for those calls
(less work), 'nat B (tensor)' evaluates NativeTensorEvaluator B11 = [B0, B1, B11] (B111 for
rank-3 bubbles), i.e. everything Ninja consumes. max diff = largest |native - OneLOop| / max|OneLOop|
over each call's outputs; relabel = largest deviation of native C0/D0 under the cyclic and
reflected relabellings of the propagators (a native self-consistency check).

## `olm-8ec256fc-sym-9cdc2949-tensor`

oneloopmaster main 8ec256fc (09b2c74 plus the contour box near vanishing Gram determinants with Vieta pole-zero offsets and a second contour, Denner-Dittmaier sheets for one- and two-mass boxes with complex masses, two-mass mass-pair complements from the channels and the soft-box relabelling at the numerical D0 entry points) + Symbolica 9cdc2949 (e679e8c1 plus symbolica-dev/symbolica#97, Float products rounded once); the current benchmark revision

| case | OneLOop all | native all | native/OneLOop | OL D0 | nat D0 | OL C0 | nat C0 | OL B | nat B (tensor) | OL A0 | nat A0 | max diff | relabel |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| four-photon-top-box-0 | 10.13 | 5.33 | 0.53x | 3.79 | 2.42 | 4.78 | 2.18 | 1.33 | 0.26 | 0.16 | 0.14 | 8.3e-14 | 5e-15 |
| gg-hhh-top-box-8 | 22.49 | 12.31 | 0.55x | 13.42 | 8.79 | 5.43 | 2.79 | 3.17 | 0.34 | 0.16 | 0.14 | 4.1e-15 | 7e-15 |
| gg-hhh-top-pentagon-26 | 98.22 | 52.08 | 0.53x | 61.85 | 39.63 | 26.28 | 11.43 | 7.28 | 0.63 | 0.20 | 0.17 | 1.1e-14 | 2e-14 |
| gg-hhh-top-triangle-0 | 1.27 | 0.57 | 0.45x | - | - | 0.67 | 0.35 | 0.50 | 0.12 | 0.13 | 0.10 | 4.6e-16 | 2e-15 |
| gg-hh-top-box-2 | 27.76 | 13.01 | 0.47x | 16.58 | 9.17 | 6.67 | 3.10 | 3.93 | 0.35 | 0.16 | 0.14 | 1.7e-14 | 4e-15 |
| gg-hh-top-triangle-0 | 1.90 | 0.66 | 0.35x | - | - | 0.91 | 0.42 | 0.87 | 0.13 | 0.13 | 0.10 | 4.6e-16 | 3e-15 |
| hgg-top-triangle-0 | 6.53 | 1.55 | 0.24x | - | - | 4.80 | 1.30 | 1.62 | 0.13 | 0.13 | 0.10 | 4.5e-15 | 7e-15 |
| madloop-four-photon-top-box-0 | 9.13 | 4.95 | 0.54x | 3.85 | 2.37 | 3.66 | 1.83 | 1.36 | 0.25 | 0.17 | 0.14 | 5.0e-15 | 5e-15 |
| madloop-gg-hhh-top-box-8 | 27.40 | 14.22 | 0.52x | 13.26 | 8.81 | 10.38 | 4.64 | 3.16 | 0.34 | 0.16 | 0.14 | 2.4e-15 | 7e-15 |
| madloop-gg-hhh-top-pentagon-26 | 107.47 | 54.37 | 0.51x | 63.62 | 39.53 | 33.44 | 13.72 | 7.22 | 0.63 | 0.20 | 0.17 | 1.4e-14 | 2e-14 |
| madloop-gg-hhh-top-triangle-0 | 1.30 | 0.57 | 0.44x | - | - | 0.68 | 0.34 | 0.49 | 0.12 | 0.13 | 0.10 | 4.6e-16 | 2e-15 |
| madloop-gg-hh-top-box-2 | 33.21 | 14.69 | 0.44x | 16.48 | 9.17 | 11.43 | 4.77 | 3.94 | 0.35 | 0.17 | 0.14 | 1.7e-14 | 5e-15 |
| madloop-gg-hh-top-triangle-0 | 1.89 | 0.65 | 0.35x | - | - | 0.90 | 0.42 | 0.86 | 0.13 | 0.13 | 0.10 | 4.6e-16 | 3e-15 |
| madloop-hgg-top-triangle-0 | 3.24 | 0.83 | 0.26x | - | - | 1.49 | 0.58 | 1.63 | 0.14 | 0.13 | 0.10 | 2.8e-15 | 7e-15 |
| syn-gghhh-ggHHH-n5-r4 | 117.19 | 57.37 | 0.49x | 69.13 | 41.77 | 36.53 | 14.34 | 8.03 | 0.65 | 0.20 | 0.17 | 1.8e-14 | 2e-14 |
| syn-gghhh-ggHHH-n5-r5 | 118.36 | 57.91 | 0.49x | 69.96 | 41.69 | 36.25 | 14.87 | 8.09 | 0.65 | 0.21 | 0.17 | 3.0e-14 | 2e-14 |
| syn-gghhh-gHgHH-n5-r4 | 64.24 | 26.36 | 0.41x | 24.66 | 12.86 | 30.68 | 12.47 | 6.82 | 0.65 | 0.21 | 0.18 | 1.8e-14 | 2e-14 |
| syn-gghhh-gHgHH-n5-r5 | 62.27 | 26.00 | 0.42x | 23.98 | 12.79 | 29.18 | 12.15 | 6.83 | 0.65 | 0.20 | 0.17 | 1.7e-14 | 3e-14 |
| syn-gghhh-nosmat-n5-r4 | 117.56 | 57.31 | 0.49x | 67.91 | 41.71 | 37.66 | 14.36 | 8.01 | 0.65 | 0.20 | 0.17 | 1.1e-14 | 1e-14 |
| syn-n2-r1 | 0.21 | 0.10 | 0.48x | - | - | - | - | 0.16 | 0.06 | 0.06 | 0.04 | 1.9e-14 | 0e+00 |
| syn-n2-r2 | 0.20 | 0.11 | 0.55x | - | - | - | - | 0.15 | 0.08 | 0.06 | 0.03 | 1.2e-15 | 0e+00 |
| syn-n2-r3 | 0.30 | 0.11 | 0.36x | - | - | - | - | 0.24 | 0.07 | 0.05 | 0.03 | 5.0e-13 | 0e+00 |
| syn-n3-r2 | 4.27 | 1.59 | 0.37x | - | - | 2.94 | 1.17 | 1.09 | 0.28 | 0.24 | 0.15 | 1.7e-14 | 2e-16 |
| syn-n3-r3 | 3.87 | 1.85 | 0.48x | - | - | 2.56 | 1.43 | 1.09 | 0.28 | 0.24 | 0.14 | 1.7e-14 | 2e-16 |
| syn-n3-r4 | 5.08 | 1.34 | 0.26x | - | - | 3.36 | 0.90 | 1.45 | 0.29 | 0.24 | 0.14 | 1.7e-14 | 2e-16 |
| syn-n4-r3 | 16.91 | 12.06 | 0.71x | 7.82 | 3.79 | 6.45 | 7.63 | 2.41 | 0.32 | 0.13 | 0.10 | 7.9e-14 | 5e-15 |
| syn-n4-r4 | 15.86 | 13.57 | 0.86x | 9.18 | 5.36 | 5.08 | 7.52 | 1.40 | 0.35 | 0.13 | 0.10 | 2.4e-14 | 1e-14 |
| syn-n4-r5 | 12.19 | 10.77 | 0.88x | 5.24 | 3.36 | 4.60 | 6.71 | 2.09 | 0.33 | 0.13 | 0.10 | 1.1e-14 | 4e-15 |
| syn-n5-r4 | 68.68 | 35.15 | 0.51x | 39.60 | 24.22 | 21.35 | 9.89 | 5.36 | 0.62 | 0.20 | 0.17 | 4.6e-14 | 3e-14 |
| syn-n5-r5 | 75.41 | 36.06 | 0.48x | 41.81 | 23.26 | 25.61 | 11.67 | 5.53 | 0.60 | 0.21 | 0.17 | 4.5e-13 | 4e-13 |
| syn-n5-r6 | 91.88 | 45.13 | 0.49x | 49.66 | 30.50 | 31.47 | 13.44 | 7.85 | 0.68 | 0.21 | 0.17 | 1.1e-14 | 3e-14 |
| syn-n6-r5 | 494.69 | 302.98 | 0.61x | 407.86 | 230.09 | 56.99 | 45.99 | 9.35 | 1.91 | 0.33 | 0.21 | 4.0e-13 | 9e-14 |
| syn-n6-r6 | 428.05 | 245.30 | 0.57x | 346.35 | 186.99 | 59.77 | 45.84 | 8.95 | 1.72 | 0.33 | 0.21 | 7.3e-13 | 8e-14 |
| syn-n6-r7 | 540.80 | 316.18 | 0.58x | 439.25 | 250.92 | 72.23 | 46.49 | 12.27 | 2.02 | 0.32 | 0.21 | 2.5e-09 | 1e-13 |

## `olm-3023fa6-sym-e679e8c1-tensor`

oneloopmaster main 3023fa6 (264a6ce plus the B1111 tensor family, conditioned C0 dilogarithm differences, the contour box for real three-mass boxes, exact Kallen edge roots, far-root r1, labelling-independent real four-mass D0, collinear few-mass boxes and commutative DoubleFloat products) + Symbolica e679e8c1; the previous benchmark revision (149ddff and 09b2c74 return the same bits on every call)

| case | OneLOop all | native all | native/OneLOop | OL D0 | nat D0 | OL C0 | nat C0 | OL B | nat B (tensor) | OL A0 | nat A0 | max diff | relabel |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| four-photon-top-box-0 | 10.13 | 5.38 | 0.53x | 3.79 | 2.42 | 4.78 | 2.31 | 1.33 | 0.26 | 0.16 | 0.14 | 8.3e-14 | 5e-15 |
| gg-hhh-top-box-8 | 22.49 | 12.67 | 0.56x | 13.42 | 8.81 | 5.43 | 2.90 | 3.17 | 0.35 | 0.16 | 0.14 | 4.1e-15 | 7e-15 |
| gg-hhh-top-pentagon-26 | 98.22 | 53.62 | 0.55x | 61.85 | 41.08 | 26.28 | 11.80 | 7.28 | 0.64 | 0.20 | 0.18 | 1.1e-14 | 2e-14 |
| gg-hhh-top-triangle-0 | 1.27 | 0.60 | 0.48x | - | - | 0.67 | 0.36 | 0.50 | 0.13 | 0.13 | 0.11 | 4.6e-16 | 2e-15 |
| gg-hh-top-box-2 | 27.76 | 13.85 | 0.50x | 16.58 | 9.37 | 6.67 | 3.26 | 3.93 | 0.37 | 0.16 | 0.14 | 1.5e-14 | 4e-15 |
| gg-hh-top-triangle-0 | 1.90 | 0.70 | 0.37x | - | - | 0.91 | 0.44 | 0.87 | 0.13 | 0.13 | 0.11 | 4.6e-16 | 3e-15 |
| hgg-top-triangle-0 | 6.53 | 1.61 | 0.25x | - | - | 4.80 | 1.34 | 1.62 | 0.14 | 0.13 | 0.11 | 4.5e-15 | 7e-15 |
| madloop-four-photon-top-box-0 | 9.13 | 5.17 | 0.57x | 3.85 | 2.37 | 3.66 | 1.95 | 1.36 | 0.27 | 0.17 | 0.14 | 5.0e-15 | 5e-15 |
| madloop-gg-hhh-top-box-8 | 27.40 | 14.69 | 0.54x | 13.26 | 8.66 | 10.38 | 4.68 | 3.16 | 0.35 | 0.16 | 0.14 | 2.4e-15 | 7e-15 |
| madloop-gg-hhh-top-pentagon-26 | 107.47 | 55.04 | 0.51x | 63.62 | 40.52 | 33.44 | 14.40 | 7.22 | 0.65 | 0.20 | 0.18 | 1.4e-14 | 2e-14 |
| madloop-gg-hhh-top-triangle-0 | 1.30 | 0.58 | 0.45x | - | - | 0.68 | 0.35 | 0.49 | 0.13 | 0.13 | 0.11 | 4.6e-16 | 2e-15 |
| madloop-gg-hh-top-box-2 | 33.21 | 15.32 | 0.46x | 16.48 | 9.24 | 11.43 | 5.00 | 3.94 | 0.37 | 0.17 | 0.14 | 1.5e-14 | 5e-15 |
| madloop-gg-hh-top-triangle-0 | 1.89 | 0.69 | 0.37x | - | - | 0.90 | 0.43 | 0.86 | 0.13 | 0.13 | 0.10 | 4.6e-16 | 3e-15 |
| madloop-hgg-top-triangle-0 | 3.24 | 0.86 | 0.27x | - | - | 1.49 | 0.61 | 1.63 | 0.14 | 0.13 | 0.11 | 2.8e-15 | 7e-15 |
| syn-gghhh-ggHHH-n5-r4 | 117.19 | 60.45 | 0.52x | 69.13 | 42.19 | 36.53 | 14.83 | 8.03 | 0.66 | 0.20 | 0.18 | 1.7e-14 | 2e-14 |
| syn-gghhh-ggHHH-n5-r5 | 118.36 | 60.62 | 0.51x | 69.96 | 43.47 | 36.25 | 15.18 | 8.09 | 0.66 | 0.21 | 0.17 | 3.6e-11 | 2e-14 |
| syn-gghhh-gHgHH-n5-r4 | 64.24 | 27.83 | 0.43x | 24.66 | 13.26 | 30.68 | 12.77 | 6.82 | 0.67 | 0.21 | 0.18 | 1.8e-14 | 2e-14 |
| syn-gghhh-gHgHH-n5-r5 | 62.27 | 28.18 | 0.45x | 23.98 | 13.31 | 29.18 | 12.77 | 6.83 | 0.66 | 0.20 | 0.17 | 1.7e-14 | 3e-14 |
| syn-gghhh-nosmat-n5-r4 | 117.56 | 58.53 | 0.50x | 67.91 | 43.73 | 37.66 | 14.60 | 8.01 | 0.65 | 0.20 | 0.18 | 1.1e-14 | 1e-14 |
| syn-n2-r1 | 0.21 | 0.10 | 0.50x | - | - | - | - | 0.16 | 0.07 | 0.06 | 0.04 | 1.9e-14 | 0e+00 |
| syn-n2-r2 | 0.20 | 0.11 | 0.56x | - | - | - | - | 0.15 | 0.08 | 0.06 | 0.04 | 1.2e-15 | 0e+00 |
| syn-n2-r3 | 0.30 | 0.11 | 0.36x | - | - | - | - | 0.24 | 0.07 | 0.05 | 0.04 | 5.0e-13 | 0e+00 |
| syn-n3-r2 | 4.27 | 1.62 | 0.38x | - | - | 2.94 | 1.18 | 1.09 | 0.28 | 0.24 | 0.15 | 1.7e-14 | 2e-16 |
| syn-n3-r3 | 3.87 | 1.88 | 0.49x | - | - | 2.56 | 1.45 | 1.09 | 0.28 | 0.24 | 0.15 | 1.7e-14 | 2e-16 |
| syn-n3-r4 | 5.08 | 1.40 | 0.28x | - | - | 3.36 | 0.94 | 1.45 | 0.30 | 0.24 | 0.15 | 1.7e-14 | 2e-16 |
| syn-n4-r3 | 16.91 | 12.25 | 0.72x | 7.82 | 3.75 | 6.45 | 7.51 | 2.41 | 0.32 | 0.13 | 0.10 | 7.9e-14 | 5e-15 |
| syn-n4-r4 | 15.86 | 13.25 | 0.84x | 9.18 | 5.25 | 5.08 | 7.26 | 1.40 | 0.35 | 0.13 | 0.10 | 2.4e-14 | 1e-14 |
| syn-n4-r5 | 12.19 | 10.50 | 0.86x | 5.24 | 3.68 | 4.60 | 6.72 | 2.09 | 0.34 | 0.13 | 0.11 | 1.1e-14 | 4e-15 |
| syn-n5-r4 | 68.68 | 36.54 | 0.53x | 39.60 | 24.51 | 21.35 | 10.10 | 5.36 | 0.65 | 0.20 | 0.17 | 4.6e-14 | 3e-14 |
| syn-n5-r5 | 75.41 | 36.01 | 0.48x | 41.81 | 22.73 | 25.61 | 11.77 | 5.53 | 0.61 | 0.21 | 0.18 | 4.5e-13 | 4e-13 |
| syn-n5-r6 | 91.88 | 45.86 | 0.50x | 49.66 | 31.28 | 31.47 | 13.75 | 7.85 | 0.68 | 0.21 | 0.17 | 1.1e-14 | 3e-14 |
| syn-n6-r5 | 494.69 | 307.86 | 0.62x | 407.86 | 259.53 | 56.99 | 44.59 | 9.35 | 1.94 | 0.33 | 0.22 | 1.3e-12 | 1e-13 |
| syn-n6-r6 | 428.05 | 273.43 | 0.64x | 346.35 | 211.76 | 59.77 | 44.48 | 8.95 | 1.73 | 0.33 | 0.21 | 7.3e-13 | 9e-14 |
| syn-n6-r7 | 540.80 | 323.53 | 0.60x | 439.25 | 262.93 | 72.23 | 45.22 | 12.27 | 2.03 | 0.32 | 0.21 | 2.5e-09 | 1e-13 |

## `olm-264a6ce-sym-e679e8c1-tensor`

oneloopmaster main 264a6ce (continuation and degenerate-kinematics fixes, hand-written C0/D0 and tensor binary64 kernels) + Symbolica e679e8c1; the previous benchmark revision

| case | OneLOop all | native all | native/OneLOop | OL D0 | nat D0 | OL C0 | nat C0 | OL B | nat B (tensor) | OL A0 | nat A0 | max diff | relabel |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| four-photon-top-box-0 | 10.13 | 5.30 | 0.52x | 3.79 | 2.27 | 4.78 | 2.20 | 1.33 | 0.26 | 0.16 | 0.15 | 8.3e-14 | 5e-15 |
| gg-hhh-top-box-8 | 22.49 | 11.79 | 0.52x | 13.42 | 8.24 | 5.43 | 2.80 | 3.17 | 0.34 | 0.16 | 0.15 | 4.1e-15 | 7e-15 |
| gg-hhh-top-pentagon-26 | 98.22 | 50.60 | 0.52x | 61.85 | 37.64 | 26.28 | 11.45 | 7.28 | 0.63 | 0.20 | 0.19 | 1.3e-14 | 2e-14 |
| gg-hhh-top-triangle-0 | 1.27 | 0.59 | 0.46x | - | - | 0.67 | 0.35 | 0.50 | 0.12 | 0.13 | 0.11 | 4.6e-16 | 2e-15 |
| gg-hh-top-box-2 | 27.76 | 12.48 | 0.45x | 16.58 | 8.63 | 6.67 | 3.10 | 3.93 | 0.35 | 0.16 | 0.15 | 3.0e-14 | 4e-15 |
| gg-hh-top-triangle-0 | 1.90 | 0.68 | 0.36x | - | - | 0.91 | 0.43 | 0.87 | 0.13 | 0.13 | 0.11 | 4.6e-16 | 3e-15 |
| hgg-top-triangle-0 | 6.53 | 1.57 | 0.24x | - | - | 4.80 | 1.32 | 1.62 | 0.13 | 0.13 | 0.11 | 4.5e-15 | 7e-15 |
| madloop-four-photon-top-box-0 | 9.13 | 4.91 | 0.54x | 3.85 | 2.27 | 3.66 | 1.85 | 1.36 | 0.26 | 0.17 | 0.15 | 5.0e-15 | 5e-15 |
| madloop-gg-hhh-top-box-8 | 27.40 | 13.69 | 0.50x | 13.26 | 8.21 | 10.38 | 4.64 | 3.16 | 0.34 | 0.16 | 0.15 | 2.4e-15 | 7e-15 |
| madloop-gg-hhh-top-pentagon-26 | 107.47 | 53.57 | 0.50x | 63.62 | 37.96 | 33.44 | 13.76 | 7.22 | 0.63 | 0.20 | 0.19 | 1.1e-14 | 2e-14 |
| madloop-gg-hhh-top-triangle-0 | 1.30 | 0.59 | 0.45x | - | - | 0.68 | 0.34 | 0.49 | 0.12 | 0.13 | 0.11 | 4.6e-16 | 2e-15 |
| madloop-gg-hh-top-box-2 | 33.21 | 14.28 | 0.43x | 16.48 | 8.66 | 11.43 | 4.78 | 3.94 | 0.36 | 0.17 | 0.15 | 3.1e-14 | 5e-15 |
| madloop-gg-hh-top-triangle-0 | 1.89 | 0.67 | 0.36x | - | - | 0.90 | 0.42 | 0.86 | 0.13 | 0.13 | 0.11 | 4.6e-16 | 3e-15 |
| madloop-hgg-top-triangle-0 | 3.24 | 0.85 | 0.26x | - | - | 1.49 | 0.57 | 1.63 | 0.14 | 0.13 | 0.11 | 2.8e-15 | 7e-15 |
| syn-gghhh-ggHHH-n5-r4 | 117.19 | 56.35 | 0.48x | 69.13 | 40.21 | 36.53 | 14.43 | 8.03 | 0.65 | 0.20 | 0.19 | 2.5e-14 | 2e-14 |
| syn-gghhh-ggHHH-n5-r5 | 118.36 | 57.71 | 0.49x | 69.96 | 40.22 | 36.25 | 14.38 | 8.09 | 0.64 | 0.21 | 0.19 | 4.4e-12 | 2e-13 |
| syn-gghhh-gHgHH-n5-r4 | 64.24 | 26.02 | 0.41x | 24.66 | 12.81 | 30.68 | 12.63 | 6.82 | 0.66 | 0.21 | 0.19 | 2.0e-14 | 2e-14 |
| syn-gghhh-gHgHH-n5-r5 | 62.27 | 25.79 | 0.41x | 23.98 | 12.57 | 29.18 | 12.16 | 6.83 | 0.65 | 0.20 | 0.19 | 1.7e-14 | 3e-14 |
| syn-gghhh-nosmat-n5-r4 | 117.56 | 56.59 | 0.48x | 67.91 | 40.21 | 37.66 | 14.36 | 8.01 | 0.65 | 0.20 | 0.19 | 1.1e-14 | 1e-14 |
| syn-n2-r1 | 0.21 | 0.11 | 0.51x | - | - | - | - | 0.16 | 0.07 | 0.06 | 0.04 | 1.9e-14 | 0e+00 |
| syn-n2-r2 | 0.20 | 0.11 | 0.57x | - | - | - | - | 0.15 | 0.08 | 0.06 | 0.04 | 1.2e-15 | 0e+00 |
| syn-n2-r3 | 0.30 | 0.11 | 0.38x | - | - | - | - | 0.24 | 0.07 | 0.05 | 0.04 | 5.0e-13 | 0e+00 |
| syn-n3-r2 | 4.27 | 1.60 | 0.38x | - | - | 2.94 | 1.17 | 1.09 | 0.28 | 0.24 | 0.16 | 1.7e-14 | 2e-16 |
| syn-n3-r3 | 3.87 | 1.85 | 0.48x | - | - | 2.56 | 1.43 | 1.09 | 0.28 | 0.24 | 0.16 | 1.7e-14 | 2e-16 |
| syn-n3-r4 | 5.08 | 1.34 | 0.26x | - | - | 3.36 | 0.90 | 1.45 | 0.29 | 0.24 | 0.16 | 1.7e-14 | 2e-16 |
| syn-n4-r3 | 16.91 | 11.64 | 0.69x | 7.82 | 3.56 | 6.45 | 7.42 | 2.41 | 0.31 | 0.13 | 0.11 | 7.9e-14 | 7e-15 |
| syn-n4-r4 | 15.86 | 13.28 | 0.84x | 9.18 | 5.26 | 5.08 | 7.25 | 1.40 | 0.35 | 0.13 | 0.11 | 2.4e-14 | 1e-14 |
| syn-n4-r5 | 12.19 | 10.41 | 0.85x | 5.24 | 3.23 | 4.60 | 6.50 | 2.09 | 0.32 | 0.13 | 0.11 | 1.1e-14 | 4e-15 |
| syn-n5-r4 | 68.68 | 33.80 | 0.49x | 39.60 | 22.86 | 21.35 | 9.85 | 5.36 | 0.63 | 0.20 | 0.19 | 2.2e-14 | 4e-14 |
| syn-n5-r5 | 75.41 | 35.07 | 0.47x | 41.81 | 22.36 | 25.61 | 11.62 | 5.53 | 0.61 | 0.21 | 0.19 | 6.2e-14 | 1e-13 |
| syn-n5-r6 | 91.88 | 44.59 | 0.49x | 49.66 | 29.58 | 31.47 | 13.55 | 7.85 | 0.68 | 0.21 | 0.19 | 1.8e-14 | 3e-14 |
| syn-n6-r5 | 494.69 | 299.16 | 0.60x | 407.86 | 243.42 | 56.99 | 43.56 | 9.35 | 1.93 | 0.33 | 0.22 | 1.4e-12 | 2e-13 |
| syn-n6-r6 | 428.05 | 257.13 | 0.60x | 346.35 | 205.27 | 59.77 | 43.31 | 8.95 | 1.73 | 0.33 | 0.22 | 7.3e-13 | 2e-13 |
| syn-n6-r7 | 540.80 | 315.88 | 0.58x | 439.25 | 259.04 | 72.23 | 43.74 | 12.27 | 2.03 | 0.32 | 0.22 | 2.5e-09 | 1e-13 |

## `olm-a92b148-sym-e679e8c1-tensor`

oneloopmaster `claude/native-masters` a92b148 (direct `dilog_complex_f64`, NativeTensorEvaluator B11/B111 for Ninja's rank-2/3 bubble calls) + Symbolica e679e8c1 (2c06f245 + binary64 dilogarithm)

| case | OneLOop all | native all | native/OneLOop | OL D0 | nat D0 | OL C0 | nat C0 | OL B | nat B (tensor) | OL A0 | nat A0 | max diff | relabel |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| four-photon-top-box-0 | 10.13 | 15.60 | 1.54x | 3.79 | 8.09 | 4.78 | 6.85 | 1.33 | 0.43 | 0.16 | 0.14 | 5.7e-14 | 4e-15 |
| gg-hhh-top-box-8 | 22.49 | 33.02 | 1.47x | 13.42 | 23.77 | 5.43 | 8.16 | 3.17 | 0.70 | 0.16 | 0.14 | 4.2e-15 | 1e-14 |
| gg-hhh-top-pentagon-26 | 98.22 | 133.68 | 1.36x | 61.85 | 103.51 | 26.28 | 27.50 | 7.28 | 1.39 | 0.20 | 0.18 | 1.8e-14 | 1e-14 |
| gg-hhh-top-triangle-0 | 1.27 | 1.74 | 1.37x | - | - | 0.67 | 1.42 | 0.50 | 0.20 | 0.13 | 0.11 | 3.3e-16 | 1e-15 |
| gg-hh-top-box-2 | 27.76 | 34.30 | 1.24x | 16.58 | 24.50 | 6.67 | 8.68 | 3.93 | 0.75 | 0.16 | 0.14 | 3.0e-14 | 7e-15 |
| gg-hh-top-triangle-0 | 1.90 | 1.90 | 1.00x | - | - | 0.91 | 1.55 | 0.87 | 0.23 | 0.13 | 0.11 | 5.3e-16 | 2e-15 |
| hgg-top-triangle-0 | 6.53 | 3.58 | 0.55x | - | - | 4.80 | 3.19 | 1.62 | 0.26 | 0.13 | 0.11 | 7.6e-15 | 9e-15 |
| madloop-four-photon-top-box-0 | 9.13 | 14.21 | 1.56x | 3.85 | 6.83 | 3.66 | 6.62 | 1.36 | 0.43 | 0.17 | 0.14 | 2.8e-14 | 4e-15 |
| madloop-gg-hhh-top-box-8 | 27.40 | 34.65 | 1.26x | 13.26 | 22.56 | 10.38 | 11.08 | 3.16 | 0.70 | 0.16 | 0.15 | 3.1e-15 | 5e-15 |
| madloop-gg-hhh-top-pentagon-26 | 107.47 | 134.10 | 1.25x | 63.62 | 100.05 | 33.44 | 31.37 | 7.22 | 1.39 | 0.20 | 0.18 | 1.4e-14 | 1e-14 |
| madloop-gg-hhh-top-triangle-0 | 1.30 | 1.73 | 1.33x | - | - | 0.68 | 1.41 | 0.49 | 0.20 | 0.13 | 0.11 | 3.3e-16 | 1e-15 |
| madloop-gg-hh-top-box-2 | 33.21 | 35.75 | 1.08x | 16.48 | 23.27 | 11.43 | 11.33 | 3.94 | 0.75 | 0.17 | 0.15 | 3.1e-14 | 1e-14 |
| madloop-gg-hh-top-triangle-0 | 1.89 | 1.87 | 0.99x | - | - | 0.90 | 1.53 | 0.86 | 0.23 | 0.13 | 0.11 | 3.6e-16 | 2e-15 |
| madloop-hgg-top-triangle-0 | 3.24 | 2.06 | 0.64x | - | - | 1.49 | 1.67 | 1.63 | 0.27 | 0.13 | 0.11 | 3.7e-15 | 9e-15 |
| syn-gghhh-ggHHH-n5-r4 | 117.19 | 139.52 | 1.19x | 69.13 | 104.30 | 36.53 | 32.29 | 8.03 | 1.45 | 0.20 | 0.18 | 3.6e-14 | 2e-14 |
| syn-gghhh-ggHHH-n5-r5 | 118.36 | 140.05 | 1.18x | 69.96 | 104.96 | 36.25 | 32.35 | 8.09 | 1.44 | 0.21 | 0.18 | 6.8e-10 | 2e-13 |
| syn-gghhh-gHgHH-n5-r4 | 64.24 | 60.46 | 0.94x | 24.66 | 29.29 | 30.68 | 28.88 | 6.82 | 1.41 | 0.21 | 0.18 | 1.8e-14 | 3e-14 |
| syn-gghhh-gHgHH-n5-r5 | 62.27 | 60.78 | 0.98x | 23.98 | 29.62 | 29.18 | 28.90 | 6.83 | 1.40 | 0.20 | 0.18 | 2.6e-14 | 5e-14 |
| syn-gghhh-nosmat-n5-r4 | 117.56 | 139.21 | 1.18x | 67.91 | 104.00 | 37.66 | 32.39 | 8.01 | 1.46 | 0.20 | 0.18 | 1.9e-14 | 1e-14 |
| syn-n2-r1 | 0.21 | 0.20 | 0.97x | - | - | - | - | 0.16 | 0.16 | 0.06 | 0.04 | 1.8e-14 | 0e+00 |
| syn-n2-r2 | 0.20 | 0.21 | 1.07x | - | - | - | - | 0.15 | 0.17 | 0.06 | 0.04 | 1.2e-15 | 0e+00 |
| syn-n2-r3 | 0.30 | 0.22 | 0.74x | - | - | - | - | 0.24 | 0.18 | 0.05 | 0.04 | 5.0e-13 | 0e+00 |
| syn-n3-r2 | 4.27 | 4.41 | 1.03x | - | - | 2.94 | 3.83 | 1.09 | 0.41 | 0.24 | 0.18 | 2.6e-14 | 8e-15 |
| syn-n3-r3 | 3.87 | 4.47 | 1.16x | - | - | 2.56 | 3.88 | 1.09 | 0.41 | 0.24 | 0.18 | 2.6e-14 | 8e-15 |
| syn-n3-r4 | 5.08 | 4.39 | 0.86x | - | - | 3.36 | 3.77 | 1.45 | 0.45 | 0.24 | 0.18 | 2.6e-14 | 8e-15 |
| syn-n4-r3 | 16.91 | 20.35 | 1.20x | 7.82 | 10.03 | 6.45 | 9.32 | 2.41 | 0.62 | 0.13 | 0.11 | 1.5e+00 | 8e-15 |
| syn-n4-r4 | 15.86 | 25.55 | 1.61x | 9.18 | 15.57 | 5.08 | 8.93 | 1.40 | 0.69 | 0.13 | 0.11 | 1.5e+00 | 2e-14 |
| syn-n4-r5 | 12.19 | 18.68 | 1.53x | 5.24 | 9.88 | 4.60 | 7.84 | 2.09 | 0.63 | 0.13 | 0.11 | 1.6e-14 | 2e-14 |
| syn-n5-r4 | 68.68 | 88.14 | 1.28x | 39.60 | 61.14 | 21.35 | 24.82 | 5.36 | 1.23 | 0.20 | 0.18 | 1.4e-14 | 4e-14 |
| syn-n5-r5 | 75.41 | 88.27 | 1.17x | 41.81 | 58.56 | 25.61 | 27.79 | 5.53 | 1.25 | 0.21 | 0.18 | 1.5e-13 | 7e-14 |
| syn-n5-r6 | 91.88 | 110.69 | 1.20x | 49.66 | 76.89 | 31.47 | 30.81 | 7.85 | 1.57 | 0.21 | 0.18 | 1.9e-14 | 2e-14 |
| syn-n6-r5 | 494.69 | 669.47 | 1.35x | 407.86 | 580.37 | 56.99 | 69.19 | 9.35 | 3.02 | 0.33 | 0.26 | 1.2e-12 | 8e-13 |
| syn-n6-r6 | 428.05 | 574.36 | 1.34x | 346.35 | 484.72 | 59.77 | 68.67 | 8.95 | 2.85 | 0.33 | 0.25 | 1.3e+00 | 2e+00 |
| syn-n6-r7 | 540.80 | 695.92 | 1.29x | 439.25 | 606.72 | 72.23 | 68.64 | 12.27 | 3.29 | 0.32 | 0.25 | 2.5e-09 | 3e-13 |

## `olm-c9dd82f-sym-e679e8c1`

oneloopmaster main c9dd82f + Symbolica e679e8c1 (registered polylog(2) callback now binary64); rank-2 bubbles as B0 only

| case | OneLOop all | native all | native/OneLOop | OL D0 | nat D0 | OL C0 | nat C0 | OL B | nat B0 only | OL A0 | nat A0 | max diff | relabel |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| four-photon-top-box-0 | 10.13 | 15.67 | 1.55x | 3.79 | 8.16 | 4.78 | 6.96 | 1.33 | 0.39 | 0.16 | 0.14 | 5.7e-14 | 4e-15 |
| gg-hhh-top-box-8 | 22.49 | 33.21 | 1.48x | 13.42 | 24.09 | 5.43 | 8.20 | 3.17 | 0.61 | 0.16 | 0.14 | 4.2e-15 | 1e-14 |
| gg-hhh-top-pentagon-26 | 98.22 | 135.23 | 1.38x | 61.85 | 104.90 | 26.28 | 27.67 | 7.28 | 1.20 | 0.20 | 0.18 | 1.8e-14 | 1e-14 |
| gg-hhh-top-triangle-0 | 1.27 | 1.73 | 1.36x | - | - | 0.67 | 1.42 | 0.50 | 0.18 | 0.13 | 0.11 | 2.6e-16 | 1e-15 |
| gg-hh-top-box-2 | 27.76 | 34.65 | 1.25x | 16.58 | 24.82 | 6.67 | 8.71 | 3.93 | 0.65 | 0.16 | 0.14 | 3.0e-14 | 7e-15 |
| gg-hh-top-triangle-0 | 1.90 | 1.89 | 1.00x | - | - | 0.91 | 1.57 | 0.87 | 0.21 | 0.13 | 0.11 | 5.3e-16 | 2e-15 |
| hgg-top-triangle-0 | 6.53 | 3.62 | 0.55x | - | - | 4.80 | 3.24 | 1.62 | 0.24 | 0.13 | 0.11 | 7.6e-15 | 9e-15 |
| madloop-four-photon-top-box-0 | 9.13 | 14.30 | 1.57x | 3.85 | 6.89 | 3.66 | 6.68 | 1.36 | 0.39 | 0.17 | 0.14 | 2.8e-14 | 4e-15 |
| madloop-gg-hhh-top-box-8 | 27.40 | 34.88 | 1.27x | 13.26 | 22.82 | 10.38 | 11.13 | 3.16 | 0.59 | 0.16 | 0.14 | 3.1e-15 | 5e-15 |
| madloop-gg-hhh-top-pentagon-26 | 107.47 | 135.64 | 1.26x | 63.62 | 101.56 | 33.44 | 31.53 | 7.22 | 1.18 | 0.20 | 0.18 | 1.4e-14 | 1e-14 |
| madloop-gg-hhh-top-triangle-0 | 1.30 | 1.72 | 1.33x | - | - | 0.68 | 1.42 | 0.49 | 0.18 | 0.13 | 0.11 | 2.6e-16 | 1e-15 |
| madloop-gg-hh-top-box-2 | 33.21 | 35.91 | 1.08x | 16.48 | 23.56 | 11.43 | 11.42 | 3.94 | 0.63 | 0.17 | 0.14 | 3.1e-14 | 1e-14 |
| madloop-gg-hh-top-triangle-0 | 1.89 | 1.86 | 0.99x | - | - | 0.90 | 1.54 | 0.86 | 0.21 | 0.13 | 0.11 | 3.6e-16 | 2e-15 |
| madloop-hgg-top-triangle-0 | 3.24 | 2.08 | 0.64x | - | - | 1.49 | 1.71 | 1.63 | 0.24 | 0.13 | 0.11 | 3.7e-15 | 9e-15 |
| syn-gghhh-ggHHH-n5-r4 | 117.19 | 141.40 | 1.21x | 69.13 | 105.78 | 36.53 | 32.48 | 8.03 | 1.23 | 0.20 | 0.18 | 3.6e-14 | 2e-14 |
| syn-gghhh-ggHHH-n5-r5 | 118.36 | 141.91 | 1.20x | 69.96 | 106.37 | 36.25 | 32.50 | 8.09 | 1.24 | 0.21 | 0.18 | 6.8e-10 | 2e-13 |
| syn-gghhh-gHgHH-n5-r4 | 64.24 | 60.83 | 0.95x | 24.66 | 29.70 | 30.68 | 29.08 | 6.82 | 1.18 | 0.21 | 0.18 | 1.8e-14 | 3e-14 |
| syn-gghhh-gHgHH-n5-r5 | 62.27 | 61.44 | 0.99x | 23.98 | 30.14 | 29.18 | 29.12 | 6.83 | 1.18 | 0.20 | 0.18 | 2.6e-14 | 5e-14 |
| syn-gghhh-nosmat-n5-r4 | 117.56 | 141.00 | 1.20x | 67.91 | 105.36 | 37.66 | 32.66 | 8.01 | 1.24 | 0.20 | 0.18 | 1.9e-14 | 1e-14 |
| syn-n2-r1 | 0.21 | 0.13 | 0.64x | - | - | - | - | 0.16 | 0.10 | 0.06 | 0.04 | 1.6e-14 | 0e+00 |
| syn-n2-r2 | 0.20 | 0.13 | 0.67x | - | - | - | - | 0.15 | 0.10 | 0.06 | 0.04 | 1.2e-14 | 0e+00 |
| syn-n2-r3 (3 calls skipped) | 0.30 | 0.04 | 0.12x | - | - | - | - | 0.24 | - | 0.05 | 0.04 | 9.1e-17 | 0e+00 |
| syn-n3-r2 | 4.27 | 4.40 | 1.03x | - | - | 2.94 | 3.85 | 1.09 | 0.37 | 0.24 | 0.18 | 2.6e-14 | 8e-15 |
| syn-n3-r3 | 3.87 | 4.45 | 1.15x | - | - | 2.56 | 3.91 | 1.09 | 0.37 | 0.24 | 0.17 | 2.6e-14 | 8e-15 |
| syn-n3-r4 (9 calls skipped) | 5.08 | 3.96 | 0.78x | - | - | 3.36 | 3.79 | 1.45 | - | 0.24 | 0.17 | 2.6e-14 | 8e-15 |
| syn-n4-r3 | 16.91 | 20.52 | 1.21x | 7.82 | 10.11 | 6.45 | 9.50 | 2.41 | 0.52 | 0.13 | 0.11 | 1.5e+00 | 8e-15 |
| syn-n4-r4 | 15.86 | 25.69 | 1.62x | 9.18 | 15.77 | 5.08 | 9.10 | 1.40 | 0.52 | 0.13 | 0.11 | 1.5e+00 | 2e-14 |
| syn-n4-r5 (18 calls skipped) | 12.19 | 18.28 | 1.50x | 5.24 | 9.94 | 4.60 | 7.97 | 2.09 | - | 0.13 | 0.11 | 1.6e-14 | 2e-14 |
| syn-n5-r4 | 68.68 | 88.87 | 1.29x | 39.60 | 61.96 | 21.35 | 25.10 | 5.36 | 1.01 | 0.20 | 0.18 | 1.4e-14 | 4e-14 |
| syn-n5-r5 | 75.41 | 89.32 | 1.18x | 41.81 | 59.28 | 25.61 | 27.93 | 5.53 | 1.07 | 0.21 | 0.18 | 1.5e-13 | 7e-14 |
| syn-n5-r6 (30 calls skipped) | 91.88 | 110.34 | 1.20x | 49.66 | 78.08 | 31.47 | 30.96 | 7.85 | - | 0.21 | 0.18 | 1.9e-14 | 2e-14 |
| syn-n6-r5 | 494.69 | 669.36 | 1.35x | 407.86 | 579.61 | 56.99 | 69.87 | 9.35 | 2.49 | 0.33 | 0.25 | 1.2e-12 | 8e-13 |
| syn-n6-r6 | 428.05 | 579.60 | 1.35x | 346.35 | 492.11 | 59.77 | 69.41 | 8.95 | 2.34 | 0.33 | 0.25 | 1.3e+00 | 2e+00 |
| syn-n6-r7 (45 calls skipped) | 540.80 | 695.83 | 1.29x | 439.25 | 608.00 | 72.23 | 69.38 | 12.27 | - | 0.32 | 0.25 | 2.0e-12 | 3e-13 |

## `olm-c9dd82f-sym-2c06f245`

oneloopmaster main c9dd82f + Symbolica 2c06f245 (the NinjaRS workspace pin): MPFR dilogarithm (historical; first measurement of this stream)

| case | OneLOop all | native all | native/OneLOop | OL D0 | nat D0 | OL C0 | nat C0 | OL B | nat B0 only | OL A0 | nat A0 | max diff | relabel |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| four-photon-top-box-0 | 10.13 | 2629.83 | 260x | 3.79 | 1898.14 | 4.78 | 688.41 | 1.33 | 0.38 | 0.16 | 0.14 | 5.7e-14 | n/a |
| gg-hhh-top-box-8 | 22.49 | 6739.04 | 300x | 13.42 | 4811.31 | 5.43 | 1904.36 | 3.17 | 0.60 | 0.16 | 0.14 | 4.2e-15 | n/a |
| gg-hhh-top-pentagon-26 | 98.22 | 39610.98 | 403x | 61.85 | 26077.51 | 26.28 | 13596.19 | 7.28 | 1.18 | 0.20 | 0.17 | 1.9e-14 | n/a |
| gg-hhh-top-triangle-0 | 1.27 | 31.21 | 25x | - | - | 0.67 | 30.87 | 0.50 | 0.18 | 0.13 | 0.10 | 2.6e-16 | n/a |
| gg-hh-top-box-2 | 27.76 | 10706.02 | 386x | 16.58 | 7040.33 | 6.67 | 3632.69 | 3.93 | 0.64 | 0.16 | 0.14 | 3.0e-14 | n/a |
| gg-hh-top-triangle-0 | 1.90 | 186.76 | 98x | - | - | 0.91 | 185.85 | 0.87 | 0.20 | 0.13 | 0.10 | 5.3e-16 | n/a |
| hgg-top-triangle-0 | 6.53 | 2046.94 | 314x | - | - | 4.80 | 2047.29 | 1.62 | 0.24 | 0.13 | 0.10 | 7.6e-15 | n/a |
| syn-gghhh-ggHHH-n5-r4 | 117.19 | 51452.40 | 439x | 69.13 | 30516.08 | 36.53 | 20917.52 | 8.03 | 1.22 | 0.20 | 0.17 | 3.6e-14 | n/a |
| syn-gghhh-ggHHH-n5-r5 | 118.36 | 53471.00 | 452x | 69.96 | 33147.38 | 36.25 | 20151.97 | 8.09 | 1.23 | 0.21 | 0.17 | 6.8e-10 | n/a |
| syn-gghhh-gHgHH-n5-r4 | 64.24 | 30602.49 | 476x | 24.66 | 15915.17 | 30.68 | 14597.27 | 6.82 | 1.17 | 0.21 | 0.18 | 1.9e-14 | n/a |
| syn-gghhh-gHgHH-n5-r5 | 62.27 | 27929.03 | 449x | 23.98 | 12952.36 | 29.18 | 14783.28 | 6.83 | 1.17 | 0.20 | 0.17 | 2.6e-14 | n/a |
| syn-gghhh-nosmat-n5-r4 | 117.56 | 48438.18 | 412x | 67.91 | 27315.42 | 37.66 | 20993.45 | 8.01 | 1.24 | 0.20 | 0.17 | 1.8e-14 | n/a |
| syn-n2-r1 | 0.21 | 0.13 | 0.64x | - | - | - | - | 0.16 | 0.10 | 0.06 | 0.03 | 1.6e-14 | n/a |
| syn-n2-r2 | 0.20 | 0.13 | 0.67x | - | - | - | - | 0.15 | 0.10 | 0.06 | 0.03 | 1.2e-14 | n/a |
| syn-n2-r3 (3 calls skipped) | 0.30 | 0.03 | 0.12x | - | - | - | - | 0.24 | - | 0.05 | 0.03 | 9.1e-17 | n/a |
| syn-n3-r2 | 4.27 | 367.17 | 86x | - | - | 2.94 | 367.24 | 1.09 | 0.36 | 0.24 | 0.17 | 2.6e-14 | n/a |
| syn-n3-r3 | 3.87 | 405.24 | 105x | - | - | 2.56 | 406.36 | 1.09 | 0.36 | 0.24 | 0.17 | 2.6e-14 | n/a |
| syn-n3-r4 (9 calls skipped) | 5.08 | 321.19 | 63x | - | - | 3.36 | 320.64 | 1.45 | - | 0.24 | 0.17 | 2.6e-14 | n/a |
| syn-n4-r3 | 16.91 | 4670.62 | 276x | 7.82 | 1768.96 | 6.45 | 2866.21 | 2.41 | 0.51 | 0.13 | 0.10 | 1.5e+00 | n/a |
| syn-n4-r4 | 15.86 | 3056.18 | 193x | 9.18 | 1451.79 | 5.08 | 1578.67 | 1.40 | 0.51 | 0.13 | 0.10 | 1.5e+00 | n/a |
| syn-n4-r5 (18 calls skipped) | 12.19 | 2314.93 | 190x | 5.24 | 845.92 | 4.60 | 1443.85 | 2.09 | - | 0.13 | 0.10 | 1.6e-14 | n/a |
| syn-n5-r4 | 68.68 | 16246.24 | 237x | 39.60 | 10117.89 | 21.35 | 6124.04 | 5.36 | 1.00 | 0.20 | 0.17 | 1.4e-14 | n/a |
| syn-n5-r5 | 75.41 | 21105.92 | 280x | 41.81 | 12980.98 | 25.61 | 8055.47 | 5.53 | 1.06 | 0.21 | 0.17 | 1.5e-13 | n/a |
| syn-n5-r6 (30 calls skipped) | 91.88 | 34901.62 | 380x | 49.66 | 19477.46 | 31.47 | 15409.95 | 7.85 | - | 0.21 | 0.17 | 1.9e-14 | n/a |
| syn-n6-r5 | 494.69 | 123927.22 | 251x | 407.86 | 103552.38 | 56.99 | 20552.25 | 9.35 | 2.46 | 0.33 | 0.24 | 1.2e-12 | n/a |
| syn-n6-r6 | 428.05 | 126306.62 | 295x | 346.35 | 104203.30 | 59.77 | 23256.36 | 8.95 | 2.33 | 0.33 | 0.24 | 1.3e+00 | n/a |
| syn-n6-r7 (45 calls skipped) | 540.80 | 128423.91 | 237x | 439.25 | 105499.49 | 72.23 | 22653.10 | 12.27 | - | 0.32 | 0.24 | 2.0e-12 | n/a |

## Notes

* **Dilogarithm.** With the NinjaRS workspace's Symbolica pin (2c06f245) the generated code's f64
  `dilog` goes through Symbolica's registered `polylog(2, z)` callback, which converts to MPFR and
  sums a series (12 us - 2 ms per call): C0/D0 were 25-500x slower than OneLOop. Symbolica e679e8c1
  (`claude/oneloopmaster-integration-dilog`: 2c06f245 + a binary64 Bernoulli-series dilogarithm,
  the specialization described in oneloopmaster's `STATUS.md`/`patches/README.md`) evaluates it
  in 17-110 ns (`DILOG_TEST=1`); oneloopmaster a92b148 calls that kernel directly.
* **Defects of earlier revisions, fixed in 264a6ce.** a92b148's native C0 with one massless and
  two massive internal lines above the massive-pair threshold had the wrong imaginary part (e.g.
  C0(0, 0, 490000; 173^2, 0, 173^2; mu^2=1e4)), syn-n4-r3/r4 had O(1) C0/D0 deviations, and
  syn-gghhh-ggHHH-n5-r5 had a D0 near a vanishing Gram determinant at 6.8e-10 (OneLOop: 7.9e-15,
  checked against 256-bit values). At 264a6ce the native C0 of syn-n6-r6 with nearly coincident
  roots was at ~2e-13 versus OneLOop's 3e-15 (OneLOop uses a dedicated near-coincident dilogarithm
  difference).
* **Changes in 3023fa6.** 3cbe08d conditions the coincident-root dilogarithm differences (that C0
  is now at 2.2e-15 against 256-bit values). 9c5490b (contour-box edge roots from exact Kallen
  functions) moves the D0 of syn-gghhh-ggHHH-n5-r5 near a vanishing Gram determinant from 4.4e-12
  to 3.6e-11 (OneLOop 7.9e-15) and syn-n5-r5's largest deviation from 6.2e-14 to 4.5e-13; the
  physical streams keep 8.3e-14. Against the 264a6ce run, native boxes take up to 14 % longer
  (4-9 % on the physical streams), and families without code changes moved by -10 to +6 %. Run
  alternately on CPU 250 (best of three, five physical streams), 3023fa6's boxes take 3-5 % longer
  than 264a6ce's, triangles and bubbles within 2 %, tadpoles 7-9 % less. Native/OneLOop is
  0.25-0.57x on the physical streams (0.24-0.54x at 264a6ce).
* **Changes in 8ec256fc (with Symbolica 9cdc2949).** The vanishing-Gram contour box (Vieta pole-zero
  offsets, second contour) brings the D0 of syn-gghhh-ggHHH-n5-r5 from 3.6e-11 to 3.0e-14 against
  OneLOop (2.6e-14 against 256-bit values, OneLOop 7.9e-15) and syn-n6-r5's largest deviation from
  1.3e-12 to 4.0e-13; gg-hh-top-box-2 (both call streams) moves from 1.5e-14 to 1.7e-14 and
  syn-gghhh-ggHHH-n5-r4 from 1.7e-14 to 1.8e-14, every other stream keeps its largest deviation (to
  two digits). Run alternately with a rebuilt 3023fa6 bench on the same core (one stream per
  process, three rounds, medians): boxes take 1.7-4.3 % longer on the physical streams (-2.4 to
  +3.9 % on the synthetic ones), triangles within 1 % (up to +5.8 % on syn-n4/n6), bubbles and
  tadpoles within 3 %; instructions per cycle of each process within -3.1 to +0.8 %. The OneLOop column
  is the committed CPU-250 replay; native/OneLOop is 0.24-0.55x on the physical streams.
