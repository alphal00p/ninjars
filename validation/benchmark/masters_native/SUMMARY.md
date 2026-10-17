# oneloopmaster native masters vs OneLOop Fortran (identical call streams)

Native: `NativeEvaluator<f64>` (feature `generated-evaluators`, direct ahead-of-time Rust, no
interpreter/JIT), one scalar `evaluate` per call, built with NinjaRS's release profile
(thin LTO, 1 CGU). OneLOop: the same calls replayed through `ninja::avh_olo` (Ninja build
`o3`, OneLOop-3.6, gfortran -O3). Both pinned to CPU 29; microseconds per phase-space
point (one lane's calls), median of >=7 blocks >=25 ms. 'B' native is B0 only, whereas OneLOop's
rank-2 bubble call (olo_b11) returns B11, B00, B1 and B0. max diff = largest scaled
difference of native vs recorded OneLOop results (D0/C0/B0/A0).

| case | OneLOop all | native all | ratio | OL D0 | nat D0 | OL C0 | nat C0 | OL B | nat B0 | OL A0 | nat A0 | max diff |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| four-photon-top-box-0 | 11.53 | 2629.83 | 228x | 5.13 | 1898.14 | 4.79 | 688.41 | 1.33 | 0.38 | 0.16 | 0.14 | 5.7e-14 |
| gg-hhh-top-box-8 | 23.61 | 6739.04 | 285x | 14.71 | 4811.31 | 5.41 | 1904.36 | 3.18 | 0.60 | 0.15 | 0.14 | 4.2e-15 |
| gg-hhh-top-pentagon-26 | 94.40 | 39610.98 | 420x | 62.06 | 26077.51 | 24.24 | 13596.19 | 7.29 | 1.18 | 0.19 | 0.17 | 1.9e-14 |
| gg-hhh-top-triangle-0 | 1.25 | 31.21 | 25x | - | - | 0.66 | 30.87 | 0.49 | 0.18 | 0.12 | 0.10 | 2.6e-16 |
| gg-hh-top-box-2 | 28.54 | 10706.02 | 375x | 17.65 | 7040.33 | 6.61 | 3632.69 | 3.92 | 0.64 | 0.15 | 0.14 | 3.0e-14 |
| gg-hh-top-triangle-0 | 1.90 | 186.76 | 98x | - | - | 0.91 | 185.85 | 0.86 | 0.20 | 0.12 | 0.10 | 5.3e-16 |
| hgg-top-triangle-0 | 6.50 | 2046.94 | 315x | - | - | 4.76 | 2047.29 | 1.61 | 0.24 | 0.12 | 0.10 | 7.6e-15 |
| syn-gghhh-ggHHH-n5-r4 | 109.31 | 51452.40 | 471x | 65.61 | 30516.08 | 34.75 | 20917.52 | 8.01 | 1.22 | 0.19 | 0.17 | 3.6e-14 |
| syn-gghhh-ggHHH-n5-r5 | 109.79 | 53471.00 | 487x | 65.96 | 33147.38 | 34.86 | 20151.97 | 8.04 | 1.23 | 0.19 | 0.17 | 6.8e-10 |
| syn-gghhh-gHgHH-n5-r4 | 61.16 | 30602.49 | 500x | 24.12 | 15915.17 | 29.55 | 14597.27 | 6.83 | 1.17 | 0.19 | 0.18 | 1.9e-14 |
| syn-gghhh-gHgHH-n5-r5 | 59.57 | 27929.03 | 469x | 23.51 | 12952.36 | 28.71 | 14783.28 | 6.86 | 1.17 | 0.19 | 0.17 | 2.6e-14 |
| syn-gghhh-nosmat-n5-r4 | 110.11 | 48438.18 | 440x | 64.96 | 27315.42 | 36.16 | 20993.45 | 7.98 | 1.24 | 0.19 | 0.17 | 1.8e-14 |
| syn-n2-r1 | 0.20 | 0.13 | 1x | - | - | - | - | 0.15 | 0.10 | 0.05 | 0.03 | 1.6e-14 |
| syn-n2-r2 | 0.20 | 0.13 | 1x | - | - | - | - | 0.15 | 0.10 | 0.05 | 0.03 | 1.2e-14 |
| syn-n2-r3 | 0.30 | 0.03 | 0x | - | - | - | - | 0.25 | - | 0.05 | 0.03 | 9.1e-17 |
| syn-n3-r2 | 4.26 | 367.17 | 86x | - | - | 2.93 | 367.24 | 1.10 | 0.36 | 0.24 | 0.17 | 2.6e-14 |
| syn-n3-r3 | 3.96 | 405.24 | 102x | - | - | 2.65 | 406.36 | 1.08 | 0.36 | 0.24 | 0.17 | 2.6e-14 |
| syn-n3-r4 | 4.98 | 321.19 | 64x | - | - | 3.28 | 320.64 | 1.45 | - | 0.24 | 0.17 | 2.6e-14 |
| syn-n4-r3 | 16.76 | 4670.62 | 279x | 7.75 | 1768.96 | 6.40 | 2866.21 | 2.41 | 0.51 | 0.12 | 0.10 | 1.5e+00 |
| syn-n4-r4 | 15.81 | 3056.18 | 193x | 9.15 | 1451.79 | 5.12 | 1578.67 | 1.41 | 0.51 | 0.12 | 0.10 | 1.5e+00 |
| syn-n4-r5 | 12.27 | 2314.93 | 189x | 5.21 | 845.92 | 4.58 | 1443.85 | 2.18 | - | 0.12 | 0.10 | 1.6e-14 |
| syn-n5-r4 | 65.61 | 16246.24 | 248x | 38.60 | 10117.89 | 21.06 | 6124.04 | 5.31 | 1.00 | 0.19 | 0.17 | 1.4e-14 |
| syn-n5-r5 | 71.46 | 21105.92 | 295x | 40.59 | 12980.98 | 24.78 | 8055.47 | 5.54 | 1.06 | 0.20 | 0.17 | 1.5e-13 |
| syn-n5-r6 | 85.64 | 34901.62 | 408x | 47.45 | 19477.46 | 29.44 | 15409.95 | 7.88 | - | 0.19 | 0.17 | 1.9e-14 |
| syn-n6-r5 | 485.65 | 123927.22 | 255x | 396.85 | 103552.38 | 55.71 | 20552.25 | 9.36 | 2.46 | 0.31 | 0.24 | 1.2e-12 |
| syn-n6-r6 | 418.49 | 126306.62 | 302x | 335.59 | 104203.30 | 57.34 | 23256.36 | 8.97 | 2.33 | 0.31 | 0.24 | 1.3e+00 |
| syn-n6-r7 | 533.99 | 128423.91 | 240x | 432.01 | 105499.49 | 69.08 | 22653.10 | 12.40 | - | 0.31 | 0.24 | 2.0e-12 |

Cause of the native C0/D0 cost: the f64 dilogarithm of the generated code
(`oneloopmaster/src/native/primitives.rs`, `<f64 as NativeFloat>::dilog`) calls Symbolica's
registered `polylog(2, z)` for `Complex<f64>`, which converts to MPFR `Float` and sums a series
(`references/symbolica` 2c06f245, `src/transcendental.rs` `polylog_eval_complex_f64`).
Measured with `DILOG_TEST=1 masters-native-bench /dev/null` on CPU 29: 12-33 us for real
arguments in (-3, 1), 108 us at z=2, 270 us at 0.5+0.5i, 2.1 ms at 0.9+0.3i, 0.69 ms at 0.3-0.8i.
