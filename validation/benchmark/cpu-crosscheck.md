# Reference timings: CPU 250 (today) vs CPU 29 (committed)

Ratio = CPU-250 re-timing (2026-10-18) / committed CPU-29 value (2026-10-17). Same executables
(sha256-verified), same protocol (Ninja replay: 9 interleaved rounds, blocks >= 25 ms; MadLoop:
run_madloop.py step-4 sequence, 2 x 9 blocks >= 25 ms). Results (amplitudes) bitwise identical: Ninja True, MadLoop True.

## Findings

* Everything except OneLOop agrees within run-to-run noise: Ninja pure_reduction and numerator_part, MadLoop
  coefficient construction, TensorNumerator and Ninja reduction have median ratios 1.000-1.022 (mostly <= 1.01).
* OneLOop masters are reproducibly slower on CPU 250 for the n=5 cases (box/pentagon kinematics): +5 to +8 %
  (gg-hhh-top-pentagon-26, syn-*n5*; MadLoop pentagon +5 %, gg-hh box +3-6 %), -4..+3 % elsewhere. A second CPU-250 pass
  reproduces it (repeat/first masters ratio 0.98-1.01) and a same-day CPU-29 control of three cases reproduces
  the committed CPU-29 values within ~1 % (masters 0.991-1.012): the shift is core/placement-specific
  (CPU 250, socket 1 / NUMA node 7, L3 complex 248-255), not day-to-day drift. During the CPU-250 runs CPUs
  251-255 (same 16 MB L3) were ~85-90 % busy with other users' work, CPU 29's L3 neighbours ~20-25 %;
  L3 sharing (OneLOop + libgfortran have the largest code/data footprint) is a plausible, unverified cause.
* full therefore moves by the masters share: median +1 %, up to +5-6 % for the n=5 cases.
* For the NinjaRS-vs-reference ratios measured on CPU 250 this means the committed CPU-29 references are
  up to ~5-8 % (masters) / ~4-6 % (full) too FAST for the n=5 cases, i.e. the same-core comparison is
  slightly more favourable to NinjaRS; for all other components the cross-core caveat is below ~1-2 %.
* MadLoop madloop_glue is a difference of separately timed executables (+-1 us); its ratios are not meaningful.

## Ninja 1.2.1 + OneLOop 3.6 replay (validation/benchmark/ninja)

numerator_part = reduction_plus_tensor_numerator - pure_reduction. 'worst' = ratio farthest from 1.

### Build `o3-fortranrules`

| cases | component | n | median | min | max | worst |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| all | pure_reduction | 34 | 1.005 | 0.970 | 1.055 | 1.055 |
| all | numerator_part | 34 | 1.005 | 0.959 | 1.055 | 1.055 |
| all | masters_only | 34 | 1.019 | 0.962 | 1.079 | 1.079 |
| all | full | 34 | 1.011 | 0.993 | 1.061 | 1.061 |
| physical | pure_reduction | 7 | 1.000 | 0.984 | 1.016 | 0.984 |
| physical | numerator_part | 7 | 1.022 | 1.004 | 1.055 | 1.055 |
| physical | masters_only | 7 | 1.012 | 0.998 | 1.075 | 1.075 |
| physical | full | 7 | 1.007 | 0.993 | 1.053 | 1.053 |
| synthetic | pure_reduction | 20 | 1.004 | 0.970 | 1.055 | 1.055 |
| synthetic | numerator_part | 20 | 1.005 | 0.959 | 1.022 | 0.959 |
| synthetic | masters_only | 20 | 1.020 | 0.962 | 1.079 | 1.079 |
| synthetic | full | 20 | 1.011 | 1.002 | 1.061 | 1.061 |
| madloop-replay | pure_reduction | 7 | 1.009 | 1.002 | 1.038 | 1.038 |
| madloop-replay | numerator_part | 7 | 1.003 | 1.000 | 1.008 | 1.008 |
| madloop-replay | masters_only | 7 | 1.018 | 0.998 | 1.053 | 1.053 |
| madloop-replay | full | 7 | 1.009 | 1.000 | 1.034 | 1.034 |

### Build `o3`

| cases | component | n | median | min | max | worst |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| all | pure_reduction | 34 | 1.006 | 0.953 | 1.018 | 0.953 |
| all | numerator_part | 34 | 1.005 | 0.990 | 1.017 | 1.017 |
| all | masters_only | 34 | 1.013 | 0.997 | 1.077 | 1.077 |
| all | full | 34 | 1.010 | 0.994 | 1.057 | 1.057 |
| physical | pure_reduction | 7 | 1.005 | 0.953 | 1.007 | 0.953 |
| physical | numerator_part | 7 | 1.005 | 0.990 | 1.017 | 1.017 |
| physical | masters_only | 7 | 1.006 | 0.999 | 1.053 | 1.053 |
| physical | full | 7 | 1.002 | 1.000 | 1.042 | 1.042 |
| synthetic | pure_reduction | 20 | 1.008 | 0.997 | 1.018 | 1.018 |
| synthetic | numerator_part | 20 | 1.005 | 0.995 | 1.011 | 1.011 |
| synthetic | masters_only | 20 | 1.017 | 0.997 | 1.077 | 1.077 |
| synthetic | full | 20 | 1.012 | 0.996 | 1.057 | 1.057 |
| madloop-replay | pure_reduction | 7 | 1.006 | 0.979 | 1.010 | 0.979 |
| madloop-replay | numerator_part | 7 | 1.005 | 1.000 | 1.008 | 1.008 |
| madloop-replay | masters_only | 7 | 1.015 | 1.001 | 1.059 | 1.059 |
| madloop-replay | full | 7 | 1.010 | 0.994 | 1.031 | 1.031 |

### Build `o3-ol34`

| cases | component | n | median | min | max | worst |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| all | pure_reduction | 27 | 1.007 | 0.989 | 1.039 | 1.039 |
| all | numerator_part | 27 | 1.006 | 0.967 | 1.031 | 0.967 |
| all | masters_only | 27 | 1.012 | 0.986 | 1.065 | 1.065 |
| all | full | 27 | 1.010 | 0.987 | 1.047 | 1.047 |
| physical | pure_reduction | 7 | 1.003 | 0.992 | 1.009 | 1.009 |
| physical | numerator_part | 7 | 1.008 | 0.967 | 1.013 | 0.967 |
| physical | masters_only | 7 | 1.012 | 0.996 | 1.052 | 1.052 |
| physical | full | 7 | 1.012 | 0.987 | 1.042 | 1.042 |
| synthetic | pure_reduction | 20 | 1.007 | 0.989 | 1.039 | 1.039 |
| synthetic | numerator_part | 20 | 1.006 | 0.979 | 1.031 | 1.031 |
| synthetic | masters_only | 20 | 1.014 | 0.986 | 1.065 | 1.065 |
| synthetic | full | 20 | 1.009 | 1.001 | 1.047 | 1.047 |

### Per case, build `o3-fortranrules` (ratios)

| case | pure_reduction | numerator_part | masters_only | full | full CPU29 (us) | full CPU250 (us) |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| four-photon-top-box-0 | 1.010 | 1.009 | 0.999 | 1.007 | 42.70 | 43.00 |
| gg-hh-top-box-2 | 1.004 | 1.009 | 1.026 | 1.015 | 60.37 | 61.29 |
| gg-hh-top-triangle-0 | 0.994 | 1.055 | 1.012 | 1.006 | 5.38 | 5.41 |
| gg-hhh-top-box-8 | 1.016 | 1.004 | 1.022 | 1.018 | 54.84 | 55.85 |
| gg-hhh-top-pentagon-26 | 0.984 | 1.026 | 1.075 | 1.053 | 134.03 | 141.07 |
| gg-hhh-top-triangle-0 | 1.000 | 1.022 | 1.009 | 0.993 | 4.78 | 4.75 |
| hgg-top-triangle-0 | 0.997 | 1.036 | 0.998 | 0.996 | 10.09 | 10.05 |
| madloop-four-photon-top-box-0 | 1.012 | 1.002 | 1.018 | 1.009 | 41.51 | 41.89 |
| madloop-gg-hh-top-box-2 | 1.009 | 1.004 | 1.034 | 1.034 | 64.50 | 66.68 |
| madloop-gg-hh-top-triangle-0 | 1.009 | 1.008 | 0.998 | 1.006 | 9.55 | 9.61 |
| madloop-gg-hhh-top-box-8 | 1.038 | 1.005 | 1.018 | 1.024 | 59.65 | 61.09 |
| madloop-gg-hhh-top-pentagon-26 | 1.008 | 1.000 | 1.053 | 1.031 | 209.38 | 215.81 |
| madloop-gg-hhh-top-triangle-0 | 1.002 | 1.003 | 1.026 | 1.000 | 8.90 | 8.90 |
| madloop-hgg-top-triangle-0 | 1.010 | 1.000 | 1.009 | 1.003 | 10.93 | 10.97 |
| syn-gghhh-gHgHH-n5-r4 | 1.008 | 1.006 | 1.057 | 1.044 | 103.54 | 108.10 |
| syn-gghhh-gHgHH-n5-r5 | 0.970 | 1.006 | 1.054 | 1.018 | 166.39 | 169.38 |
| syn-gghhh-ggHHH-n5-r4 | 1.004 | 1.013 | 1.066 | 1.061 | 152.70 | 162.05 |
| syn-gghhh-ggHHH-n5-r5 | 1.055 | 0.998 | 1.079 | 1.048 | 217.54 | 227.96 |
| syn-gghhh-nosmat-n5-r4 | 1.000 | 1.005 | 1.073 | 1.048 | 153.89 | 161.32 |
| syn-n2-r1 | 1.018 | 0.959 | 0.977 | 1.011 | 0.67 | 0.67 |
| syn-n2-r2 | 1.046 | 0.984 | 1.019 | 1.012 | 1.08 | 1.10 |
| syn-n2-r3 | 1.006 | 1.010 | 0.962 | 1.010 | 4.04 | 4.08 |
| syn-n3-r2 | 1.013 | 0.998 | 0.997 | 1.003 | 7.72 | 7.75 |
| syn-n3-r3 | 1.006 | 1.003 | 1.004 | 1.002 | 11.63 | 11.65 |
| syn-n3-r4 | 0.999 | 1.005 | 1.021 | 1.004 | 32.78 | 32.92 |
| syn-n4-r3 | 1.016 | 1.002 | 1.013 | 1.013 | 29.50 | 29.89 |
| syn-n4-r4 | 1.008 | 1.007 | 1.005 | 1.010 | 47.63 | 48.11 |
| syn-n4-r5 | 1.003 | 1.007 | 1.013 | 1.007 | 113.47 | 114.26 |
| syn-n5-r4 | 0.998 | 1.022 | 1.055 | 1.048 | 107.75 | 112.92 |
| syn-n5-r5 | 0.997 | 1.002 | 1.060 | 1.024 | 179.27 | 183.65 |
| syn-n5-r6 | 1.000 | 1.001 | 1.076 | 1.011 | 406.83 | 411.21 |
| syn-n6-r5 | 1.004 | 1.009 | 1.022 | 1.011 | 607.14 | 614.00 |
| syn-n6-r6 | 1.003 | 1.007 | 1.019 | 1.008 | 725.33 | 731.22 |
| syn-n6-r7 | 1.004 | 1.007 | 1.010 | 1.005 | 1353.12 | 1360.17 |

### Same-core noise today: second CPU-250 pass / first CPU-250 pass

| build | component | n | median | min | max | worst |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| o3 | pure_reduction | 34 | 1.000 | 0.991 | 1.055 | 1.055 |
| o3 | numerator_part | 34 | 1.001 | 0.989 | 1.020 | 1.020 |
| o3 | masters_only | 34 | 0.999 | 0.979 | 1.013 | 0.979 |
| o3 | full | 34 | 1.001 | 0.992 | 1.038 | 1.038 |
| o3-fortranrules | pure_reduction | 34 | 0.999 | 0.959 | 1.042 | 0.959 |
| o3-fortranrules | numerator_part | 34 | 1.000 | 0.776 | 1.030 | 0.776 |
| o3-fortranrules | masters_only | 34 | 0.999 | 0.980 | 1.011 | 0.980 |
| o3-fortranrules | full | 34 | 1.000 | 0.987 | 1.014 | 1.014 |
| o3-ol34 | pure_reduction | 27 | 0.999 | 0.981 | 1.016 | 0.981 |
| o3-ol34 | numerator_part | 27 | 0.999 | 0.986 | 1.009 | 0.986 |
| o3-ol34 | masters_only | 27 | 0.999 | 0.976 | 1.110 | 1.110 |
| o3-ol34 | full | 27 | 0.999 | 0.986 | 1.008 | 0.986 |

### Control: same cases on CPU 29 today (CPU29-today / CPU29-committed vs CPU250 / CPU29-committed)

| case | build | pure_reduction (29 / 250) | numerator_part (29 / 250) | masters_only (29 / 250) | full (29 / 250) |
| --- | --- | ---: | ---: | ---: | ---: |
| gg-hhh-top-pentagon-26 | o3 | 1.003 / 1.005 | 1.008 / 1.006 | 1.012 / 1.053 | 1.012 / 1.042 |
| gg-hhh-top-pentagon-26 | o3-fortranrules | 0.983 / 0.984 | 1.010 / 1.026 | 1.010 / 1.075 | 1.009 / 1.053 |
| gg-hhh-top-pentagon-26 | o3-ol34 | 1.001 / 1.003 | 1.002 / 1.009 | 1.005 / 1.052 | 1.005 / 1.042 |
| syn-gghhh-ggHHH-n5-r5 | o3 | 1.005 / 1.008 | 1.003 / 1.004 | 1.003 / 1.077 | 1.013 / 1.031 |
| syn-gghhh-ggHHH-n5-r5 | o3-fortranrules | 1.005 / 1.055 | 0.991 / 0.998 | 1.006 / 1.079 | 1.014 / 1.048 |
| syn-gghhh-ggHHH-n5-r5 | o3-ol34 | 0.999 / 1.005 | 1.002 / 1.005 | 0.991 / 1.065 | 0.998 / 1.039 |
| syn-n6-r6 | o3 | 1.002 / 1.018 | 1.001 / 0.999 | 1.007 / 1.007 | 0.991 / 0.996 |
| syn-n6-r6 | o3-fortranrules | 1.004 / 1.003 | 1.007 / 1.007 | 1.006 / 1.019 | 1.004 / 1.008 |
| syn-n6-r6 | o3-ol34 | 0.999 / 1.014 | 1.004 / 1.004 | 1.007 / 1.017 | 1.003 / 1.004 |

## MadLoop standalone, per point, one helicity (validation/benchmark/madloop)

### `O3` build: summary over the 7 graphs

| component | n | median | min | max | worst |
| --- | ---: | ---: | ---: | ---: | ---: |
| numerator_coefficients | 7 | 1.007 | 0.992 | 1.012 | 1.012 |
| tensor_numerator | 7 | 1.006 | 1.002 | 1.040 | 1.040 |
| ninja_reduction | 7 | 1.009 | 0.997 | 1.027 | 1.027 |
| madloop_glue | 5 | 1.111 | 1.032 | 1.380 | 1.380 |
| masters | 7 | 1.020 | 0.994 | 1.057 | 1.057 |
| full | 7 | 1.015 | 1.008 | 1.033 | 1.033 |
| ninja_total | 7 | 1.019 | 1.005 | 1.033 | 1.033 |
| full_instrumented | 7 | 1.018 | 1.007 | 1.034 | 1.034 |
| helicity_summed | 7 | 1.014 | 1.009 | 1.033 | 1.033 |

madloop_glue absolute change (us): -0.02 .. +1.11 (glue is a difference of separately timed executables, known to about +-1 us)

### `O3` build per graph: CPU29 -> CPU250 (us), ratio

| graph | numerator_coefficients | tensor_numerator | ninja_reduction | madloop_glue | masters | full |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| hgg-top-triangle-0 | 9.49 -> 9.53 (1.004) | 4.77 -> 4.85 (1.017) | 3.37 -> 3.42 (1.014) | 0.71 -> 0.82 (1.147) | 3.26 -> 3.33 (1.020) | 21.41 -> 21.72 (1.015) |
| four-photon-top-box-0 | 18.64 -> 18.82 (1.009) | 22.57 -> 22.68 (1.005) | 10.55 -> 10.63 (1.008) | -0.44 -> -0.45 (-) | 9.08 -> 9.27 (1.020) | 59.86 -> 60.55 (1.011) |
| gg-hh-top-triangle-0 | 7.82 -> 7.90 (1.010) | 4.75 -> 4.95 (1.040) | 3.35 -> 3.36 (1.003) | 0.53 -> 0.59 (1.111) | 1.97 -> 1.96 (0.994) | 18.25 -> 18.39 (1.008) |
| gg-hh-top-box-2 | 12.17 -> 12.31 (1.012) | 22.59 -> 22.65 (1.003) | 10.55 -> 10.84 (1.027) | 0.99 -> 1.36 (1.380) | 32.12 -> 33.94 (1.057) | 78.21 -> 80.78 (1.033) |
| gg-hhh-top-triangle-0 | 7.83 -> 7.88 (1.007) | 4.73 -> 4.79 (1.012) | 3.33 -> 3.32 (0.997) | 0.53 -> 0.55 (1.032) | 1.25 -> 1.25 (1.001) | 17.53 -> 17.69 (1.009) |
| gg-hhh-top-box-8 | 12.22 -> 12.29 (1.006) | 22.50 -> 22.63 (1.006) | 10.49 -> 10.59 (1.009) | 1.21 -> 1.29 (1.064) | 27.23 -> 28.21 (1.036) | 73.35 -> 74.42 (1.015) |
| gg-hhh-top-pentagon-26 | 23.98 -> 23.78 (0.992) | 80.55 -> 80.67 (1.002) | 26.02 -> 26.51 (1.019) | -0.17 -> 0.94 (-) | 105.10 -> 110.46 (1.051) | 234.94 -> 241.36 (1.027) |

### `default` build: summary over the 7 graphs

| component | n | median | min | max | worst |
| --- | ---: | ---: | ---: | ---: | ---: |
| numerator_coefficients | 7 | 1.011 | 1.002 | 1.029 | 1.029 |
| tensor_numerator | 7 | 1.003 | 0.986 | 1.021 | 1.021 |
| ninja_reduction | 7 | 1.008 | 0.992 | 1.016 | 1.016 |
| madloop_glue | 7 | 1.127 | 1.029 | 1.705 | 1.705 |
| masters | 7 | 1.021 | 0.997 | 1.052 | 1.052 |
| full | 7 | 1.021 | 1.010 | 1.035 | 1.035 |
| ninja_total | 7 | 1.012 | 0.993 | 1.024 | 1.024 |
| full_instrumented | 7 | 1.016 | 1.011 | 1.031 | 1.031 |
| helicity_summed | 7 | 1.013 | 1.007 | 1.027 | 1.027 |

madloop_glue absolute change (us): +0.04 .. +2.12 (glue is a difference of separately timed executables, known to about +-1 us)

### `default` build per graph: CPU29 -> CPU250 (us), ratio

| graph | numerator_coefficients | tensor_numerator | ninja_reduction | madloop_glue | masters | full |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| hgg-top-triangle-0 | 13.07 -> 13.17 (1.008) | 4.76 -> 4.86 (1.021) | 3.39 -> 3.45 (1.016) | 1.28 -> 1.40 (1.094) | 3.31 -> 3.35 (1.012) | 25.61 -> 26.15 (1.021) |
| four-photon-top-box-0 | 20.31 -> 20.48 (1.008) | 22.59 -> 22.66 (1.003) | 10.73 -> 10.65 (0.992) | 3.60 -> 4.11 (1.142) | 9.24 -> 9.43 (1.021) | 65.80 -> 66.78 (1.015) |
| gg-hh-top-triangle-0 | 11.97 -> 12.25 (1.023) | 4.89 -> 4.82 (0.986) | 3.38 -> 3.35 (0.992) | 1.35 -> 1.41 (1.042) | 1.97 -> 1.99 (1.011) | 23.23 -> 23.47 (1.010) |
| gg-hh-top-box-2 | 16.53 -> 16.86 (1.020) | 22.58 -> 22.69 (1.005) | 10.72 -> 10.80 (1.008) | 2.02 -> 2.86 (1.413) | 32.63 -> 33.88 (1.038) | 84.09 -> 86.58 (1.030) |
| gg-hhh-top-triangle-0 | 11.86 -> 12.21 (1.029) | 4.74 -> 4.79 (1.011) | 3.34 -> 3.35 (1.003) | 1.32 -> 1.36 (1.029) | 1.27 -> 1.27 (0.997) | 22.26 -> 22.61 (1.016) |
| gg-hhh-top-box-8 | 16.58 -> 16.76 (1.011) | 22.59 -> 22.53 (0.997) | 10.60 -> 10.73 (1.012) | 2.25 -> 2.54 (1.127) | 27.61 -> 28.30 (1.025) | 78.70 -> 80.62 (1.024) |
| gg-hhh-top-pentagon-26 | 25.58 -> 25.64 (1.002) | 81.23 -> 80.61 (0.992) | 26.06 -> 26.38 (1.012) | 3.00 -> 5.12 (1.705) | 105.39 -> 110.90 (1.052) | 239.77 -> 248.26 (1.035) |

### Same-core noise today: second CPU-250 MadLoop pass / first CPU-250 pass

| build | component | n | median | min | max | worst |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| O3 | numerator_coefficients | 7 | 0.999 | 0.990 | 1.014 | 1.014 |
| O3 | tensor_numerator | 7 | 1.002 | 0.975 | 1.006 | 0.975 |
| O3 | ninja_reduction | 7 | 1.000 | 0.995 | 1.010 | 1.010 |
| O3 | madloop_glue | 6 | 0.950 | 0.758 | 1.077 | 0.758 |
| O3 | masters | 7 | 1.000 | 0.999 | 1.002 | 1.002 |
| O3 | full | 7 | 1.002 | 0.996 | 1.004 | 1.004 |
| default | numerator_coefficients | 7 | 1.001 | 0.995 | 1.006 | 1.006 |
| default | tensor_numerator | 7 | 1.002 | 0.998 | 1.007 | 1.007 |
| default | ninja_reduction | 7 | 1.000 | 0.992 | 1.012 | 1.012 |
| default | madloop_glue | 7 | 0.952 | 0.677 | 1.107 | 0.677 |
| default | masters | 7 | 1.003 | 0.987 | 1.007 | 0.987 |
| default | full | 7 | 0.997 | 0.995 | 1.007 | 1.007 |

glue absolute change (us), O3: -0.23 .. +0.27

### Baseline: CPU-29 review re-timing / committed CPU-29 (retime-review.json)

| build | component | n | median | min | max |
| --- | --- | ---: | ---: | ---: | ---: |
| default | numerator_coefficients | 7 | 0.997 | 0.994 | 0.999 |
| default | tensor_numerator | 7 | 1.000 | 0.978 | 1.017 |
| default | ninja_reduction | 7 | 1.004 | 0.983 | 1.005 |
| default | madloop_glue | 7 | 0.995 | 0.851 | 1.579 |
| default | masters | 7 | 1.008 | 0.995 | 1.018 |
| default | full | 7 | 1.000 | 0.996 | 1.005 |
| O3 | numerator_coefficients | 7 | 1.000 | 0.997 | 1.006 |
| O3 | tensor_numerator | 7 | 1.000 | 0.996 | 1.005 |
| O3 | ninja_reduction | 7 | 1.001 | 0.997 | 1.008 |
| O3 | madloop_glue | 5 | 1.008 | 0.945 | 1.093 |
| O3 | masters | 7 | 1.003 | 0.986 | 1.014 |
| O3 | full | 7 | 1.000 | 0.995 | 1.008 |
## Commands (scratch harness in refcpu250/harness/, all timed processes pinned with taskset -c 250)

```sh
source <scratchpad>/devenv.sh
# Ninja replay: run_ninja_bench.py copy with OUT/RAW redirected to refcpu250/, HERE/TARGET = the
# references/claude_reference worktree that built the binaries (byte-identical harness files), rebuild disabled
python3 harness/run_ninja_bench_scratch.py $(cat harness/cases-3var.txt) --variants o3 o3-fortranrules o3-ol34 --cpu 250 --rounds 9 --block-ms 25
python3 harness/run_ninja_bench_scratch.py $(cat harness/cases-madloop.txt) --variants o3-fortranrules o3 --cpu 250 --rounds 9 --block-ms 25
# MadLoop: run_madloop.py step-4 timing sequence on cp -a copies of the rep1 exports (refcpu250/exports/)
PYTHONDONTWRITEBYTECODE=1 python3 harness/retime_madloop_cpu250.py --cpu 250
python3 harness/summarize_scratch.py            # refcpu250/madloop/{summary.json,SUMMARY.md}
# noise baselines: same two Ninja commands via harness/run_ninja_bench_repeat.py (-> repeat/ninja),
# retime_madloop_cpu250.py --out-dir repeat/madloop, and a CPU-29 control of 3 cases via
# harness/run_ninja_bench_cpu29check.py ... --cpu 29 (-> cpu29check/ninja)
python3 harness/compare.py --repeat repeat/ninja --repeat-madloop repeat/madloop --control cpu29check/ninja
```
