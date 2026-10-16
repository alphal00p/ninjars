The shared physical benchmark uses a separate frozen binary64 input pool, `validation/evidence/gg-hhh-changing-pool.json`. It fixes the `pp` helicity and varies `sqrt(s)=450,525,600`, with equal-energy equilateral final Higgs momenta. The model input order is `[MT,ymt,MH,Gf,aS,aEWM1,MZ]`; all three points use the same model values, zero widths and scale squared10000. The final Higgs-pair invariant crosses the massive-top-pair threshold between the second and third points. No point lies on that threshold.

`phase_space_pool.rs` constructs exact rational/radical momenta using Symbolica, checks conservation and mass shells with the shared FeynKit `FourMomentum` type, evaluates at256 bits and rounds each component once to binary64. Both the shared-provider benchmark and MadLoop must consume those stored numerical rows unchanged. The reference harness verifies that MadLoop read back exactly the same binary64 values. This removes differences from evaluating algebraically equivalent radicals through different floating-point sequences.

Reproduce the pool and reference using an existing locked all-feature CLI artifact manifest and the previously exported/probed graph55:

```sh
python3 validation/madloop/gg_hhh/phase_space_pool.py \
  --cargo-artifacts target/physical/gg-hhh/final-binary/cargo-artifacts.jsonl \
  --output-dir target/physical/gg-hhh/new-changing-pool
python3 validation/madloop/gg_hhh/changing_points.py \
  --pool target/physical/gg-hhh/new-changing-pool/pool.json \
  --baseline-probe-dir target/reference/madloop-gg-hhh-probes/probe-55 \
  --export-dir target/reference/madloop-gg-hhh-external/graph-55 \
  --output-dir target/reference/madloop-gg-hhh-new-changing-points
python3 -m unittest discover -s validation/madloop/gg_hhh -p test_changing_points.py -v
```

The driver copies the verified graph55 harness into a new artifact directory, replaces only its fixed momentum assignments with a five-row input read, and recompiles the original loop numerator and CutTools interface. It preserves fixed modes1/4, the forbidden-counterterm callback trap, the fixed model, and the70-point complex loop grid. The original reference export, original driver and main50-graph fixtures remain unchanged.

The archived reference `gg-hhh-changing-madloop.json` passes420 internal raw/wrapper checks and all six fixed-mode integrations. It records the graph26→MadLoop55 routing, loop probes, external wavefunctions and couplings, CutTools shifts and linear Laurent results. The raw conversion is `−1/2`; the integrated conversion is `−8π²`. Independent comparison against the shared FeynKit numerical preparation is a separate evidence item; reference self-consistency alone does not establish that comparison or any performance gate. Fixed mode4 still promotes binary64 external inputs, and is not a high-precision physical-input oracle.
