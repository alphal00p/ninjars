# Ill-conditioned synthetic lanes

Evidence for the two synthetic bench cases that differ from original Ninja by
more than 1e-6 (marked † in the report's Table 2): `syn-n6-r6` and
`syn-gghhh-ggHHH-n5-r5`. Both binaries load the bench case
(`validation/benchmark/synthetic/<case>.json`) and Ninja's reference results
(`validation/benchmark/ninja/<case>.json`), compute the exact Laurent
coefficients with the generic NinjaRS engine in Symbolica `Float` (and
DoubleFloat) with oneloopmaster's exact masters at the working precision, and
compare the binary64 fast engine with native masters, with Ninja's recorded
OneLOop masters and with exact masters rounded to binary64, and original
Ninja + OneLOop, against them.

```sh
cargo run --release --bin n6r6 > n6r6.jsonl     # see src/bin/n6r6.rs for modes
cargo run --release --bin n5r5 > n5r5.jsonl
python3 an6.py n6r6.jsonl
python3 an5.py n5r5.jsonl
```

Measured with NinjaRS 606d94a and oneloopmaster 264a6ce (`n6r6.jsonl`,
`n5r5.jsonl`, `n5r5_d0.txt`); the native-master columns were re-measured with
oneloopmaster 3023fa6 (`*-olm3023fa6.*`). The n6r6 rerun was made without
OneLOop's quadruple-precision tables: its six `master_check` records (256 and
400 bits, lanes 0-2) list no bad calls, but there they only confirm that every
exact master evaluated and make no quadruple-precision comparison. The exact
values are unchanged. oneloopmaster 149ddff, the current pin, leaves the
3023fa6 columns as they are: its native masters return the same bits as
3023fa6's on every master call Ninja records for these cases, and `case-bench`
gives identical reference checks. Errors are |x - exact| / |exact eps0| on the
finite part (eps0).

| `syn-n6-r6` | Ninja + OneLOop | fast, native masters (264a6ce → 3023fa6 = 149ddff) | fast, Ninja's OneLOop masters | fast, exact masters (rounded) |
|---|---|---|---|---|
| lane 0 | 2.32e-2 | 2.35e-1 → 1.31e-2 | 1.23e-2 | 1.50e-2 |
| lane 1 | 2.54e-10 | 6.87e-10 → 6.87e-10 | 6.95e-10 | 3.30e-10 |
| lane 2 | 1.27e-10 | 4.91e-10 → 4.91e-10 | 9.87e-11 | 1.46e-10 |

Lane 0 cancels by about 1e12 (exact eps0 = -4.3127e-17 + 3.7935e-17 i
against cut-constructible and rational parts of 7e-5); Float 256 and 400 bits
agree in all 40 printed digits, and all master calls agree with OneLOop
quadruple precision at both precisions (`master_check`). Replacing the native
masters by exact ones one call at a time (`n6r6_pc.jsonl`) attributed 0.221 of
the 264a6ce lane-0 error to a single C0 with nearly coincident roots,
C0(-237088.3, -236743.3, 0; 29929 - 259.5i, 0, 6464.16 - 168.84i), native
relative error 2.4e-13 against 3.3e-15 for OneLOop; the reflected labelling
of the same C0 is accurate to 2.3e-14 (`probe.jsonl`). oneloopmaster 3cbe08d
conditions those dilogarithm differences (native 2.2e-15), and the fast engine
with native masters is now off by 1.3e-2 on this lane, like Ninja.

| `syn-gghhh-ggHHH-n5-r5` | Ninja + OneLOop | fast, native masters (264a6ce → 3023fa6 = 149ddff) | fast, Ninja's OneLOop masters | fast, exact masters (rounded) |
|---|---|---|---|---|
| lane 0 | 3.52e-12 | 6.85e-12 → 4.78e-12 | 3.82e-12 | 3.35e-12 |
| lane 1 | 2.99e-4 | 3.36e-4 → 4.58e-4 | 3.09e-4 | 3.09e-4 |
| lane 2 | 7.16e-12 | 7.37e-12 → 7.77e-12 | 8.23e-12 | 7.01e-12 |

Lane 1 borders a box with normalized Gram determinant 8.2e-8; its finite part
is a ~1e-7 remainder of cancelling box and triangle terms. The binary64
reduction itself loses the digits (exact masters rounded to binary64 still
give 3.1e-4), in both codes. The D0 at that box is in `n5r5_d0.txt` and
`n5r5_d0-olm3023fa6.txt`: native relative error 4.4e-12 at 264a6ce (6.8e-10
before it) and 3.6e-11 at 3023fa6 and 149ddff, OneLOop 7.9e-15. The 3023fa6
value comes from oneloopmaster 9c5490b (contour-box edge roots from exact
Källén functions), which moved this D0 by 4.0e-11; it raises the lane's error
from 3.4e-4 to 4.6e-4. This regression has been reported upstream.
