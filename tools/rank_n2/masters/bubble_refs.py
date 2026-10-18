"""High-precision Feynman-parameter references for the finite parts of B111 and B1111
(OneLOop normalisation, mu^2 = 1), and comparison with the Rust route output.

B_{1^a} finite = (-1)^(a+1) int_0^1 x^a log(Delta(x) - i0) dx,
Delta(x) = -x(1-x) p2 + (1-x) m0 + x m1   (squared masses, Im m <= 0).
Poles: B111 -1/4, B1111 +1/5 (zero for the fully scaleless point).
"""
import sys
import mpmath as mp

mp.mp.dps = 50

POINTS = [
    ("below", [3.0, 0.75, 0.0, 1.5, 0.0, 1.0]),
    ("above", [11.0, 1.125, 0.0, 0.625, 0.0, 1.0]),
    ("spacelike", [-7.25, 0.375, 0.0, 2.875, 0.0, 1.0]),
    ("complex", [3.0, 0.75, -0.03125, 1.5, -0.078125, 1.0]),
    ("p0_unequal", [0.0, 2.0, 0.0, 5.0, 0.0, 1.0]),
    ("p0_unequal_swapped", [0.0, 5.0, 0.0, 2.0, 0.0, 1.0]),
    ("p0_equal", [0.0, 1.25, 0.0, 1.25, 0.0, 1.0]),
    ("m0_zero", [2.5, 0.0, 0.0, 1.5, 0.0, 1.0]),
    ("m1_zero", [2.5, 1.5, 0.0, 0.0, 0.0, 1.0]),
    ("massless_space", [-4.0, 0.0, 0.0, 0.0, 0.0, 1.0]),
    ("massless_time", [4.0, 0.0, 0.0, 0.0, 0.0, 1.0]),
    ("onshell_m0_zero", [1.5, 0.0, 0.0, 1.5, 0.0, 1.0]),
    ("small_p_2^-4", [0.0625, 1.0, 0.0, 1.5, 0.0, 1.0]),
    ("small_p_2^-8", [0.00390625, 1.0, 0.0, 1.5, 0.0, 1.0]),
    ("small_p_2^-12", [0.000244140625, 1.0, 0.0, 1.5, 0.0, 1.0]),
    ("small_p_2^-16", [1.52587890625e-05, 1.0, 0.0, 1.5, 0.0, 1.0]),
    ("small_p_2^-24", [5.9604644775390625e-08, 1.0, 0.0, 1.5, 0.0, 1.0]),
    ("small_p_equal_2^-8", [0.00390625, 1.0, 0.0, 1.0, 0.0, 1.0]),
    ("large_p", [10000.0, 1.0, 0.0, 2.0, 0.0, 1.0]),
]


def finite(a, p):
    p2 = mp.mpf(p[0])
    m0 = mp.mpc(p[1], p[2])
    m1 = mp.mpc(p[3], p[4])

    def log_delta(x):
        d = -x * (1 - x) * p2 + (1 - x) * m0 + x * m1
        if d.imag == 0:
            if d.real < 0:
                return mp.log(-d.real) - 1j * mp.pi
            return mp.log(d.real)
        return mp.log(d)

    points = [mp.mpf(0), mp.mpf(1)]
    if p[2] == 0 and p[4] == 0 and p2 != 0:
        b = m1.real - m0.real - p2
        disc = b * b - 4 * p2 * m0.real
        if disc >= 0:
            for r in ((-b + mp.sqrt(disc)) / (2 * p2), (-b - mp.sqrt(disc)) / (2 * p2)):
                if 0 < r < 1:
                    points.append(r)
    points = sorted(set(points))
    value = mp.quad(lambda x: x ** a * log_delta(x), points)
    return (-1) ** (a + 1) * value


def main():
    refs = {}
    for name, p in POINTS:
        for a in (3, 4):
            refs[(a, name)] = finite(a, p)
    if len(sys.argv) < 2:
        for (a, name), v in refs.items():
            print(a, name, mp.nstr(v, 25))
        return
    rows = []
    with open(sys.argv[1]) as handle:
        for line in handle:
            parts = line.split()
            if len(parts) != 7 or not parts[1].startswith("B1"):
                continue
            route, family, name = parts[0], parts[1], parts[2]
            a = len(family) - 1
            value = mp.mpc(mp.mpf(parts[3]), mp.mpf(parts[4]))
            pole = mp.mpc(mp.mpf(parts[5]), mp.mpf(parts[6]))
            ref = refs[(a, name)]
            err = abs(value - ref) / max(abs(ref), mp.mpf('1e-300'))
            scaleless = name.startswith("massless") is False and False
            expected_pole = mp.mpf(-1) / 4 if a == 3 else mp.mpf(1) / 5
            pole_err = abs(pole - expected_pole)
            rows.append((route, family, name, err, pole_err, value, ref))
    print(f"{'route':16s} {'fam':6s} {'point':22s} {'rel.err finite':>15s} {'|pole-exp|':>11s}  value / reference")
    for route, family, name, err, pole_err, value, ref in rows:
        print(f"{route:16s} {family:6s} {name:22s} {mp.nstr(err, 3):>15s} {mp.nstr(pole_err, 3):>11s}  "
              f"{mp.nstr(value, 17)} / {mp.nstr(ref, 17)}")


if __name__ == "__main__":
    main()
