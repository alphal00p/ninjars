"""Assemble the machine-readable formulas file (formulas_rank_n2.json) and the
CSE-optimised subtraction code (subtractions_rank_n2_cse.txt).

Everything here is derived/validated by the other scripts:
  decomposition_check.py   (residue spaces, counts, 0-point term)
  laurent_prototype.py     (extraction recipes, layouts, box on-cut mu expansion)
  subtraction_formulas.py  (subtraction terms; engine == original Ninja for old terms)
  check_formulas_vs_series.py (closed forms == independent series code)
  integrals_check.py       (integrated values; 3 symbolic methods + end-to-end numerics)
"""
import json
import os

import sympy as sp

import subtraction_formulas as SF

HERE = os.path.dirname(os.path.abspath(__file__))

# ------------------------------------------------------------------ residues
TRI_NS = {0: "C0", 7: "I3[mu2] = 1/2", 14: "I3[mu4] = (s12+s23+s13)/24 - (m1+m2+m3)/6"}
BUB_NS = {0: "B0", 1: "(p.e2) B1", 2: "(p.e2)^2 B11", 13: "(p.e2)^3 B111", 20: "(p.e2)^4 B1111",
          9: "I2[mu2] = (3(m1+m2) - s)/6", 10: "I2[mu2 qe2] = (p.e2)(s - 2 m1 - 4 m2)/12",
          21: "I2[mu2 qe2^2] = (p.e2)^2 (5 m1 + 15 m2 - 3 s)/60",
          22: "I2[mu4] = -(s^2 - 5 s (m1+m2) + 10 (m1^2 + m1 m2 + m2^2))/60"}
TAD_NS = {0: "A0", 14: "I1[mu2] = m^4/2", 15: "I1[qe3 qe4] = (e3.e4)(m^2 A0/4 + m^4/8)"}
BOX_NS = {0: "D0", 2: "I4[mu2] = O(eps) -> 0", 4: "I4[mu4] = -1/6",
          6: "I4[mu6] = (m1+m2+m3+m4)/12 - (s12+s13+s14+s23+s24+s34)/60"}


def mono_str(j, pw):
    parts = []
    if j:
        parts.append("mu2" + (f"^{j}" if j > 1 else ""))
    for nm, p in sorted(pw.items()):
        q = "qe" + nm[1:] if nm != "v" else "(qbar.Vort)"
        parts.append(q + (f"^{p}" if p > 1 else ""))
    return "*".join(parts) if parts else "1"


def residue_table(table, ns, first_new):
    out = []
    for idx, j, pw in table:
        deg = sum(pw.values()) + 2 * j
        out.append({"index": idx, "monomial": mono_str(j, pw), "mu2_power": j,
                    "powers": pw, "weighted_degree_k_minus": deg,
                    "new_at_rank_n+2": idx >= first_new,
                    "integrates_to": ns.get(idx, "0 (spurious)")})
    return out


def main():
    from laurent_prototype import TRIANGLE, BUBBLE, TADPOLE, BOX
    data = {}
    data["conventions"] = {
        "metric": "(+,-,-,-)",
        "denominators": "D_i = (q+V_i)^2 - m_i^2 - mu2  (m_i^2 written m_i below)",
        "rank": "weighted degree of N(q,mu2): q counts 1, mu2 counts 2; r = n+2",
        "qbar": "qbar = q + V_{i1} with i1 the first (lowest) index of the cut",
        "qe_i": "qbar . e_i with Ninja's cut bases (basis.cc): e1,e2 null in the plane of the cut momenta "
                "(bubble: e2 = reference vector), e3,e4 null transverse, e3.e4 = -e1.e2 = mp34",
        "residue_degree": "R_k = r - n + k = k + 2 for r = n+2 (weighted)",
        "integral_normalisation": "OneLOop/Ninja: d^dq/(i pi^(d/2) r_Gamma) mu^(2eps); rational terms are "
                                  "normalisation independent",
        "masters": "B_{1^a}(s,m1,m2): coefficient of p^mu1..p^mua in int q^mu1..q^mua/((q^2-m1)((q+p)^2-m2)),"
                   " p = V_{i2}-V_{i1}, OneLOop convention (oneloopmaster two_point.rs b1,b11,b111,b1111)",
    }
    data["residues"] = {
        "pentagon": {"ncoef": 1, "monomials": [{"index": 0, "monomial": "mu2", "integrates_to": "O(eps) -> 0"}],
                     "note": "unchanged for every rank; not needed with the mu-expansion"},
        "box": {"ncoef": 7, "monomials": residue_table(BOX, BOX_NS, 6),
                "note": "x = (q+V0).Vort; new: c6 mu2^3 (non-spurious, rational)"},
        "triangle": {"ncoef": 21, "monomials": residue_table(TRIANGLE, TRI_NS, 15),
                     "note": "all six new terms spurious"},
        "bubble": {"ncoef": 35, "monomials": residue_table(BUBBLE, BUB_NS, 20),
                   "note": "new non-spurious: c20 (B1111), c21 (rational), c22 (rational)"},
        "tadpole": {"ncoef": 35, "monomials": residue_table(TADPOLE, TAD_NS, 16),
                    "note": "slot 9 (qe1 qe2) unused as in Ninja; all 20 new terms spurious (odd in qbar)"},
        "hexagon_and_higher": "none (5 denominators fix q and mu2)",
        "zero_point": "c_empty * prod_all D_i: needed at r = n+2 for every n (constant if all tadpoles share "
                      "one basis {e1..e4}, as Ninja's tadpole_basis does; degree <= 1 for n = 1 or for "
                      "independent tadpole bases). Vanishes on every cut, integrates to 0 (scaleless); "
                      "matters only for global N=N tests.",
    }
    data["laurent_layouts"] = {
        "triangle_T3": {"levels": "level i = t^(R-i), mu2^j for j <= i/2 (ascending j)", "R": "r-n+3 = 5",
                        "slots": [list(s) for s in SF.LAYOUT["triangle_rank_n+2"]], "slot_label": "[t, x, mu2]"},
        "bubble_T2": {"levels": "level i = t^(R-i), x^a mu2^j with a+2j <= i, ordered by a then j",
                      "R": "r-n+2 = 4", "level_size": "(floor(i/2)+1)*(i+1-floor(i/2)) = 1,2,4,6,9 "
                      "(Ninja's SizeHelper<2> = 1+i+i/2+i/3 is wrong for i = 4)",
                      "slots": [list(s) for s in SF.LAYOUT["bubble_rank_n+2"]]},
        "tadpole_T3": {"R": "r-n+1 = 3", "slots": [list(s) for s in SF.LAYOUT["tadpole_rank_n+2"]]},
        "rank_n+1_reference": {"bubble": [list(s) for s in SF.LAYOUT["bubble_rank_n+1"]],
                               "tadpole": [list(s) for s in SF.LAYOUT["tadpole_rank_n+1"]]},
    }
    # division index maps for the T2 layout (generic rule + explicit level-4 maps)
    def t2_level(i):
        return [(a, j) for a in range(i + 1) for j in range(i // 2 + 1) if a + 2 * j <= i]
    maps = {}
    for i in range(1, 5):
        lv, lm1 = t2_level(i), t2_level(i - 1)
        d1 = []
        for src_k, (a, j) in enumerate(lm1):
            d1.append({"from": src_k, "z0_to": lv.index((a, j)), "z1_x_to": lv.index((a + 1, j))})
        entry = {"subtractD1 (level i -= d1[0]*level(i-1) + d1[1]*x*level(i-1))": d1}
        if i >= 2:
            lm2 = t2_level(i - 2)
            d2 = []
            for src_k, (a, j) in enumerate(lm2):
                d2.append({"from": src_k, "d2_0_to": lv.index((a, j)), "d2_mu2_to": lv.index((a, j + 1)),
                           "d2_x_to": lv.index((a + 1, j)), "d2_x2_to": lv.index((a + 2, j))})
            entry["subtractD2 (level i -= (d2[0] + d2[1] mu2 + d2[2] x + d2[3] x^2)*level(i-2))"] = d2
        entry["level_i_contents_(x_power, mu2_power)"] = lv
        maps[f"level_{i}"] = entry
    data["division_T2_index_maps"] = maps
    data["extraction"] = {
        "box": {
            "c0": "average of N/prod(D) on the two 4-dim solutions l+-, mu2 = 0",
            "mu_expansion": "q = (q0 - V0) + t*v, mu2 = t^2 v^2 + mu0 with q0 = (l+ + l-)/2 (qbar coords), "
                            "v = Vort, mu0 = q0^2 - m_{i1}  (EXACT d-dim cut; Ninja's rank<=n+1 convention "
                            "mu2 = t^2 v^2 is wrong at the 3rd term)",
            "terms": "n0,n1,n2 = coefficients of t^r, t^(r-1), t^(r-2)",
            "division": "for each uncut l: d0 = 2 v.kk, d1 = 2 q0.kk + f_l (kk = V_l - V0, f_l = kk^2 + m_i1 - m_l);"
                        " n0/=d0; n1=(n1-d1*n0)/d0; n2=(n2-d1*n1)/d0",
            "c6": "n0 / v2^3", "c5": "n1 / v2^3", "c4": "n2 / v2^2 - 3*mu0*c6",
            "c2,c3 (tests only)": "one finite-mu2 sample s: P(s)=avg, Q(s)=diff/(2 x v2) minus pentagons; "
                                  "c2 = (P - c0 - c4 s^2 - c6 s^3)/s, c3 = (Q - c1 - c5 s^2)/s",
            "no_mu_expansion": "even part c0+c2 s+c4 s^2+c6 s^3 needs 4 mu2 samples (e.g. 0,s,-s,2s), odd part 3",
        },
        "triangle": {
            "expansions": "A: q = amu + t e3 + (param+mu2)/t * e4/(2 mp34);  B: e3 <-> e4;  mindeg 5 both",
            "c0,c7,c14": "(Q_A[t^0 mu2^j] + Q_B[t^0 mu2^j])/2, j = 0,1,2 (box terms cancel in the average)",
            "qe4^b mu2^j": "Q_A[t^b mu2^j] / mp34^b  (c4,c5,c6,c13,c16 ; c9,c11,c18 ; c20)",
            "qe3^b mu2^j": "Q_B[t^b mu2^j] / mp34^b  (c1,c2,c3,c12,c15 ; c8,c10,c17 ; c19)",
            "subtractions": "none",
        },
        "bubble": {
            "expansions": "A: q = e1eff + x e2eff + t e3 + (p0+p1 x+p2 x^2+mu2)/t * e4/(2 mp34), mindeg 4; "
                          "B: e3 <-> e4, mindeg 3 (only if tadpoles of its lines are needed or tests)",
            "qe2^a qe4^b mu2^j": "Q_A[t^b x^a mu2^j] / (mp12^a mp34^b)  (b >= 0; b = 0 gives the 9 "
                                 "orientation-free coefficients incl. all non-spurious ones)",
            "qe2^a qe3^b mu2^j": "Q_B[t^b x^a mu2^j] / (mp12^a mp34^b)  (b >= 1)",
            "subtractions": "parent triangles (tri_to_bub), applied to Q_A (all slots) and Q_B (t >= 1)",
        },
        "tadpole": {
            "expansions": "A: q = -V0 + t e1 + (m^2+mu2)/t * e2/(2 mp12);  B: q = -V0 + t e3 + (m^2+mu2)/t * "
                          "e4/(2 mp34);  mindeg 3 both",
            "c0": "Q_A[t^0]", "c14": "Q_A[t^0 mu2]", "c15": "2 (Q_B[t^0 mu2] - c14) / mp34",
            "subtractions": "parent bubbles (bub_to_tad) and triangles (tri_to_tad), t^0 and t^0 mu2 slots",
        },
        "numerator_expansion_depths": {"box Mu": "3 terms (was 2 at r=n+1, 1 at r=n)",
                                       "triangle T3": "mindeg r-n+3 = 5 (12 values per orientation)",
                                       "bubble T2": "mindeg 4 (22 values) + swapped mindeg 3 (13 values)",
                                       "tadpole T3": "mindeg 3 (6 values) x 2 orientations"},
    }
    subs = json.load(open(os.path.join(HERE, "subtractions_rank_n2.json")))
    data["subtractions"] = {
        "description": SF.__doc__,
        "reuse": "the complete contributions of all rank<=n+1 monomials are exactly original Ninja's "
                 "x1cuts::correctbubcoeffs / correcttadcoeffs / correcttadcoeffsfull (verified to 1e-15 "
                 "against libninja); at r=n+2 they are reused unchanged, re-indexed by slot label [t,x,mu2] "
                 "into the rank n+2 layouts. The terms below are ADDED for the new monomials.",
        "symbols": {
            "tri_to_bub": "U_i = eb_i.u (u = et3 or et4 as stated), K_i = eb_i.k (eb1 = e1effl, eb2 = e2eff, "
                          "eb3 = growing e3, eb4 = e4eff), f, P0..P2 = Ninja bubble params",
            "tri_to_tad": "U3 = e.u, U4 = e4eff.u (e = growing tadpole vector), A_i = e.k_i, B_i = e4eff.k_i, "
                          "f0, f1, P = m^2",
            "bub_to_tad": "S_i = e.eb_i, T_i = e4eff.eb_i, W = eb2.kbol (kbol = V0_bubble - V0_tadpole), "
                          "A = e.k, B = e4eff.k, f, P = m^2"},
        "tri_to_bub": subs["tri_to_bub"], "tri_to_tad": subs["tri_to_tad"], "bub_to_tad": subs["bub_to_tad"],
    }
    data["integrated"] = {
        "box": "c0*D0 - c4/6 + c6*((m1+m2+m3+m4)/12 - (s12+s13+s14+s23+s24+s34)/60)",
        "triangle": "c0*C0 + c7/2 + c14*((s12+s23+s13)/24 - (m1+m2+m3)/6)   [unchanged from r=n+1]",
        "bubble": "c0*B0 + c1*ke2*B1 + c2*ke2^2*B11 + c13*ke2^3*B111 + c20*ke2^4*B1111 "
                  "+ c9*(3*(m1+m2)-s)/6 + c10*ke2*(s-2*m1-4*m2)/12 + c21*ke2^2*(5*m1+15*m2-3*s)/60 "
                  "- c22*(s^2-5*s*(m1+m2)+10*(m1^2+m1*m2+m2^2))/60 ;  ke2 = (V_i2-V_i1).e2, s = (V_i2-V_i1)^2",
        "tadpole": "(c0 + c15*mp34*m/4)*A0(m) + m^2/2*(c14 + c15*mp34/4)   [unchanged from r=n+1]",
        "pentagon": "0", "zero_point": "0",
        "rational_terms_general": {
            "I1[mu2]": "m^2/2 (m = m^2 of the line)",
            "I2[mu2]": "-(s - 3 m1 - 3 m2)/6",
            "I2[mu2 (qbar.e)]": "(p.e)(s - 2 m1 - 4 m2)/12   (any e)",
            "I2[mu2 (qbar.e)^2]": "(p.e)^2 (5 m1 + 15 m2 - 3 s)/60 + e^2 (s^2 - 5 s (m1+m2) + 10 (m1^2 + m1 m2 + "
                                  "m2^2))/120   (Ninja's e2 is null: e^2 = 0)",
            "I2[mu4]": "-(s^2 - 5 s (m1+m2) + 10 (m1^2 + m1 m2 + m2^2))/60",
            "I3[mu2]": "1/2", "I3[mu4]": "(s12+s13+s23)/24 - (m1+m2+m3)/6",
            "I4[mu4]": "-1/6", "I4[mu6]": "(m1+m2+m3+m4)/12 - (s12+s13+s14+s23+s24+s34)/60",
            "general": "I_n[mu^(2a) f] = (-eps)(1-eps)...(a-1-eps) I_n^(d+2a)[f] = -(a-1)! x Res_eps I_n^(d+2a)[f]",
        },
        "tensor_masters": {"I2[(qbar.e2)^a]": "(p.e2)^a B_{1^a}, a = 0..4 (e2 null kills all g^{mu nu} terms)",
                           "I1[(qbar.e3)(qbar.e4)]": "(e3.e4) A00 = (e3.e4)(m^2 A0/4 + m^4/8)"},
    }
    with open(os.path.join(HERE, "formulas_rank_n2.json"), "w") as fh:
        json.dump(data, fh, indent=1, default=str)
    print("wrote formulas_rank_n2.json")

    # ---------------------------------------------------------------- CSE code
    lines = ["// CSE-optimised NEW coefficient-level subtraction terms for rank r = n+2.",
             "// Add  divnum[slot] -= out[slot]  on top of original Ninja's x1cuts routines (re-indexed).",
             "// Slot labels [t,x,mu2]; symbols as in formulas_rank_n2.json 'subtractions.symbols'.",
             "// *_3 / *_4 suffix: projections on the parent triangle's et3 / et4.", ""]
    # tri->bub
    sym3 = {s: sp.Symbol(str(s) + "_3") for s in (SF.U1, SF.U2, SF.U3, SF.U4)}
    sym4 = {s: sp.Symbol(str(s) + "_4") for s in (SF.U1, SF.U2, SF.U3, SF.U4)}
    for kind, layout, idxs in (("tri_to_bub", SF.LAYOUT["bubble_rank_n+2"], range(15, 21)),
                               ("tri_to_tad", SF.LAYOUT["tadpole_rank_n+2"], range(15, 21)),
                               ("bub_to_tad", SF.LAYOUT["tadpole_rank_n+2"], range(20, 35))):
        tot = {s: 0 for s in layout}
        for idx in idxs:
            c = sp.Symbol(f"c{idx}")
            if kind == "bub_to_tad":
                F = SF.bub_formula(idx)
                for s, e in F.items():
                    tot[s] += c * e
            else:
                F, which = SF.tri_formula(idx, "bub" if kind == "tri_to_bub" else "tad")
                rep = sym3 if which == "et3" else sym4
                for s, e in F.items():
                    tot[s] += c * e.xreplace(rep)
        exprs = [sp.together(tot[s]) for s in layout]
        repl, red = sp.cse(exprs, symbols=sp.numbered_symbols("z"))
        lines.append(f"// ---------------- {kind}: {len(layout)} slots")
        for sym_, e in repl:
            lines.append(f"{sym_} = {sp.ccode(e)};")
        for s, e in zip(layout, red):
            if e != 0:
                lines.append(f"out[{list(s)}] = {sp.ccode(e)};")
        lines.append("")
    with open(os.path.join(HERE, "subtractions_rank_n2_cse.txt"), "w") as fh:
        fh.write("\n".join(lines))
    print(f"wrote subtractions_rank_n2_cse.txt ({len(lines)} lines)")


if __name__ == "__main__":
    main()
