//! Coefficient-level subtractions of original Ninja (`coefficient_level_subs.cc`,
//! FORM-generated), translated mechanically by `tools/fast/translate_subtractions.py`.
//! Do not edit by hand. Parent coefficients that original Ninja never sets
//! for a rank are folded as exact zeros; statement order is preserved.
//! Source SHA-256: cf408bc630a23ffa2a95a1f2479c3f6059d1f862fa6851897200deb788becc42.

#![allow(clippy::all)]

use super::complex::{C64, CMom, RMom, mp_cc, mp_cr, mp_rr};

/// `cuts::correctbubcoeffs` for rank - n = -2, all coefficients.
#[inline(never)]
#[allow(clippy::too_many_arguments, unused_variables)]
pub(super) fn bubble_from_triangle_rm2_all(
    divnum: &mut [C64],
    et3: &CMom,
    et4: &CMom,
    c: &[C64],
    eb3: &CMom,
    eb4: &CMom,
    param: &[C64; 3],
    eb1k: C64,
    eb2k: f64,
    eb3k: C64,
    ieb3k: C64,
    eb4k: C64,
    eb1t: &[C64; 4],
    f: C64,
) {
    let z1_1 = (ieb3k * ieb3k) * ieb3k;
    let z2_1 = eb3k;
    let z3_1 = mp_cc(eb3, et3);
    let z5_1 = mp_cc(eb3, et4);
    let z19_1 = 0.5 * c[1];
    let z19_2 = z3_1 * z19_1;
    let z20_2 = 0.5 * c[4];
    let z20_3 = z5_1 * z20_2;
    let z15_2 = z20_3 + z19_2;
    let z15_3 = z2_1 * z15_2;
    let z15_4 = z15_3;
    let z15_5 = z2_1 * z15_4;
    let z15_6 = z15_5;
    let z15_7 = z1_1 * z15_6;
    divnum[0] -= z15_7;
}

/// `cuts::correctbubcoeffs` for rank - n = -1, all coefficients.
#[inline(never)]
#[allow(clippy::too_many_arguments, unused_variables)]
pub(super) fn bubble_from_triangle_rm1_all(
    divnum: &mut [C64],
    et3: &CMom,
    et4: &CMom,
    c: &[C64],
    eb3: &CMom,
    eb4: &CMom,
    param: &[C64; 3],
    eb1k: C64,
    eb2k: f64,
    eb3k: C64,
    ieb3k: C64,
    eb4k: C64,
    eb1t: &[C64; 4],
    f: C64,
) {
    let z1_1 = (ieb3k * ieb3k) * ieb3k;
    let z2_1 = eb3k;
    let z3_1 = mp_cc(eb3, et3);
    let z5_1 = mp_cc(eb3, et4);
    let z8_1 = eb1k;
    let z9_1 = eb1t[0];
    let z10_1 = eb1t[1];
    let z11_1 = eb2k;
    let z12_1 = eb1t[2];
    let z13_1 = eb1t[3];
    let z14_1 = z3_1 * z3_1;
    let z18_1 = z5_1 * z5_1;
    let z20_1 = 0.5 * z1_1;
    let z21_1 = z20_1 * z2_1;
    let z23_1 = 0.5 * f;
    let z26_1 = c[5];
    let z27_1 = z26_1 * z18_1;
    let z29_1 = c[2];
    let z30_1 = z14_1 * z29_1;
    let z27_2 = z27_1 + z30_1;
    let z30_2 = z2_1 * z27_2;
    let z30_3 = z30_2;
    let z30_4 = z30_3 * z21_1;
    let z32_1 = (-z11_1) * z27_2;
    let z18_2 = z23_1 + z8_1;
    let z23_2 = z32_1;
    let z29_2 = (z12_1 * z3_1) * z29_1;
    let z26_2 = (z5_1 * z13_1) * z26_1;
    let z26_3 = z26_2 + z29_2;
    let z26_4 = z2_1 * z26_3;
    let z23_3 = z26_4 + (0.5 * z23_2);
    let z23_4 = z2_1 * z23_3;
    let z23_5 = z23_4;
    let z23_6 = z1_1 * z23_5;
    let z18_3 = (-z27_2) * z18_2;
    let z18_4 = z18_3;
    let z19_1 = c[2];
    let z19_2 = z9_1 * z19_1;
    let z19_3 = z19_2 + (0.5 * c[1]);
    let z19_4 = z3_1 * z19_3;
    let z20_2 = c[5];
    let z20_3 = z10_1 * z20_2;
    let z20_4 = z20_3 + (0.5 * c[4]);
    let z20_5 = z5_1 * z20_4;
    let z15_2 = z20_5 + z19_4;
    let z15_3 = z2_1 * z15_2;
    let z15_4 = (0.5 * z18_4) + z15_3;
    let z15_5 = z2_1 * z15_4;
    let z15_6 = z15_5;
    let z15_7 = z1_1 * z15_6;
    divnum[0] -= z30_4;
    divnum[2] -= z23_6;
    divnum[1] -= z15_7;
}

/// `cuts::correctbubcoeffs` for rank - n = -1, second-orientation coefficients.
#[inline(never)]
#[allow(clippy::too_many_arguments, unused_variables)]
pub(super) fn bubble_from_triangle_rm1_spurious(
    divnum: &mut [C64],
    et3: &CMom,
    et4: &CMom,
    c: &[C64],
    eb3: &CMom,
    eb4: &CMom,
    param: &[C64; 3],
    eb1k: C64,
    eb2k: f64,
    eb3k: C64,
    ieb3k: C64,
    eb4k: C64,
    eb1t: &[C64; 4],
    f: C64,
) {
    let z1_1 = mp_cc(eb3, et3);
    let z2_1 = ieb3k;
    let z3_1 = mp_cc(eb3, et4);
    let z11_1 = 0.5 * z2_1;
    let z13_1 = z3_1 * z3_1;
    let z16_1 = z1_1 * z1_1;
    let z16_2 = c[2] * z16_1;
    let z13_2 = c[5] * z13_1;
    let z10_1 = z13_2 + z16_2;
    let z10_2 = z10_1 * z11_1;
    divnum[0] -= z10_2;
}

/// `cuts::correctbubcoeffs` for rank - n = 0, all coefficients.
#[inline(never)]
#[allow(clippy::too_many_arguments, unused_variables)]
pub(super) fn bubble_from_triangle_r0_all(
    divnum: &mut [C64],
    et3: &CMom,
    et4: &CMom,
    c: &[C64],
    eb3: &CMom,
    eb4: &CMom,
    param: &[C64; 3],
    eb1k: C64,
    eb2k: f64,
    eb3k: C64,
    ieb3k: C64,
    eb4k: C64,
    eb1t: &[C64; 4],
    f: C64,
) {
    let z1_1 = (ieb3k * ieb3k) * ieb3k;
    let z2_1 = eb3k;
    let z3_1 = mp_cc(eb3, et3);
    let z4_1 = mp_cc(eb4, et3);
    let z5_1 = mp_cc(eb3, et4);
    let z6_1 = mp_cc(eb4, et4);
    let z7_1 = eb4k;
    let z8_1 = eb1k;
    let z9_1 = eb1t[0];
    let z10_1 = eb1t[1];
    let z11_1 = eb2k;
    let z12_1 = eb1t[2];
    let z13_1 = eb1t[3];
    let z14_1 = z3_1 * z3_1;
    let z15_1 = z14_1 * z4_1;
    let z16_1 = 3.0 * c[3];
    let z16_2 = z16_1 * z15_1;
    let z17_1 = 3.0 * c[6];
    let z18_1 = z5_1 * z5_1;
    let z17_2 = (z17_1 * z18_1) * z6_1;
    let z19_1 = z3_1 * c[8];
    let z20_1 = c[9] * z5_1;
    let z16_3 = ((z20_1 + z17_2) + z19_1) + z16_2;
    let z16_4 = z2_1 * z16_3;
    let z17_3 = c[3] * ((z3_1 * z3_1) * z3_1);
    let z19_2 = c[6] * ((z5_1 * z5_1) * z5_1);
    let z17_4 = z17_3 + z19_2;
    let z19_3 = z7_1 * z17_4;
    let z16_5 = z16_4 - z19_3;
    let z20_2 = 0.5 * z1_1;
    let z21_1 = z20_2 * z2_1;
    let z16_6 = z16_5 * z21_1;
    let z22_1 = z8_1 * z17_4;
    let z23_1 = 0.5 * f;
    let z24_1 = z23_1 * z17_4;
    let z24_2 = z22_1 + z24_1;
    let z25_1 = z10_1 * c[6];
    let z26_1 = c[5] + (3.0 * z25_1);
    let z27_1 = z26_1 * z18_1;
    let z28_1 = z9_1 * c[3];
    let z29_1 = c[2] + (3.0 * z28_1);
    let z30_1 = z14_1 * z29_1;
    let z27_2 = z27_1 + z30_1;
    let z30_2 = z2_1 * z27_2;
    let z30_3 = z30_2 - z24_2;
    let z30_4 = z30_3 * z21_1;
    let z31_1 = (z17_4 * (z2_1 * z2_1)) * z20_2;
    let z32_1 = (-z11_1) * z27_2;
    let z15_2 = z15_1 * c[3];
    let z18_2 = z18_1 * c[6];
    let z33_1 = z18_2 * z6_1;
    let z15_3 = z15_2 + z33_1;
    let z33_2 = 3.0 * z2_1;
    let z34_1 = z15_3 * z33_2;
    let z34_2 = z34_1 - z19_3;
    let z34_3 = param[1] * z34_2;
    let z18_3 = z18_2 * z13_1;
    let z14_2 = (z14_1 * z12_1) * c[3];
    let z14_3 = z18_3 + z14_2;
    let z18_4 = z23_1 + z8_1;
    let z23_2 = (-z14_3) * z18_4;
    let z23_3 = (z34_3 + (3.0 * z23_2)) + z32_1;
    let z29_2 = (z12_1 * z3_1) * z29_1;
    let z26_2 = (z5_1 * z13_1) * z26_1;
    let z26_3 = z26_2 + z29_2;
    let z26_4 = z2_1 * z26_3;
    let z23_4 = z26_4 + (0.5 * z23_3);
    let z23_5 = z2_1 * z23_4;
    let z24_3 = z11_1 * z24_2;
    let z23_6 = z24_3 + z23_5;
    let z23_7 = z1_1 * z23_6;
    let z24_4 = (-z11_1) * z17_4;
    let z26_5 = z14_3 * z33_2;
    let z24_5 = z24_4 + z26_5;
    let z21_2 = z24_5 * z21_1;
    let z24_6 = param[2] * z15_3;
    let z26_6 = (z5_1 * c[6]) * (z13_1 * z13_1);
    let z29_3 = (c[3] * z3_1) * (z12_1 * z12_1);
    let z24_7 = (z29_3 + z26_6) + z24_6;
    let z24_8 = z24_7 * z33_2;
    let z26_7 = (-param[2]) * z19_3;
    let z14_4 = z11_1 * z14_3;
    let z14_5 = (z24_8 - (3.0 * z14_4)) + z26_7;
    let z14_6 = z2_1 * z14_5;
    let z24_9 = z17_4 * (z11_1 * z11_1);
    let z14_7 = z24_9 + z14_6;
    let z14_8 = z14_7 * z20_2;
    let z18_5 = (-z27_2) * z18_4;
    let z19_4 = (-param[0]) * z19_3;
    let z18_6 = z19_4 + z18_5;
    let z15_4 = param[0] * z15_3;
    let z19_5 = c[2] + (1.5 * z28_1);
    let z19_6 = z9_1 * z19_5;
    let z19_7 = z19_6 + (0.5 * c[1]);
    let z19_8 = z3_1 * z19_7;
    let z20_3 = c[5] + (1.5 * z25_1);
    let z20_4 = z10_1 * z20_3;
    let z20_5 = z20_4 + (0.5 * c[4]);
    let z20_6 = z5_1 * z20_5;
    let z15_5 = ((1.5 * z15_4) + z20_6) + z19_8;
    let z15_6 = z2_1 * z15_5;
    let z15_7 = (0.5 * z18_6) + z15_6;
    let z15_8 = z2_1 * z15_7;
    let z18_7 = f * z17_4;
    let z18_8 = (0.25 * z18_7) + z22_1;
    let z18_9 = f * z18_8;
    let z17_5 = z17_4 * (z8_1 * z8_1);
    let z17_6 = z18_9 + z17_5;
    let z15_9 = (0.5 * z17_6) + z15_8;
    let z15_10 = z1_1 * z15_9;
    divnum[0] -= z31_1;
    divnum[2] -= z21_2;
    divnum[4] -= z16_6;
    divnum[6] -= z14_8;
    divnum[1] -= z30_4;
    divnum[5] -= z23_7;
    divnum[3] -= z15_10;
}

/// `cuts::correctbubcoeffs` for rank - n = 0, second-orientation coefficients.
#[inline(never)]
#[allow(clippy::too_many_arguments, unused_variables)]
pub(super) fn bubble_from_triangle_r0_spurious(
    divnum: &mut [C64],
    et3: &CMom,
    et4: &CMom,
    c: &[C64],
    eb3: &CMom,
    eb4: &CMom,
    param: &[C64; 3],
    eb1k: C64,
    eb2k: f64,
    eb3k: C64,
    ieb3k: C64,
    eb4k: C64,
    eb1t: &[C64; 4],
    f: C64,
) {
    let z1_1 = mp_cc(eb3, et3);
    let z2_1 = ieb3k;
    let z3_1 = mp_cc(eb3, et4);
    let z4_1 = eb2k;
    let z5_1 = eb1t[2];
    let z6_1 = eb1t[3];
    let z7_1 = eb1k;
    let z8_1 = eb1t[0];
    let z9_1 = eb1t[1];
    let z10_1 = c[3] * ((z1_1 * z1_1) * z1_1);
    let z11_1 = c[6] * ((z3_1 * z3_1) * z3_1);
    let z10_2 = z10_1 + z11_1;
    let z11_2 = 0.5 * z2_1;
    let z12_1 = z10_2 * z11_2;
    let z13_1 = z3_1 * z3_1;
    let z14_1 = z13_1 * c[6];
    let z15_1 = z6_1 * z14_1;
    let z16_1 = z1_1 * z1_1;
    let z17_1 = z16_1 * c[3];
    let z18_1 = z5_1 * z17_1;
    let z15_2 = z15_1 + z18_1;
    let z10_3 = z10_2 * z2_1;
    let z18_2 = (-z4_1) * z10_3;
    let z15_3 = (3.0 * z15_2) + z18_2;
    let z15_4 = z15_3 * z11_2;
    let z17_2 = z8_1 * z17_1;
    let z14_2 = z9_1 * z14_1;
    let z14_3 = z17_2 + z14_2;
    let z17_3 = (-z7_1) - (0.5 * f);
    let z10_4 = z17_3 * z10_3;
    let z16_2 = c[2] * z16_1;
    let z13_2 = c[5] * z13_1;
    let z10_5 = ((z10_4 + z13_2) + z16_2) + (3.0 * z14_3);
    let z10_6 = z10_5 * z11_2;
    divnum[0] -= z12_1;
    divnum[2] -= z15_4;
    divnum[1] -= z10_6;
}

/// `x1cuts::correctbubcoeffs` (rank n+1), all coefficients.
#[inline(never)]
#[allow(clippy::too_many_arguments, unused_variables)]
pub(super) fn bubble_from_triangle_x1_all(
    divnum: &mut [C64],
    et3: &CMom,
    et4: &CMom,
    c: &[C64],
    eb3: &CMom,
    eb4: &CMom,
    param: &[C64; 3],
    eb1k: C64,
    eb2k: f64,
    eb3k: C64,
    ieb3k: C64,
    eb4k: C64,
    eb1t: &[C64; 4],
    f: C64,
) {
    let z1_1 = ((ieb3k * ieb3k) * ieb3k) * ieb3k;
    let z2_1 = eb1k;
    let z3_1 = eb3k;
    let z4_1 = mp_cc(eb3, et3);
    let z5_1 = mp_cc(eb4, et3);
    let z6_1 = mp_cc(eb3, et4);
    let z7_1 = mp_cc(eb4, et4);
    let z8_1 = eb4k;
    let z9_1 = eb1t[0];
    let z10_1 = eb1t[1];
    let z11_1 = eb2k;
    let z12_1 = eb1t[2];
    let z13_1 = eb1t[3];
    let z14_1 = z9_1 * c[12];
    let z15_1 = (2.0 * z14_1) + (0.5 * c[3]);
    let z16_1 = z4_1 * z4_1;
    let z17_1 = z16_1 * z5_1;
    let z18_1 = z17_1 * z15_1;
    let z19_1 = 0.5 * c[6];
    let z20_1 = z6_1 * z6_1;
    let z21_1 = z19_1 * z20_1;
    let z22_1 = z21_1 * z7_1;
    let z18_2 = z18_1 + z22_1;
    let z22_2 = 3.0 * z3_1;
    let z18_3 = z18_2 * z22_2;
    let z23_1 = (z4_1 * z4_1) * z4_1;
    let z24_1 = z15_1 * z23_1;
    let z25_1 = (z6_1 * z6_1) * z6_1;
    let z26_1 = z19_1 * z25_1;
    let z24_2 = z24_1 + z26_1;
    let z26_2 = z24_2 * z8_1;
    let z18_4 = z18_3 - z26_2;
    let z26_3 = z3_1 * z3_1;
    let z18_5 = z18_4 * z26_3;
    let z27_1 = z7_1 * c[13];
    let z28_1 = z27_1 * z25_1;
    let z29_1 = z23_1 * c[12];
    let z30_1 = z29_1 * z5_1;
    let z28_2 = z28_1 + z30_1;
    let z30_2 = 2.0 * z3_1;
    let z31_1 = z28_2 * z30_2;
    let z32_1 = c[12] * (z16_1 * z16_1);
    let z33_1 = c[13] * (((z6_1 * z6_1) * z6_1) * z6_1);
    let z32_2 = z32_1 + z33_1;
    let z33_2 = z32_2 * z8_1;
    let z33_3 = z31_1 - z33_2;
    let z34_1 = z2_1 * z3_1;
    let z35_1 = z33_3 * z34_1;
    let z28_3 = z28_2 * z3_1;
    let z36_1 = 0.5 * z32_2;
    let z37_1 = z36_1 * z8_1;
    let z28_4 = z28_3 - z37_1;
    let z38_1 = f * z3_1;
    let z39_1 = z28_4 * z38_1;
    let z27_2 = z27_1 * z20_1;
    let z40_1 = z27_2 * z22_2;
    let z41_1 = z25_1 * c[13];
    let z42_1 = z41_1 * z8_1;
    let z40_2 = z42_1 - z40_1;
    let z42_2 = 2.0 * z26_3;
    let z40_3 = z40_2 * z42_2;
    let z43_1 = z40_3 * z10_1;
    let z35_2 = ((z39_1 - z18_5) + z35_1) + z43_1;
    let z39_2 = 0.5 * z16_1;
    let z43_2 = z26_3 * z2_1;
    let z44_1 = (-z43_2) * z39_2;
    let z45_1 = z26_3 * f;
    let z46_1 = z16_1 * z45_1;
    let z47_1 = (z3_1 * z3_1) * z3_1;
    let z48_1 = z47_1 * z4_1;
    let z49_1 = z9_1 * z48_1;
    let z44_2 = ((-0.25 * z46_1) + z49_1) + z44_1;
    let z44_3 = c[10] * z44_2;
    let z46_2 = z20_1 * z26_3;
    let z49_2 = 0.5 * f;
    let z50_1 = z49_2 + z2_1;
    let z50_2 = z46_2 * z50_1;
    let z51_1 = z47_1 * z6_1;
    let z52_1 = z10_1 * z51_1;
    let z52_2 = (-0.5 * z50_2) + z52_1;
    let z52_3 = c[11] * z52_2;
    let z53_1 = 0.5 * z48_1;
    let z54_1 = c[8] * z53_1;
    let z55_1 = 0.5 * z51_1;
    let z56_1 = c[9] * z55_1;
    let z44_4 = (((z56_1 + z44_3) + z52_3) + z54_1) - z35_2;
    let z44_5 = z1_1 * z44_4;
    let z52_4 = z36_1 * z34_1;
    let z54_2 = z14_1 + (0.25 * c[3]);
    let z56_2 = z54_2 * z23_1;
    let z25_2 = z25_1 * c[6];
    let z56_3 = z56_2 + (0.25 * z25_2);
    let z57_1 = (-z56_3) * z26_3;
    let z58_1 = z32_2 * z38_1;
    let z57_2 = ((0.125 * z58_1) + z57_1) + z52_4;
    let z57_3 = f * z57_2;
    let z58_2 = z42_2 * z2_1;
    let z45_2 = z58_2 + z45_1;
    let z58_3 = (-z41_1) * z45_2;
    let z59_1 = z20_1 * c[6];
    let z60_1 = z47_1 * z59_1;
    let z61_1 = 3.0 * z47_1;
    let z62_1 = z61_1 * z20_1;
    let z63_1 = (z10_1 * c[13]) * z62_1;
    let z58_4 = (z63_1 + (1.5 * z60_1)) + z58_3;
    let z58_5 = z10_1 * z58_4;
    let z31_2 = z37_1 - z31_1;
    let z31_3 = z31_2 * z26_3;
    let z37_2 = (-param[0]) * z31_3;
    let z60_2 = z24_2 * z26_3;
    let z52_5 = (-z60_2) + z52_4;
    let z52_6 = z2_1 * z52_5;
    let z63_2 = z9_1 * z9_1;
    let z64_1 = z63_2 * c[12];
    let z65_1 = c[3] * z9_1;
    let z64_2 = ((3.0 * z64_1) + (1.5 * z65_1)) + (0.5 * c[2]);
    let z64_3 = z64_2 * z16_1;
    let z66_1 = 0.5 * z20_1;
    let z67_1 = z66_1 * c[5];
    let z64_4 = z64_3 + z67_1;
    let z67_2 = z64_4 * z47_1;
    let z37_3 = (((z58_5 + z57_3) + z52_6) + z67_2) + z37_2;
    let z37_4 = z1_1 * z37_3;
    let z52_7 = c[11] * z66_1;
    let z57_4 = c[10] * z39_2;
    let z52_8 = z57_4 + z52_7;
    let z52_9 = z47_1 * z52_8;
    let z52_10 = (-z31_3) + z52_9;
    let z52_11 = z1_1 * z52_10;
    let z57_5 = 2.0 * z10_1;
    let z58_6 = z41_1 * z57_5;
    let z58_7 = z58_6 + z24_2;
    let z58_8 = z47_1 * z58_7;
    let z66_2 = (-z36_1) * z43_2;
    let z67_3 = z32_2 * f;
    let z68_1 = z26_3 * z67_3;
    let z58_9 = ((-0.25 * z68_1) + z66_2) + z58_8;
    let z58_10 = z1_1 * z58_9;
    let z66_3 = (z1_1 * z47_1) * z36_1;
    let z68_2 = param[0] * z3_1;
    let z69_1 = z68_2 * z33_3;
    let z64_5 = z64_4 * z26_3;
    let z69_2 = z64_5 + z69_1;
    let z70_1 = z32_2 * z2_1;
    let z71_1 = 1.5 * z70_1;
    let z14_2 = (4.0 * z14_1) + c[3];
    let z14_3 = z14_2 * z23_1;
    let z14_4 = z25_2 + z14_3;
    let z14_5 = z3_1 * z14_4;
    let z14_6 = z14_5 - z71_1;
    let z14_7 = z2_1 * z14_6;
    let z23_2 = z24_2 * z3_1;
    let z24_3 = z71_1 - z23_2;
    let z25_3 = (-0.375 * z67_3) - z24_3;
    let z25_4 = f * z25_3;
    let z14_8 = (z25_4 + z14_7) - z69_2;
    let z14_9 = z11_1 * z14_8;
    let z25_5 = (-param[1]) * z35_2;
    let z29_2 = z29_1 * z12_1;
    let z71_2 = z41_1 * z13_1;
    let z29_3 = z29_2 + z71_2;
    let z71_3 = z29_3 * z30_2;
    let z72_1 = z71_3 * z2_1;
    let z73_1 = z16_1 * z12_1;
    let z54_3 = (-z54_2) * z73_1;
    let z59_2 = z13_1 * z59_1;
    let z54_4 = (-0.25 * z59_2) + z54_3;
    let z59_3 = 3.0 * z26_3;
    let z54_5 = z54_4 * z59_3;
    let z74_1 = 0.5 * z38_1;
    let z75_1 = z29_3 * z74_1;
    let z54_6 = (z75_1 + z54_5) + z72_1;
    let z54_7 = f * z54_6;
    let z75_2 = z13_1 * c[13];
    let z76_1 = z75_2 * z51_1;
    let z77_1 = z46_2 * z11_1;
    let z78_1 = (-c[13]) * z77_1;
    let z76_2 = (2.0 * z76_1) + z78_1;
    let z76_3 = z10_1 * z76_2;
    let z78_2 = z46_2 * z75_2;
    let z79_1 = (-f) - (2.0 * z2_1);
    let z79_2 = z78_2 * z79_1;
    let z80_1 = z51_1 * z13_1;
    let z81_1 = c[6] * z80_1;
    let z76_4 = (z76_3 + z81_1) + z79_2;
    let z46_3 = z46_2 * c[6];
    let z79_3 = 1.5 * z46_3;
    let z81_2 = 4.0 * z34_1;
    let z82_1 = f * z30_2;
    let z82_2 = z82_1 + z81_2;
    let z82_3 = z41_1 * z82_2;
    let z82_4 = (-z79_3) + z82_3;
    let z82_5 = z11_1 * z82_4;
    let z76_5 = z82_5 + (3.0 * z76_4);
    let z76_6 = z10_1 * z76_5;
    let z27_3 = z27_2 * z13_1;
    let z82_6 = c[12] * z12_1;
    let z17_2 = z17_1 * z82_6;
    let z17_3 = z27_3 + z17_2;
    let z17_4 = z17_3 * z22_2;
    let z22_3 = z29_3 * z8_1;
    let z17_5 = z17_4 - z22_3;
    let z17_6 = z17_5 * z42_2;
    let z22_4 = param[0] * z17_6;
    let z27_4 = z15_1 * z73_1;
    let z21_2 = z21_1 * z13_1;
    let z21_3 = z27_4 + z21_2;
    let z27_5 = z21_3 * z59_3;
    let z72_2 = (-z27_5) + z72_1;
    let z72_3 = z2_1 * z72_2;
    let z73_2 = z63_2 * z82_6;
    let z65_2 = (3.0 * z65_1) + c[2];
    let z65_3 = z12_1 * z65_2;
    let z65_4 = (6.0 * z73_2) + z65_3;
    let z65_5 = z4_1 * z65_4;
    let z73_3 = (c[5] * z6_1) * z13_1;
    let z65_6 = z73_3 + z65_5;
    let z65_7 = z65_6 * z47_1;
    let z14_10 = (((((z25_5 + z76_6) + z14_9) + z54_7) + z72_3) + z65_7) + z22_4;
    let z14_11 = z1_1 * z14_10;
    let z22_5 = z11_1 * z3_1;
    let z25_6 = z22_5 * z33_3;
    let z17_7 = z17_6 - z25_6;
    let z25_7 = z80_1 - (0.5 * z77_1);
    let z25_8 = c[11] * z25_7;
    let z33_4 = z26_3 * z11_1;
    let z39_3 = (-z33_4) * z39_2;
    let z54_8 = z12_1 * z48_1;
    let z39_4 = z54_8 + z39_3;
    let z39_5 = c[10] * z39_4;
    let z25_9 = (z39_5 + z25_8) + z17_7;
    let z25_10 = z1_1 * z25_9;
    let z32_3 = z32_2 * z34_1;
    let z34_2 = z36_1 * z38_1;
    let z32_4 = (z34_2 - z60_2) + z32_3;
    let z32_5 = z11_1 * z32_4;
    let z34_3 = (-z29_3) * z45_2;
    let z21_4 = z21_3 * z61_1;
    let z38_2 = z75_2 * z62_1;
    let z39_6 = (-z41_1) * z33_4;
    let z38_3 = z38_2 + z39_6;
    let z38_4 = z38_3 * z57_5;
    let z39_7 = (-param[1]) * z31_3;
    let z21_5 = (((z39_7 + z38_4) + z32_5) + z21_4) + z34_3;
    let z21_6 = z1_1 * z21_5;
    let z32_6 = 2.0 * z47_1;
    let z34_4 = z29_3 * z32_6;
    let z33_5 = (-z36_1) * z33_4;
    let z33_6 = z34_4 + z33_5;
    let z33_7 = z1_1 * z33_6;
    let z24_4 = (-0.75 * z67_3) - z24_3;
    let z24_5 = z11_1 * z24_4;
    let z34_5 = z29_3 * z81_2;
    let z38_5 = f * z71_3;
    let z24_6 = ((z24_5 + z38_5) - z27_5) + z34_5;
    let z24_7 = z11_1 * z24_6;
    let z27_6 = (-param[2]) * z35_2;
    let z34_6 = z12_1 * z12_1;
    let z16_2 = (c[12] * z34_6) * z16_1;
    let z35_3 = z13_1 * z13_1;
    let z38_6 = z35_3 * c[13];
    let z20_2 = z38_6 * z20_1;
    let z16_3 = z16_2 + z20_2;
    let z20_3 = (-z26_3) * z49_2;
    let z20_4 = z20_3 - z43_2;
    let z20_5 = z16_3 * z20_4;
    let z26_4 = (z6_1 * z35_3) * z19_1;
    let z15_2 = (z4_1 * z34_6) * z15_1;
    let z15_3 = z26_4 + z15_2;
    let z15_4 = z15_3 * z47_1;
    let z15_5 = z15_4 + z20_5;
    let z20_6 = param[1] * z17_7;
    let z26_5 = z41_1 * z22_5;
    let z26_6 = (-3.0 * z78_2) + z26_5;
    let z26_7 = z11_1 * z26_6;
    let z34_7 = z51_1 * z38_6;
    let z26_8 = (3.0 * z34_7) + z26_7;
    let z26_9 = z26_8 * z57_5;
    let z15_6 = (((z27_6 + z20_6) + z26_9) + (3.0 * z15_5)) + z24_7;
    let z15_7 = z1_1 * z15_6;
    let z20_7 = z16_3 * z61_1;
    let z24_8 = (-z29_3) * z42_2;
    let z22_6 = z36_1 * z22_5;
    let z22_7 = z24_8 + z22_6;
    let z22_8 = z11_1 * z22_7;
    let z24_9 = (-param[2]) * z31_3;
    let z20_8 = (z24_9 + z20_7) + z22_8;
    let z20_9 = z1_1 * z20_8;
    let z16_4 = (-z16_3) * z59_3;
    let z22_9 = (-z11_1) * z36_1;
    let z22_10 = z71_3 + z22_9;
    let z22_11 = z11_1 * z22_10;
    let z16_5 = z16_4 + z22_11;
    let z16_6 = z11_1 * z16_5;
    let z17_8 = param[2] * z17_7;
    let z22_12 = (z6_1 * c[13]) * ((z13_1 * z13_1) * z13_1);
    let z24_10 = (z4_1 * c[12]) * ((z12_1 * z12_1) * z12_1);
    let z22_13 = z22_12 + z24_10;
    let z22_14 = z22_13 * z32_6;
    let z16_7 = (z17_8 + z22_14) + z16_6;
    let z16_8 = z1_1 * z16_7;
    let z17_9 = 0.75 * z70_1;
    let z22_15 = z3_1 * z56_3;
    let z22_16 = ((-0.125 * z67_3) + z22_15) - z17_9;
    let z22_17 = z22_16 * z49_2;
    let z24_11 = (-z28_4) * z68_2;
    let z17_10 = z23_2 - z17_9;
    let z17_11 = z2_1 * z17_10;
    let z17_12 = ((z22_17 + z17_11) - (0.5 * z64_5)) + z24_11;
    let z17_13 = f * z17_12;
    let z22_18 = z30_2 * z2_1;
    let z24_12 = z41_1 * z22_18;
    let z24_13 = (-z79_3) + z24_12;
    let z24_14 = z2_1 * z24_13;
    let z22_19 = z74_1 + z22_18;
    let z22_20 = z41_1 * z22_19;
    let z22_21 = (-0.75 * z46_3) + z22_20;
    let z22_22 = f * z22_21;
    let z26_10 = (-c[13]) * z50_2;
    let z19_2 = z19_1 * z51_1;
    let z19_3 = z19_2 + z26_10;
    let z26_11 = (c[13] * z51_1) * z57_5;
    let z19_4 = (3.0 * z19_3) + z26_11;
    let z19_5 = z10_1 * z19_4;
    let z26_12 = (-param[0]) * z40_3;
    let z27_7 = c[5] * z51_1;
    let z19_6 = (((z19_5 + z22_22) + z24_14) + z27_7) + z26_12;
    let z19_7 = z10_1 * z19_6;
    let z18_6 = param[0] * z18_5;
    let z22_23 = (-z2_1) * z36_1;
    let z22_24 = z23_2 + z22_23;
    let z22_25 = z2_1 * z22_24;
    let z22_26 = z22_25 - z69_2;
    let z22_27 = z2_1 * z22_26;
    let z23_3 = 2.0 * c[12];
    let z23_4 = z63_2 * z23_3;
    let z23_5 = c[2] + z23_4;
    let z23_6 = z9_1 * z23_5;
    let z24_15 = 1.5 * c[3];
    let z24_16 = z63_2 * z24_15;
    let z23_7 = z23_6 + z24_16;
    let z23_8 = z23_7 * z48_1;
    let z24_17 = c[4] * z55_1;
    let z26_13 = c[1] * z53_1;
    let z17_14 = (((((z26_13 + z19_7) + z24_17) + z17_13) + z22_27) + z23_8) + z18_6;
    let z17_15 = z1_1 * z17_14;
    divnum[7] -= z17_15;
    divnum[8] -= z44_5;
    divnum[3] -= z37_4;
    divnum[4] -= z52_11;
    divnum[1] -= z58_10;
    divnum[0] -= z66_3;
    divnum[9] -= z14_11;
    divnum[10] -= z25_10;
    divnum[5] -= z21_6;
    divnum[2] -= z33_7;
    divnum[11] -= z15_7;
    divnum[6] -= z20_9;
    divnum[12] -= z16_8;
}

/// `x1cuts::correctbubcoeffs` (rank n+1), second-orientation coefficients.
#[inline(never)]
#[allow(clippy::too_many_arguments, unused_variables)]
pub(super) fn bubble_from_triangle_x1_spurious(
    divnum: &mut [C64],
    et3: &CMom,
    et4: &CMom,
    c: &[C64],
    eb3: &CMom,
    eb4: &CMom,
    param: &[C64; 3],
    eb1k: C64,
    eb2k: f64,
    eb3k: C64,
    ieb3k: C64,
    eb4k: C64,
    eb1t: &[C64; 4],
    f: C64,
) {
    let z1_1 = ((ieb3k * ieb3k) * ieb3k) * ieb3k;
    let z2_1 = eb1k;
    let z3_1 = eb3k;
    let z4_1 = mp_cc(eb3, et3);
    let z5_1 = mp_cc(eb3, et4);
    let z6_1 = eb1t[0];
    let z7_1 = eb1t[1];
    let z8_1 = mp_cc(eb4, et3);
    let z9_1 = mp_cc(eb4, et4);
    let z10_1 = eb4k;
    let z11_1 = eb2k;
    let z12_1 = eb1t[2];
    let z13_1 = eb1t[3];
    let z14_1 = z5_1 * z5_1;
    let z15_1 = c[13] * z1_1;
    let z16_1 = z14_1 * z15_1;
    let z17_1 = z7_1 * z16_1;
    let z18_1 = 0.5 * z1_1;
    let z19_1 = z14_1 * z18_1;
    let z20_1 = c[6] * z19_1;
    let z20_2 = z20_1 + z17_1;
    let z20_3 = z7_1 * z20_2;
    let z21_1 = z4_1 * z4_1;
    let z22_1 = z21_1 * z18_1;
    let z23_1 = (z6_1 * c[3]) * z22_1;
    let z24_1 = c[12] * z1_1;
    let z25_1 = z21_1 * z24_1;
    let z26_1 = (z6_1 * z6_1) * z25_1;
    let z20_4 = (z20_3 + z23_1) + z26_1;
    let z23_2 = (z4_1 * z4_1) * z4_1;
    let z26_2 = z23_2 * z24_1;
    let z27_1 = z26_2 * z8_1;
    let z28_1 = (z5_1 * z5_1) * z5_1;
    let z29_1 = z28_1 * z15_1;
    let z30_1 = z29_1 * z9_1;
    let z27_2 = z27_1 + z30_1;
    let z30_2 = 2.0 * z27_2;
    let z31_1 = param[0] * z30_2;
    let z32_1 = c[2] * z22_1;
    let z20_5 = (z32_1 + (3.0 * z20_4)) + z31_1;
    let z20_6 = z3_1 * z20_5;
    let z31_2 = z26_2 * z6_1;
    let z32_2 = z29_1 * z7_1;
    let z31_3 = z31_2 + z32_2;
    let z28_2 = z28_1 * c[6];
    let z23_3 = z23_2 * c[3];
    let z23_4 = z28_2 + z23_3;
    let z28_3 = z1_1 * z23_4;
    let z28_4 = (-0.25 * z28_3) - z31_3;
    let z28_5 = f * z28_4;
    let z15_2 = z15_1 * (z14_1 * z14_1);
    let z32_3 = z24_1 * (((z4_1 * z4_1) * z4_1) * z4_1);
    let z15_3 = z15_2 + z32_3;
    let z32_4 = z10_1 * z15_3;
    let z32_5 = 0.5 * z32_4;
    let z33_1 = (-param[0]) * z32_5;
    let z20_7 = (z20_6 + z33_1) + z28_5;
    let z20_8 = z3_1 * z20_7;
    let z23_5 = z23_4 * z18_1;
    let z23_6 = z23_5 + (2.0 * z31_3);
    let z28_6 = z23_6 * z3_1;
    let z31_4 = 0.5 * z15_3;
    let z33_2 = z31_4 * z2_1;
    let z28_7 = z28_6 - z33_2;
    let z33_3 = z31_4 * f;
    let z34_1 = z33_3 - z28_7;
    let z34_2 = z2_1 * z34_1;
    let z35_1 = z15_3 * (f * f);
    let z20_9 = (z34_2 + (0.125 * z35_1)) + z20_8;
    let z20_10 = z3_1 * z20_9;
    let z34_3 = (z3_1 * z3_1) * z3_1;
    let z35_2 = (c[5] * z34_3) * z19_1;
    let z20_11 = z35_2 + z20_10;
    let z19_2 = c[11] * z19_1;
    let z22_2 = c[10] * z22_1;
    let z19_3 = (z22_2 + z19_2) + z30_2;
    let z19_4 = z3_1 * z19_3;
    let z19_5 = (-z32_5) + z19_4;
    let z22_3 = z3_1 * z3_1;
    let z19_6 = z19_5 * z22_3;
    let z35_3 = z15_3 * f;
    let z28_8 = (-0.25 * z35_3) + z28_7;
    let z28_9 = z22_3 * z28_8;
    let z34_4 = z34_3 * z31_4;
    let z21_2 = z21_1 * z12_1;
    let z24_2 = (z21_2 * z24_1) * z6_1;
    let z17_2 = z13_1 * z17_1;
    let z17_3 = z24_2 + z17_2;
    let z14_2 = (c[6] * z14_1) * z13_1;
    let z21_3 = c[3] * z21_2;
    let z14_3 = z14_2 + z21_3;
    let z14_4 = z14_3 * z18_1;
    let z14_5 = z14_4 + (2.0 * z17_3);
    let z17_4 = param[1] * z30_2;
    let z14_6 = (3.0 * z14_5) + z17_4;
    let z14_7 = z3_1 * z14_6;
    let z17_5 = (-z11_1) * z23_6;
    let z18_2 = z26_2 * z12_1;
    let z21_4 = z29_1 * z13_1;
    let z18_3 = z18_2 + z21_4;
    let z21_5 = (-f) * z18_3;
    let z23_7 = (-param[1]) * z32_5;
    let z14_8 = ((z14_7 + z21_5) + z17_5) + z23_7;
    let z14_9 = z3_1 * z14_8;
    let z17_6 = 2.0 * z18_3;
    let z18_4 = z17_6 * z3_1;
    let z15_4 = z15_3 * z11_1;
    let z15_5 = z15_4 - z18_4;
    let z15_6 = z2_1 * z15_5;
    let z21_6 = z11_1 * z33_3;
    let z14_10 = (z15_6 + z21_6) + z14_9;
    let z14_11 = z3_1 * z14_10;
    let z15_7 = (-z11_1) * z31_4;
    let z15_8 = z15_7 + z18_4;
    let z15_9 = z15_8 * z22_3;
    let z17_7 = (-z11_1) * z17_6;
    let z18_5 = (z12_1 * z12_1) * z25_1;
    let z16_2 = (z13_1 * z13_1) * z16_1;
    let z16_3 = z18_5 + z16_2;
    let z16_4 = z3_1 * z16_3;
    let z16_5 = z17_7 + (3.0 * z16_4);
    let z16_6 = z3_1 * z16_5;
    let z17_8 = (z11_1 * z11_1) * z31_4;
    let z16_7 = z17_8 + z16_6;
    let z16_8 = z3_1 * z16_7;
    let z17_9 = 2.0 * z3_1;
    let z17_10 = z27_2 * z17_9;
    let z17_11 = (-z32_5) + z17_10;
    let z17_12 = (param[2] * z17_11) * z22_3;
    let z16_9 = z16_8 + z17_12;
    divnum[3] -= z20_11;
    divnum[4] -= z19_6;
    divnum[1] -= z28_9;
    divnum[0] -= z34_4;
    divnum[5] -= z14_11;
    divnum[2] -= z15_9;
    divnum[6] -= z16_9;
}

/// `x1cuts::correcttadcoeffs` (triangle parent, rank n+1).
#[inline(never)]
#[allow(clippy::too_many_arguments, unused_variables)]
pub(super) fn tadpole_from_triangle_x1(
    divnum: &mut [C64],
    et3: &CMom,
    et4: &CMom,
    c: &[C64],
    e3: &CMom,
    e4: &CMom,
    param: C64,
    e3k0: C64,
    ie3k0: C64,
    e4k0: C64,
    f0: C64,
    e3k1: C64,
    ie3k1: C64,
    e4k1: C64,
    f1: C64,
) {
    let z1_1 = ((((ie3k0 * ie3k0) * ie3k0) * ie3k1) * ie3k1) * ie3k1;
    let z2_1 = mp_cc(e3, et3);
    let z3_1 = e3k0;
    let z4_1 = e3k1;
    let z5_1 = e4k1;
    let z6_1 = e4k0;
    let z7_1 = mp_cc(e4, et3);
    let z8_1 = mp_cc(e3, et4);
    let z9_1 = mp_cc(e4, et4);
    let z10_1 = z3_1 * z3_1;
    let z11_1 = z10_1 * z4_1;
    let z12_1 = z5_1 * z11_1;
    let z13_1 = z4_1 * z4_1;
    let z14_1 = (z13_1 * z6_1) * z3_1;
    let z12_2 = z12_1 + z14_1;
    let z14_2 = 0.25 * z8_1;
    let z14_3 = z14_2 * z12_2;
    let z15_1 = z4_1 * z3_1;
    let z16_1 = z15_1 * z15_1;
    let z17_1 = z16_1 * z9_1;
    let z14_4 = z14_3 - z17_1;
    let z14_5 = c[13] * z14_4;
    let z17_2 = (-z8_1) * z14_5;
    let z18_1 = 0.25 * z16_1;
    let z19_1 = c[11] * z18_1;
    let z17_3 = z19_1 + z17_2;
    let z19_2 = z8_1 * z8_1;
    let z17_4 = z17_3 * z19_2;
    let z20_1 = z2_1 * c[12];
    let z12_3 = z12_2 * z20_1;
    let z21_1 = (z7_1 * z16_1) * c[12];
    let z12_4 = (-z21_1) + (0.25 * z12_3);
    let z21_2 = (-z2_1) * z12_4;
    let z18_2 = c[10] * z18_1;
    let z18_3 = z18_2 + z21_2;
    let z21_3 = z2_1 * z2_1;
    let z18_4 = z18_3 * z21_3;
    let z17_5 = z17_4 + z18_4;
    let z17_6 = z1_1 * z17_5;
    let z11_2 = f1 * z11_1;
    let z13_2 = z13_1 * f0;
    let z18_5 = z13_2 * z3_1;
    let z11_3 = z11_2 + z18_5;
    let z18_6 = (-c[3]) * z11_3;
    let z15_2 = z15_1 * f1;
    let z13_3 = z15_2 + z13_2;
    let z13_4 = z13_3 * f0;
    let z10_2 = z10_1 * (f1 * f1);
    let z10_3 = z10_2 + z13_4;
    let z13_5 = z10_3 * z20_1;
    let z13_6 = (0.5 * z13_5) + z18_6;
    let z13_7 = z2_1 * z13_6;
    let z15_3 = c[2] * z16_1;
    let z13_8 = z15_3 + (0.5 * z13_7);
    let z13_9 = z13_8 * z21_3;
    let z10_4 = (z19_2 * c[13]) * z10_3;
    let z15_4 = c[5] * z16_1;
    let z10_5 = z15_4 + (0.25 * z10_4);
    let z10_6 = z10_5 * z19_2;
    let z15_5 = (z8_1 * z8_1) * z8_1;
    let z11_4 = (c[6] * z11_3) * z15_5;
    let z10_7 = (z13_9 + z10_6) - (0.5 * z11_4);
    let z11_5 = (-z14_5) * z15_5;
    let z12_5 = (-z12_4) * ((z2_1 * z2_1) * z2_1);
    let z11_6 = z11_5 + z12_5;
    let z11_7 = param * z11_6;
    let z10_8 = (0.25 * z10_7) + z11_7;
    let z10_9 = z1_1 * z10_8;
    divnum[3] -= z17_6;
    divnum[2] -= z10_9;
}

/// `x1cuts::correcttadcoeffs` (bubble parent, rank n+1).
#[inline(never)]
#[allow(clippy::too_many_arguments, unused_variables)]
pub(super) fn tadpole_from_bubble_x1(
    divnum: &mut [C64],
    eb2: &RMom,
    eb3: &CMom,
    eb4: &CMom,
    kbol: &RMom,
    c: &[C64],
    e3: &CMom,
    e4: &CMom,
    param: C64,
    e3k: C64,
    ie3k: C64,
    e4k: C64,
    f: C64,
) {
    let z1_1 = (ie3k * ie3k) * ie3k;
    let z2_1 = e3k;
    let z3_1 = mp_cr(e3, eb2);
    let z4_1 = mp_cr(e4, eb2);
    let z5_1 = mp_cc(e4, eb3);
    let z6_1 = mp_cc(e4, eb4);
    let z7_1 = mp_cc(e3, eb3);
    let z8_1 = mp_cc(e3, eb4);
    let z9_1 = e4k;
    let z10_1 = mp_rr(eb2, kbol);
    let z11_1 = z4_1 * z8_1;
    let z12_1 = 0.5 * z3_1;
    let z13_1 = z12_1 * z6_1;
    let z11_2 = z11_1 + z13_1;
    let z11_3 = z11_2 * c[17];
    let z13_2 = z4_1 * z7_1;
    let z12_2 = z12_1 * z5_1;
    let z12_3 = z13_2 + z12_2;
    let z12_4 = z12_3 * c[16];
    let z13_3 = (c[19] * z6_1) * z8_1;
    let z14_1 = (z5_1 * c[18]) * z7_1;
    let z15_1 = (c[13] * z3_1) * z4_1;
    let z11_4 = (((z14_1 + z13_3) + (1.5 * z15_1)) + z11_3) + z12_4;
    let z12_5 = (0.5 * c[10]) + z11_4;
    let z12_6 = z3_1 * z12_5;
    let z13_4 = z7_1 * z7_1;
    let z14_2 = (z5_1 * z13_4) * c[14];
    let z15_2 = z8_1 * z8_1;
    let z16_1 = (c[15] * z15_2) * z6_1;
    let z14_3 = z14_2 + z16_1;
    let z16_2 = z13_4 * c[18];
    let z17_1 = z15_2 * c[19];
    let z16_3 = z16_2 + z17_1;
    let z17_2 = z16_3 * z4_1;
    let z14_4 = z17_2 + (3.0 * z14_3);
    let z17_3 = z7_1 * c[11];
    let z18_1 = c[12] * z8_1;
    let z17_4 = (z18_1 + z17_3) + z14_4;
    let z12_7 = z12_6 + (0.5 * z17_4);
    let z12_8 = z2_1 * z12_7;
    let z17_5 = z3_1 * z3_1;
    let z18_2 = z17_5 * c[13];
    let z18_3 = z16_3 + z18_2;
    let z18_4 = z18_3 * z3_1;
    let z19_1 = c[17] * z8_1;
    let z20_1 = c[16] * z7_1;
    let z19_2 = z19_1 + z20_1;
    let z17_6 = z17_5 * z19_2;
    let z20_2 = c[14] * ((z7_1 * z7_1) * z7_1);
    let z21_1 = c[15] * ((z8_1 * z8_1) * z8_1);
    let z17_7 = ((z18_4 + z21_1) + z17_6) + z20_2;
    let z18_5 = z17_7 * z9_1;
    let z12_9 = (-0.5 * z18_5) + z12_8;
    let z20_3 = z2_1 * z1_1;
    let z12_10 = z12_9 * z20_3;
    let z16_4 = (-z16_3) * z10_1;
    let z13_5 = (-c[4]) * z13_4;
    let z15_3 = (-c[6]) * z15_2;
    let z13_6 = (z16_4 + z15_3) + z13_5;
    let z13_7 = f * z13_6;
    let z15_4 = (-c[8]) * z8_1;
    let z16_5 = (-c[7]) * z7_1;
    let z15_5 = z15_4 + z16_5;
    let z15_6 = f * z15_5;
    let z16_6 = z3_1 * f;
    let z21_2 = 3.0 * c[13];
    let z22_1 = (-z10_1) * z21_2;
    let z22_2 = z22_1 - c[2];
    let z22_3 = z22_2 * z16_6;
    let z15_7 = z22_3 + z15_6;
    let z15_8 = z3_1 * z15_7;
    let z13_8 = z15_8 + z13_7;
    let z15_9 = (-param) * z18_5;
    let z16_7 = ((-z19_2) * z16_6) * z10_1;
    let z13_9 = (z15_9 + (0.5 * z13_8)) + z16_7;
    let z11_5 = z3_1 * z11_4;
    let z11_6 = (0.5 * z14_4) + z11_5;
    let z11_7 = param * z11_6;
    let z14_5 = 0.5 * z19_2;
    let z15_10 = z10_1 * z10_1;
    let z14_6 = z15_10 * z14_5;
    let z16_8 = c[8] * z10_1;
    let z16_9 = z16_8 + c[5];
    let z16_10 = z8_1 * z16_9;
    let z18_6 = c[7] * z10_1;
    let z18_7 = z18_6 + c[3];
    let z18_8 = z7_1 * z18_7;
    let z16_11 = z18_8 + z16_10;
    let z15_11 = z15_10 * z21_2;
    let z15_12 = c[1] + z15_11;
    let z18_9 = c[2] * z10_1;
    let z15_13 = (0.5 * z15_12) + z18_9;
    let z15_14 = z3_1 * z15_13;
    let z11_8 = ((z11_7 + (0.5 * z16_11)) + z15_14) + z14_6;
    let z11_9 = z2_1 * z11_8;
    let z11_10 = (0.5 * z13_9) + z11_9;
    let z11_11 = z11_10 * z20_3;
    let z13_10 = (z1_1 * z17_7) * (f * f);
    let z11_12 = (0.125 * z13_10) + z11_11;
    divnum[3] -= z12_10;
    divnum[2] -= z11_12;
}
