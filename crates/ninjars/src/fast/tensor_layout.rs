//! Monomial ordering of Ninja's tensor numerator interface.
//!
//! Original Ninja (`tensor_ninja.cc`, following arXiv:1405.0301 Eq. (C.15))
//! and MadLoop store a polynomial numerator of rank `R` in the loop-momentum
//! components `q = (q0, q1, q2, q3)` as a contiguous array of
//! `binomial(R + 4, 4)` coefficients: the homogeneous rank-0 block, then
//! rank 1, ..., rank `R`. Within the homogeneous rank-`r` block, monomials are
//! produced by `ten_build` from the rank-`(r-1)` block:
//! `[q0 * M[0]] ++ [q1 * M[i], i < r] ++ [q2 * M[i], i < r(r+1)/2] ++
//! [q3 * M[i], i < r(r+1)(r+2)/6]`. Every monomial appears exactly once, so
//! `N(q) = sum_k tensor[k] * monomial_k(q)` with no symmetry factors.

/// `binomial(n + k - 1, k) = n (n + 1) ... (n + k - 1) / k!`, exactly, or
/// `None` beyond `usize` (each partial product of `j` consecutive integers
/// divided by `j!` is an integer, computed in `u128`).
const fn rising(n: usize, k: usize) -> Option<usize> {
    let mut value: u128 = 1;
    let mut j = 0;
    while j < k {
        value = match value.checked_mul(n as u128 + j as u128) {
            Some(product) => product / (j as u128 + 1),
            None => return None,
        };
        j += 1;
    }
    if value > usize::MAX as u128 {
        None
    } else {
        Some(value as usize)
    }
}

/// `value` as `usize`, or `None` beyond it (32-bit targets).
const fn narrow(value: u64) -> Option<usize> {
    if value > usize::MAX as u64 {
        None
    } else {
        Some(value as usize)
    }
}

/// [`ncoeff`], or `None` where the count exceeds `usize`.
pub const fn checked_ncoeff(rank: usize) -> Option<usize> {
    // Below 2^16 - 4 the product of the four factors fits `u64`: the
    // unchecked formula, as fast as before.
    if rank < (1 << 16) - 4 {
        let r = rank as u64;
        return narrow((r + 1) * (r + 2) * (r + 3) * (r + 4) / 24);
    }
    match rank.checked_add(1) {
        Some(n) => rising(n, 4),
        None => None,
    }
}

/// [`ncoeff_homogeneous`], or `None` where the count exceeds `usize`.
pub const fn checked_ncoeff_homogeneous(rank: usize) -> Option<usize> {
    // Below 2^21 - 3 the product of the three factors fits `u64`.
    if rank < (1 << 21) - 3 {
        let r = rank as u64;
        return narrow((r + 1) * (r + 2) * (r + 3) / 6);
    }
    match rank.checked_add(1) {
        Some(n) => rising(n, 3),
        None => None,
    }
}

/// [`block_offset`], or `None` where the offset exceeds `usize`.
pub const fn checked_block_offset(rank: usize) -> Option<usize> {
    if rank == 0 {
        Some(0)
    } else {
        checked_ncoeff(rank - 1)
    }
}

/// Number of coefficients of a (non-homogeneous) rank-`rank` tensor,
/// `binomial(rank + 4, 4)`. Panics where it exceeds `usize` (from a rank of
/// about 1.45 * 10^5 on 64-bit targets; [`checked_ncoeff`] returns `None`
/// there).
pub const fn ncoeff(rank: usize) -> usize {
    match checked_ncoeff(rank) {
        Some(n) => n,
        None => panic!("tensor_layout::ncoeff: the coefficient count exceeds usize"),
    }
}

/// Number of coefficients of a homogeneous rank-`rank` block,
/// `binomial(rank + 3, 3)`. Panics where it exceeds `usize` (from a rank of
/// about 4.8 * 10^6 on 64-bit targets; [`checked_ncoeff_homogeneous`] returns
/// `None` there).
pub const fn ncoeff_homogeneous(rank: usize) -> usize {
    match checked_ncoeff_homogeneous(rank) {
        Some(n) => n,
        None => panic!("tensor_layout::ncoeff_homogeneous: the coefficient count exceeds usize"),
    }
}

/// Offset of the homogeneous rank-`rank` block inside a full tensor. Panics
/// where it exceeds `usize` ([`checked_block_offset`] returns `None` there).
pub const fn block_offset(rank: usize) -> usize {
    match checked_block_offset(rank) {
        Some(n) => n,
        None => panic!("tensor_layout::block_offset: the offset exceeds usize"),
    }
}

/// Coefficients of a rank-`rank` numerator with explicit `mu²` powers
/// `0..=mu_powers` (see [`super::MuTensorNumerator`]): block `k` is a dense
/// rank-`(rank - 2k)` tensor multiplying `mu^(2k)`, so the total is
/// `sum_k binomial(rank - 2k + 4, 4)`. Requires `2 * mu_powers <= rank`.
pub const fn ncoeff_mu(rank: usize, mu_powers: usize) -> usize {
    mu_block_offset(rank, mu_powers + 1)
}

/// Offset of the `mu^(2k)` block of a [`ncoeff_mu`] layout. Panics where it
/// exceeds `usize`.
pub const fn mu_block_offset(rank: usize, k: usize) -> usize {
    let mut total: usize = 0;
    let mut j = 0;
    while j < k {
        total = match total.checked_add(ncoeff(rank - 2 * j)) {
            Some(total) => total,
            None => panic!("tensor_layout::mu_block_offset: the offset exceeds usize"),
        };
        j += 1;
    }
    total
}

/// Index of `q0^e0 q1^e1 q2^e2 q3^e3` inside its homogeneous block, or
/// `None` where it exceeds `usize` (for a crate indexing exponents it read
/// from a file before validating them: no panic, no wrapped index).
///
/// `ten_build` generates a monomial of rank `r` from the one with its
/// highest-index variable lowered by one: `[q0 M0] ++ [q1 M(i<r1)] ++
/// [q2 M(i<r2)] ++ [q3 M(i<r3)]`, with r1 = r, r2 = r(r+1)/2 and
/// r3 = r(r+1)(r+2)/6, so each lowering adds the offset of that variable's
/// run in the rank-`r` block: `1 + r + r(r+1)/2` for `q3`, `1 + r` for `q2`
/// and `1` for `q1`. The sums of these offsets over each variable's
/// lowerings are triangular and tetrahedral numbers, summed in closed form:
/// the cost is constant in the exponents, with no recursion (one call per
/// unit of exponent overflowed the stack for exponents of order 2^20). The
/// sums are exact in `u128` (a rank below 2^34, a tetrahedral number below
/// 2^102).
pub fn checked_homogeneous_index(exponents: [u32; 4]) -> Option<usize> {
    let [e0, e1, e2, e3] = exponents.map(u128::from);
    // T(n) = 1 + ... + n and H(n) = T(1) + ... + T(n).
    let triangular = |n: u128| n * (n + 1) / 2;
    let tetrahedral = |n: u128| n * (n + 1) * (n + 2) / 6;
    let rank = e0 + e1 + e2 + e3;
    // The q3 lowerings, at ranks rank, rank - 1, ..., rank - e3 + 1.
    let below_q3 = rank - e3;
    let q3 = e3
        + (triangular(rank) - triangular(below_q3))
        + (tetrahedral(rank) - tetrahedral(below_q3));
    // The q2 lowerings, at ranks below_q3, ..., below_q3 - e2 + 1.
    let below_q2 = below_q3 - e2;
    let q2 = e2 + (triangular(below_q3) - triangular(below_q2));
    usize::try_from(q3 + q2 + e1).ok()
}

/// Index of `q0^e0 q1^e1 q2^e2 q3^e3` inside a full tensor array, or `None`
/// where it exceeds `usize`.
pub fn checked_monomial_index(exponents: [u32; 4]) -> Option<usize> {
    let rank = exponents.iter().map(|&e| u64::from(e)).sum::<u64>();
    checked_block_offset(usize::try_from(rank).ok()?)?
        .checked_add(checked_homogeneous_index(exponents)?)
}

/// Index of `q0^e0 q1^e1 q2^e2 q3^e3` inside its homogeneous block
/// ([`checked_homogeneous_index`]). Panics where it exceeds `usize` (from a
/// rank of about 4.8 * 10^6 on 64-bit targets).
pub fn homogeneous_index(exponents: [u32; 4]) -> usize {
    checked_homogeneous_index(exponents)
        .expect("tensor_layout::homogeneous_index: the index exceeds usize")
}

/// Index of `q0^e0 q1^e1 q2^e2 q3^e3` inside a full tensor array
/// ([`checked_monomial_index`]). Panics where it exceeds `usize` (from a
/// rank of about 1.45 * 10^5 on 64-bit targets).
pub fn monomial_index(exponents: [u32; 4]) -> usize {
    checked_monomial_index(exponents)
        .expect("tensor_layout::monomial_index: the index exceeds usize")
}

/// Exponent tuples of every coefficient of a rank-`rank` tensor, in storage
/// order.
pub fn monomials(rank: usize) -> Vec<[u32; 4]> {
    let mut all = vec![[0u32; 4]];
    let mut lower = vec![[0u32; 4]];
    for r in 1..=rank {
        let r1 = r;
        let r2 = r * (r + 1) / 2;
        let r3 = r * (r + 1) * (r + 2) / 6;
        let mut block = Vec::with_capacity(ncoeff_homogeneous(r));
        let bump = |m: [u32; 4], k: usize| {
            let mut m = m;
            m[k] += 1;
            m
        };
        block.push(bump(lower[0], 0));
        block.extend(lower[..r1].iter().map(|&m| bump(m, 1)));
        block.extend(lower[..r2].iter().map(|&m| bump(m, 2)));
        block.extend(lower[..r3].iter().map(|&m| bump(m, 3)));
        all.extend_from_slice(&block);
        lower = block;
    }
    all
}

#[cfg(test)]
mod tests {
    use super::*;

    /// `ten_build`'s recursion, one call per unit of exponent: the reference
    /// of the closed form.
    fn recursive_index(exponents: [u32; 4]) -> usize {
        let [e0, e1, e2, e3] = exponents;
        let r = (e0 + e1 + e2 + e3) as usize;
        if e3 > 0 {
            1 + r + r * (r + 1) / 2 + recursive_index([e0, e1, e2, e3 - 1])
        } else if e2 > 0 {
            1 + r + recursive_index([e0, e1, e2 - 1, 0])
        } else if e1 > 0 {
            1 + recursive_index([e0, e1 - 1, 0, 0])
        } else {
            0
        }
    }

    #[test]
    fn closed_form_index_equals_the_recursion() {
        for rank in 0..=16 {
            for m in monomials(rank) {
                assert_eq!(homogeneous_index(m), recursive_index(m), "{m:?}");
            }
        }
    }

    /// Exponents far beyond any loop numerator's: the index costs no stack
    /// (the recursion overflowed a thread's stack at 2^20) and is exact:
    /// `q3^n` ends the rank-`n` block, `q2^n` its `q2` run, and `q1^n` sits
    /// at `n`.
    #[test]
    fn large_exponents_index_without_recursion() {
        for n in [1usize << 10, 1 << 20, 1 << 21] {
            let e = n as u32;
            assert_eq!(homogeneous_index([0, 0, 0, e]), ncoeff_homogeneous(n) - 1);
            assert_eq!(homogeneous_index([0, 0, e, 0]), n + n * (n + 1) / 2);
            assert_eq!(homogeneous_index([0, e, 0, 0]), n);
            assert_eq!(homogeneous_index([e, 0, 0, 0]), 0);
        }
        let n = 1usize << 12;
        let e = n as u32;
        assert_eq!(monomial_index([0, 0, 0, e]), ncoeff(n) - 1);
    }

    /// `binomial(n, k)` in `u128` (the tests' reference, `n` below 2^32 + 5).
    fn binomial(n: u128, k: u128) -> u128 {
        (0..k).fold(1, |value, j| value * (n - j) / (j + 1))
    }

    /// The largest `n` with `f(n) <= usize::MAX`, for `f` increasing.
    fn edge(f: impl Fn(u128) -> u128) -> u128 {
        let (mut low, mut high) = (0u128, 1u128 << 24);
        while high - low > 1 {
            let mid = (low + high) / 2;
            if f(mid) <= usize::MAX as u128 {
                low = mid
            } else {
                high = mid
            }
        }
        low
    }

    /// At and beyond `usize`: the checked indices and counts are exact up
    /// to their edge and `None` past it (no wrapped value), for exponents
    /// up to `u32::MAX` in each variable; the unchecked ones panic there.
    #[test]
    fn indices_and_counts_are_exact_up_to_usize_and_none_beyond() {
        // q3^n is the last monomial of the rank-n tensor: ncoeff(n) - 1.
        let full = edge(|n| binomial(n + 4, 4) - 1);
        let block = edge(|n| binomial(n + 3, 3) - 1);
        let count = edge(|n| binomial(n + 4, 4));
        assert!((140_000..150_000).contains(&full), "{full}");
        assert!((4_700_000..4_900_000).contains(&block), "{block}");
        for n in [
            full - 1,
            full,
            full + 1,
            block,
            block + 1,
            1 << 31,
            u32::MAX as u128,
        ] {
            let e = n as u32;
            let want = |value: u128| usize::try_from(value).ok();
            assert_eq!(
                checked_monomial_index([0, 0, 0, e]),
                want(binomial(n + 4, 4) - 1),
                "{n}"
            );
            assert_eq!(
                checked_homogeneous_index([0, 0, 0, e]),
                want(binomial(n + 3, 3) - 1),
                "{n}"
            );
            assert_eq!(
                checked_homogeneous_index([0, 0, e, 0]),
                want(n + n * (n + 1) / 2),
                "{n}"
            );
            assert_eq!(checked_homogeneous_index([0, e, 0, 0]), want(n), "{n}");
            assert_eq!(checked_ncoeff(n as usize), want(binomial(n + 4, 4)), "{n}");
        }
        assert_eq!(
            checked_ncoeff(count as usize),
            Some(binomial(count + 4, 4) as usize)
        );
        assert_eq!(checked_ncoeff(count as usize + 1), None);
        assert_eq!(checked_ncoeff(usize::MAX), None);
        assert_eq!(
            checked_block_offset(count as usize + 1),
            Some(ncoeff(count as usize))
        );
        // The counts' fast paths end below the products' 64-bit edge; the
        // exact path takes over there and ends at the count's edge.
        let homogeneous = edge(|n| binomial(n + 3, 3));
        for n in [
            0u128,
            1,
            (1 << 16) - 5,
            (1 << 16) - 4,
            (1 << 21) - 4,
            (1 << 21) - 3,
            homogeneous,
            homogeneous + 1,
        ] {
            let want = |value: u128| usize::try_from(value).ok();
            assert_eq!(checked_ncoeff(n as usize), want(binomial(n + 4, 4)), "{n}");
            assert_eq!(
                checked_ncoeff_homogeneous(n as usize),
                want(binomial(n + 3, 3)),
                "{n}"
            );
        }
        assert_eq!(checked_ncoeff_homogeneous(usize::MAX), None);
        let e = u32::MAX;
        assert_eq!(checked_monomial_index([e, e, e, e]), None);
        assert_eq!(checked_homogeneous_index([e, e, e, e]), None);
        assert_eq!(checked_homogeneous_index([e, 0, 0, 0]), Some(0));
        let wrapped = std::panic::catch_unwind(|| monomial_index([0, 0, 0, full as u32 + 1]));
        assert!(wrapped.is_err());
        let wrapped = std::panic::catch_unwind(|| ncoeff(count as usize + 1));
        assert!(wrapped.is_err());
    }

    #[test]
    fn ordering_is_a_bijection_matching_ten_build() {
        for rank in 0..=8 {
            let monomials = monomials(rank);
            assert_eq!(monomials.len(), ncoeff(rank));
            for (index, &m) in monomials.iter().enumerate() {
                assert_eq!(monomial_index(m), index, "rank {rank} monomial {m:?}");
            }
            let mut sorted = monomials.clone();
            sorted.sort();
            sorted.dedup();
            assert_eq!(sorted.len(), monomials.len());
        }
        assert_eq!(
            monomials(1),
            vec![
                [0, 0, 0, 0],
                [1, 0, 0, 0],
                [0, 1, 0, 0],
                [0, 0, 1, 0],
                [0, 0, 0, 1]
            ]
        );
    }
}
