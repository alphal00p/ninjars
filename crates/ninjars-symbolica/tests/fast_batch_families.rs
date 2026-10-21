//! Batches of lanes from unrelated loop families (fast engine, bitwise).
//!
//! A caller may batch lanes of unrelated loop families that share `(n,
//! rank)` into one [`FastAmplitude`] call (pyAmpliCol does). Every lane's
//! result must then be bitwise its result alone. The other batching tests
//! mix only related lanes: V7c (`rank_n2_invariance.rs`) batches the lanes of
//! one case file; `fast/tests_x2.rs` (`mixed_batches_split_and_stay_bitwise`)
//! batches random lanes of mixed rank at declared rank n+2 with hash masters,
//! at width 7; `rank_n2_padding.rs` (`fast_mixed_rank_batches_*`) batches the
//! lanes of one generator (`rn2-int-complex`) at n = 3 and 4.
//!
//! Here the lanes of every bench case with one `n` (`validation/rank-n2`:
//! suite, controls and identity cases; `validation/benchmark`: the physical
//! top-loop cases, the synthetic and the synthetic rank n+2 cases) form a
//! pool, so a batch mixes kinematics (integer, physical, shifted, IR-soft,
//! exact dyadic), masses (zero, real, complex), numerators, `mu²` powers and
//! scales `mu_R²` of unrelated families:
//! - rank <= n+1 (the default engine, original Ninja's algorithm): for every
//!   `(n, rank)` with `rank <= n + 1` and at least two cases, the lanes of
//!   the cases of exactly that `(n, rank)`;
//! - declared rank n+2: for every `n = 1..7`, the lanes of every case of that
//!   `n` (ranks 0 to n+2), declared at rank n+2: by default, where each lane
//!   is reduced at its own rank (`fast/declared_rank.rs`), and through
//!   [`AtDeclaredRank`] (the rank n+2 engine for every lane).
//!
//! Each pool runs as plain `TensorNumerator` lanes (the cases without `mu²`
//! powers) and as `MuTensorNumerator` lanes (every case, laid out with the
//! pool's largest number of `mu²` powers; the lanes with fewer get zero
//! blocks), each with the S-matrix of every lane (lanes without one are left
//! out) and without any S-matrix. For n >= 2 every pool also holds a failing
//! lane, second in its order: its first lane with the second propagator
//! replaced by the first (every cut through both is singular, as in
//! `rank_n2_padding.rs`), which must not succeed.
//!
//! The check: every lane alone (width 1) on a fresh amplitude is the
//! reference. The pool's lanes in interleaved order (lane `l` of every case
//! before lane `l + 1`, so that neighbours come from different families) and
//! in reverse, cut into consecutive batches of widths 1, 3, 8 and 32 (the
//! last batch wraps around, so a pool shorter than 32 lanes repeats lanes in
//! one batch), run through one amplitude per pool; at every position the
//! status and the five values (Laurent coefficients, cut-constructible and
//! rational parts) must equal the lane's reference bitwise (results are
//! preset to the default, so a failing lane's values are compared too).
//! Pools with fewer than two cases are skipped. At declared rank
//! n+2 by default, the reference of a lane of rank `r < n + 2` must also
//! equal the lane declared at rank `r` (with `min(mu² powers, r/2)` blocks)
//! bitwise. Masters: `OneLoopNativeMasters`.

mod rank_n2_reference_support;

use std::collections::BTreeSet;

use ninjars::fast::{
    AtDeclaredRank, C64, FastAmplitude, FastInput, FastNumerator, FastOptions, FastResult,
    FastStatus, MuTensorNumerator, RMom, TensorNumerator,
    tensor_layout::{mu_block_offset, ncoeff, ncoeff_mu},
};
use ninjars_symbolica::OneLoopNativeMasters;
use rank_n2_reference_support::{
    cases_in,
    harness::{BenchCase, BenchLane, fast_values},
    rank_n2, root,
};

const WIDTHS: [usize; 4] = [1, 3, 8, 32];

/// Every bench case of the validation directories.
fn all_cases() -> Vec<BenchCase> {
    let dirs = [
        rank_n2().join("cases"),
        rank_n2().join("controls/cases"),
        rank_n2().join("identity/cases"),
        root().join("validation/benchmark/cases"),
        root().join("validation/benchmark/synthetic"),
        root().join("validation/benchmark/synthetic-rank-n2"),
    ];
    let cases: Vec<BenchCase> = dirs
        .iter()
        .flat_map(|dir| cases_in(dir, |_| true))
        .map(|file| BenchCase::load(&file).unwrap())
        .collect();
    assert!(cases.len() >= 400, "{} bench cases", cases.len());
    cases
}

/// The bits of the five values of a lane.
fn bits(result: &FastResult) -> [(u64, u64); 5] {
    fast_values(result).map(|z| (z.re.to_bits(), z.im.to_bits()))
}

/// Lane `lane` of `case` laid out at rank `rank >= case.rank` with
/// `mu_powers >= case.mu_powers` powers of `mu²` (zero new coefficients and
/// blocks).
fn relaid(case: &BenchCase, lane: usize, rank: usize, mu_powers: usize) -> BenchLane {
    assert!(rank >= case.rank && mu_powers >= case.mu_powers && 2 * mu_powers <= rank);
    let source = &case.lanes[lane];
    let mut tensor = vec![C64::ZERO; ncoeff_mu(rank, mu_powers)];
    for k in 0..=case.mu_powers {
        let (from, to, size) = (
            mu_block_offset(case.rank, k),
            mu_block_offset(rank, k),
            ncoeff(case.rank - 2 * k),
        );
        tensor[to..to + size].copy_from_slice(&source.tensor[from..from + size]);
    }
    BenchLane {
        tensor,
        ..source.clone()
    }
}

/// `lane` with its second propagator replaced by its first (S-matrix
/// included), as in `rank_n2_padding.rs`.
fn with_duplicated_propagator(lane: &BenchLane, n: usize) -> BenchLane {
    let mut out = lane.clone();
    out.id = format!("{}-duplicated", lane.id);
    out.shifts[1] = out.shifts[0];
    out.masses[1] = out.masses[0];
    if let Some(s) = &mut out.s_matrix {
        let old = s.clone();
        let line = |i: usize| if i == 1 { 0 } else { i };
        for i in 0..n {
            for j in 0..n {
                s[i * n + j] = old[line(i) * n + line(j)];
            }
        }
    }
    out
}

/// The mass pattern of a lane: 0 all masses zero, 1 real, 2 some complex.
fn mass_class(lane: &BenchLane) -> u8 {
    if lane.masses.iter().any(|m| m.im != 0.) {
        2
    } else if lane.masses.iter().any(|m| m.re != 0.) {
        1
    } else {
        0
    }
}

/// Lanes of unrelated families laid out at one `(n, rank, mu_powers)`.
struct Pool {
    name: String,
    n: usize,
    rank: usize,
    mu_powers: usize,
    /// `(family, lane)`, in interleaved order.
    lanes: Vec<(String, BenchLane)>,
}

impl Pool {
    /// The lanes of `cases` in interleaved order, with their S-matrices (lanes
    /// without one are left out) or without any.
    fn new(
        name: &str,
        cases: &[&BenchCase],
        n: usize,
        rank: usize,
        mu_powers: usize,
        s_matrix: bool,
    ) -> Self {
        let longest = cases.iter().map(|c| c.lanes.len()).max().unwrap_or(0);
        let mut lanes = Vec::new();
        for index in 0..longest {
            for case in cases {
                assert_eq!(case.n, n, "{}", case.name);
                if index >= case.lanes.len() {
                    continue;
                }
                let mut lane = relaid(case, index, rank, mu_powers);
                if !s_matrix {
                    lane.s_matrix = None;
                } else if lane.s_matrix.is_none() {
                    continue;
                }
                lanes.push((case.name.clone(), lane));
            }
        }
        // A failing lane among them: the first lane with its second
        // propagator replaced by its first (every cut through both is
        // singular), second in the order.
        if n >= 2 && !lanes.is_empty() {
            let (family, lane) = &lanes[0];
            let failing = (
                format!("{family}-duplicated"),
                with_duplicated_propagator(lane, n),
            );
            lanes.insert(1, failing);
        }
        Self {
            name: format!(
                "{name} n={n} rank={rank} mu^{mu_powers} S-matrix={s_matrix}: {} lanes of {} cases",
                lanes.len(),
                cases.len()
            ),
            n,
            rank,
            mu_powers,
            lanes,
        }
    }

    /// The number of cases among `order` (the failing lane counts as its
    /// source case).
    fn families(&self, order: &[usize]) -> usize {
        order
            .iter()
            .map(|&i| {
                let family = &self.lanes[i].0;
                family.strip_suffix("-duplicated").unwrap_or(family)
            })
            .collect::<BTreeSet<_>>()
            .len()
    }

    fn mass_classes(&self, order: &[usize]) -> usize {
        order
            .iter()
            .map(|&i| mass_class(&self.lanes[i].1))
            .collect::<BTreeSet<_>>()
            .len()
    }
}

/// Lanes at `(rank, mu_powers)` as one batch: `TensorNumerator` without
/// `mu²` powers, `MuTensorNumerator` otherwise, optionally wrapped in
/// [`AtDeclaredRank`]. Results preset to the default.
fn run(
    amplitude: &mut FastAmplitude,
    lanes: &[&BenchLane],
    rank: usize,
    mu_powers: usize,
    at_declared_rank: bool,
    masters: &mut OneLoopNativeMasters,
) -> (Vec<FastResult>, Vec<FastStatus>) {
    let width = lanes.len();
    let shifts: Vec<RMom> = lanes
        .iter()
        .flat_map(|l| l.shifts.iter().copied())
        .collect();
    let masses: Vec<C64> = lanes
        .iter()
        .flat_map(|l| l.masses.iter().copied())
        .collect();
    let scales: Vec<f64> = lanes.iter().map(|l| l.mu2).collect();
    let s_matrix: Option<Vec<f64>> = lanes
        .iter()
        .map(|l| l.s_matrix.clone())
        .collect::<Option<Vec<_>>>()
        .map(|rows| rows.concat());
    assert!(
        s_matrix.is_some() || lanes.iter().all(|l| l.s_matrix.is_none()),
        "a batch has S-matrices for every lane or for none"
    );
    let input = FastInput {
        lanes: width,
        shifts: &shifts,
        masses_squared: &masses,
        scales_squared: &scales,
        s_matrix: s_matrix.as_deref(),
    };
    let coefficients: Vec<C64> = lanes
        .iter()
        .flat_map(|l| l.tensor.iter().copied())
        .collect();
    let mut results = vec![FastResult::default(); width];
    let mut statuses = vec![FastStatus::NotEvaluated; width];
    fn evaluate<N: FastNumerator>(
        amplitude: &mut FastAmplitude,
        input: &FastInput,
        numerator: &mut N,
        at_declared_rank: bool,
        masters: &mut OneLoopNativeMasters,
        results: &mut [FastResult],
        statuses: &mut [FastStatus],
    ) {
        if at_declared_rank {
            let mut numerator = AtDeclaredRank(numerator);
            amplitude.evaluate(input, &mut numerator, masters, results, statuses)
        } else {
            amplitude.evaluate(input, numerator, masters, results, statuses)
        }
        .unwrap();
    }
    if mu_powers == 0 {
        let mut tensor = TensorNumerator::new(rank, width);
        tensor.set_coefficients(width, &coefficients).unwrap();
        evaluate(
            amplitude,
            &input,
            &mut tensor,
            at_declared_rank,
            masters,
            &mut results,
            &mut statuses,
        );
    } else {
        let mut tensor = MuTensorNumerator::new(rank, mu_powers, width).unwrap();
        tensor.set_coefficients(width, &coefficients).unwrap();
        evaluate(
            amplitude,
            &input,
            &mut tensor,
            at_declared_rank,
            masters,
            &mut results,
            &mut statuses,
        );
    }
    (results, statuses)
}

/// `lane_rank` of a lane laid out at `(rank, mu_powers)`.
fn lane_rank(lane: &BenchLane, rank: usize, mu_powers: usize) -> usize {
    let mut numerator: Box<dyn FastNumerator> = if mu_powers == 0 {
        let mut tensor = TensorNumerator::new(rank, 1);
        tensor.set_coefficients(1, &lane.tensor).unwrap();
        Box::new(tensor)
    } else {
        let mut tensor = MuTensorNumerator::new(rank, mu_powers, 1).unwrap();
        tensor.set_coefficients(1, &lane.tensor).unwrap();
        Box::new(tensor)
    };
    numerator.lane_rank(0).unwrap()
}

/// A lane laid out at `(rank, mu_powers)` re-laid at its own rank `own`
/// with `min(mu_powers, own / 2)` powers of `mu²` (the coefficients above
/// `own` are zero).
fn at_own_rank(lane: &BenchLane, rank: usize, mu_powers: usize, own: usize) -> (BenchLane, usize) {
    let own_mu = mu_powers.min(own / 2);
    let mut tensor = vec![C64::ZERO; ncoeff_mu(own, own_mu)];
    for k in 0..=own_mu {
        let (from, to, size) = (
            mu_block_offset(rank, k),
            mu_block_offset(own, k),
            ncoeff(own - 2 * k),
        );
        tensor[to..to + size].copy_from_slice(&lane.tensor[from..from + size]);
    }
    (
        BenchLane {
            tensor,
            ..lane.clone()
        },
        own_mu,
    )
}

/// Consecutive batches of `width` positions of `order`; the last wraps
/// around.
fn batches(order: &[usize], width: usize) -> Vec<Vec<usize>> {
    (0..order.len().div_ceil(width))
        .map(|b| {
            (0..width)
                .map(|i| order[(b * width + i) % order.len()])
                .collect()
        })
        .collect()
}

#[derive(Default)]
struct Counts {
    pools: usize,
    lanes: usize,
    failed_lanes: usize,
    lowered_lanes: usize,
    batches: usize,
    checked: usize,
    /// Batches of width >= 3 with lanes of at least two families and two
    /// mass patterns.
    mixed_batches: usize,
}

/// The check of the module documentation on one pool.
fn check_pool(pool: &Pool, at_declared_rank: bool, counts: &mut Counts) {
    let (n, rank, mu_powers) = (pool.n, pool.rank, pool.mu_powers);
    let lowered = rank == n + 2 && !at_declared_rank;
    let mode = if at_declared_rank {
        " (AtDeclaredRank)"
    } else {
        ""
    };
    let count = pool.lanes.len();
    assert!(
        pool.families(&(0..count).collect::<Vec<_>>()) >= 2,
        "{}: fewer than two families",
        pool.name
    );
    let mut reference_masters = OneLoopNativeMasters::new().unwrap();
    let options = FastOptions::default();
    let reference: Vec<(FastResult, FastStatus)> = pool
        .lanes
        .iter()
        .map(|(family, lane)| {
            let mut amplitude = FastAmplitude::new(n, rank, options).unwrap();
            let (results, statuses) = run(
                &mut amplitude,
                &[lane],
                rank,
                mu_powers,
                at_declared_rank,
                &mut reference_masters,
            );
            if lowered {
                let own = lane_rank(lane, rank, mu_powers);
                if own < rank {
                    let (low, own_mu) = at_own_rank(lane, rank, mu_powers, own);
                    let mut at_own = FastAmplitude::new(n, own, options).unwrap();
                    let (want, want_statuses) = run(
                        &mut at_own,
                        &[&low],
                        own,
                        own_mu,
                        false,
                        &mut reference_masters,
                    );
                    let label = format!(
                        "{}: {family} lane {} (rank {own}) alone at declared rank {rank}",
                        pool.name, lane.id
                    );
                    assert_eq!(statuses[0], want_statuses[0], "{label}: own-rank status");
                    assert_eq!(bits(&results[0]), bits(&want[0]), "{label}: own rank");
                    counts.lowered_lanes += 1;
                }
            }
            (results[0], statuses[0])
        })
        .collect();
    for ((family, lane), (_, status)) in pool.lanes.iter().zip(&reference) {
        if family.ends_with("-duplicated") {
            assert_ne!(
                *status,
                FastStatus::Success,
                "{}{mode}: {family} lane {} with a duplicated propagator",
                pool.name,
                lane.id
            );
        }
    }
    counts.failed_lanes += reference
        .iter()
        .filter(|(_, status)| *status != FastStatus::Success)
        .count();
    let mut masters = OneLoopNativeMasters::new().unwrap();
    let mut amplitude = FastAmplitude::new(n, rank, options).unwrap();
    let interleaved: Vec<usize> = (0..count).collect();
    let reversed: Vec<usize> = (0..count).rev().collect();
    for (order_name, order) in [("interleaved", &interleaved), ("reversed", &reversed)] {
        for width in WIDTHS {
            for (index, batch) in batches(order, width).iter().enumerate() {
                let lanes: Vec<&BenchLane> = batch.iter().map(|&i| &pool.lanes[i].1).collect();
                let (results, statuses) = run(
                    &mut amplitude,
                    &lanes,
                    rank,
                    mu_powers,
                    at_declared_rank,
                    &mut masters,
                );
                for (position, &lane) in batch.iter().enumerate() {
                    let (want, want_status) = reference[lane];
                    let (family, data) = &pool.lanes[lane];
                    let label = format!(
                        "{}{mode}: {family} lane {} at position {position} of {order_name} batch {index} of width {width}",
                        pool.name, data.id
                    );
                    assert_eq!(statuses[position], want_status, "{label}: status");
                    assert_eq!(bits(&results[position]), bits(&want), "{label}");
                    counts.checked += 1;
                }
                if width >= 3 && pool.families(batch) >= 2 && pool.mass_classes(batch) >= 2 {
                    counts.mixed_batches += 1;
                }
                counts.batches += 1;
            }
        }
    }
    counts.pools += 1;
    counts.lanes += count;
    eprintln!("{}{mode}: bitwise", pool.name);
}

/// The plain and `mu²` pools of `cases` at `(n, rank)`, with and without
/// S-matrices.
fn pools(name: &str, cases: &[&BenchCase], n: usize, rank: usize) -> Vec<Pool> {
    let plain: Vec<&BenchCase> = cases.iter().copied().filter(|c| c.mu_powers == 0).collect();
    let mu_powers = cases.iter().map(|c| c.mu_powers).max().unwrap_or(0);
    let mut out = Vec::new();
    for s_matrix in [true, false] {
        if plain.len() >= 2 {
            out.push(Pool::new(name, &plain, n, rank, 0, s_matrix));
        }
        if mu_powers > 0 {
            out.push(Pool::new(name, cases, n, rank, mu_powers, s_matrix));
        }
    }
    out.retain(|pool| pool.families(&(0..pool.lanes.len()).collect::<Vec<_>>()) >= 2);
    out
}

fn report(label: &str, counts: &Counts) {
    eprintln!(
        "FAST_BATCH_FAMILIES {label}: {} pools, {} lanes ({} failed, {} lowered and checked against their own rank), {} batches, {} lane positions bitwise, {} batches of width >= 3 mixing families and mass patterns",
        counts.pools,
        counts.lanes,
        counts.failed_lanes,
        counts.lowered_lanes,
        counts.batches,
        counts.checked,
        counts.mixed_batches
    );
}

/// Rank <= n+1: lanes of unrelated cases of one `(n, rank)` batched equal
/// their single-lane results bitwise.
#[test]
fn unrelated_families_batch_bitwise_below_rank_n2() {
    let cases = all_cases();
    let mut counts = Counts::default();
    let mut groups = BTreeSet::new();
    for n in 1..=7 {
        for rank in 0..=n + 1 {
            let group: Vec<&BenchCase> = cases
                .iter()
                .filter(|c| c.n == n && c.rank == rank)
                .collect();
            if group.len() < 2 {
                continue;
            }
            for pool in pools("rank <= n+1", &group, n, rank) {
                check_pool(&pool, false, &mut counts);
                groups.insert((n, rank));
            }
        }
    }
    report("rank <= n+1", &counts);
    // Every n = 1..7 at rank n+1, and ranks below.
    assert!((1..=7).all(|n| groups.contains(&(n, n + 1))), "{groups:?}");
    assert!(groups.len() >= 20, "{groups:?}");
    assert!(counts.mixed_batches > 0);
}

/// Declared rank n+2: lanes of unrelated cases of one `n` and every rank up
/// to n+2, each reduced at its own rank by default, batched equal their
/// single-lane results bitwise; and so through [`AtDeclaredRank`].
#[test]
fn unrelated_families_batch_bitwise_at_declared_rank_n2() {
    let cases = all_cases();
    let mut counts = Counts::default();
    let mut at_declared = Counts::default();
    for n in 1..=7 {
        let group: Vec<&BenchCase> = cases
            .iter()
            .filter(|c| c.n == n && c.rank <= n + 2)
            .collect();
        let pools = pools("declared rank n+2", &group, n, n + 2);
        assert!(pools.len() >= 2, "n={n}: {} pools", pools.len());
        for pool in &pools {
            check_pool(pool, false, &mut counts);
            check_pool(pool, true, &mut at_declared);
        }
    }
    report("declared rank n+2", &counts);
    report("declared rank n+2, AtDeclaredRank", &at_declared);
    assert!(counts.lowered_lanes > 0 && counts.mixed_batches > 0);
}
