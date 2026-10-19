use crate::{Error, Result};
use std::collections::HashMap;

/// Numerator rank relative to the number of propagators `n`.
///
/// Rank counts the loop momentum with weight one and `mu²` with weight two.
/// The class fixes every residue layout: the excess `rank - n` (0 for any
/// rank `<= n`) raises the degree of each `k`-point residue to
/// `k + excess` (see [`residue_coefficient_count`]). The generic engine
/// accepts every class; the fast engine (`ninjars::fast`) accepts `UpToN` and
/// `NPlus1` (`tools/rank_n2/README.md`).
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub enum RankClass {
    /// `rank <= n` (excess 0).
    UpToN,
    /// `rank = n + 1` (excess 1, original Ninja's `Xninja`).
    NPlus1,
    /// `rank = n + 2` (excess 2).
    NPlus2,
}

impl RankClass {
    /// The class of a rank-`rank` numerator over `n` propagators, or `None`
    /// above `n + 2`.
    pub const fn of(n: usize, rank: usize) -> Option<Self> {
        if rank <= n {
            Some(Self::UpToN)
        } else if rank - n == 1 {
            Some(Self::NPlus1)
        } else if rank - n == 2 {
            Some(Self::NPlus2)
        } else {
            None
        }
    }

    /// `max(rank - n, 0)`.
    pub const fn excess(self) -> u8 {
        match self {
            Self::UpToN => 0,
            Self::NPlus1 => 1,
            Self::NPlus2 => 2,
        }
    }
}

/// Number of residue coefficients of a cut of `cut_size` lines (1..=5) in
/// original Ninja's index order, unused entries included.
///
/// | cut size | `UpToN` | `NPlus1` | `NPlus2` |
/// |---|---|---|---|
/// | 5 (pentagon) | 1 | 1 | 1 |
/// | 4 (box) | 5 | 6 | 7 |
/// | 3 (triangle) | 10 | 15 | 21 |
/// | 2 (bubble) | 10 | 20 | 35 |
/// | 1 (tadpole, integrated layout) | 5 | 16 | 16 |
///
/// The tadpole keeps Ninja's 16-entry integrated layout at `NPlus2` (only
/// c0, c14 and c15 are computed; the new degree-3 terms are odd and
/// integrate to zero). Panics for other cut sizes.
pub const fn residue_coefficient_count(cut_size: usize, class: RankClass) -> usize {
    match (cut_size, class) {
        (5, _) => 1,
        (4, RankClass::UpToN) => 5,
        (4, RankClass::NPlus1) => 6,
        (4, RankClass::NPlus2) => 7,
        (3, RankClass::UpToN) => 10,
        (3, RankClass::NPlus1) => 15,
        (3, RankClass::NPlus2) => 21,
        (2, RankClass::UpToN) => 10,
        (2, RankClass::NPlus1) => 20,
        (2, RankClass::NPlus2) => 35,
        (1, RankClass::UpToN) => 5,
        (1, RankClass::NPlus1 | RankClass::NPlus2) => 16,
        _ => panic!("residue cuts have one to five lines"),
    }
}

/// A cut and its uncut propagators, in original propagator order.
#[derive(Clone, Debug)]
pub struct Cut {
    pub partition: Vec<usize>,
    pub complement: Vec<usize>,
    /// Indices of strict higher cuts containing this cut.
    pub parents: Vec<usize>,
}

/// Reusable combinatorial reduction data, independent of numerical kinematics.
#[derive(Clone, Debug)]
pub struct Topology {
    propagators: usize,
    rank: usize,
    cuts: Vec<Cut>,
}

impl Topology {
    pub fn new(propagators: usize, rank: usize) -> Result<Self> {
        if propagators == 0 {
            return Err(Error::NoPropagators);
        }
        if rank > propagators.saturating_add(2) {
            return Err(Error::Rank { rank, propagators });
        }
        let mut cuts = Vec::new();
        for size in (1..=propagators.min(5)).rev() {
            let mut partition: Vec<_> = (0..size).collect();
            loop {
                cuts.push(Cut {
                    complement: (0..propagators)
                        .filter(|i| !partition.contains(i))
                        .collect(),
                    partition: partition.clone(),
                    parents: Vec::new(),
                });
                let Some(i) = (0..size)
                    .rev()
                    .find(|&i| partition[i] < propagators - size + i)
                else {
                    break;
                };
                partition[i] += 1;
                for j in i + 1..size {
                    partition[j] = partition[j - 1] + 1;
                }
            }
        }
        // Enumerate each cut's strict supersets of at most five propagators
        // instead of testing every earlier cut: the scan was quadratic in the
        // number of cuts (seconds for every stage of twenty propagators).
        let index: HashMap<Vec<usize>, usize> = cuts
            .iter()
            .enumerate()
            .map(|(i, cut)| (cut.partition.clone(), i))
            .collect();
        for cut in &mut cuts {
            let mut parents = Vec::new();
            let mut extra = Vec::new();
            supersets(
                &cut.partition,
                &cut.complement,
                0,
                propagators.min(5) - cut.partition.len(),
                &mut extra,
                &mut |superset| parents.push(index[superset]),
            );
            parents.sort_unstable();
            cut.parents = parents;
        }
        Ok(Self {
            propagators,
            rank,
            cuts,
        })
    }

    pub fn propagators(&self) -> usize {
        self.propagators
    }
    pub fn rank(&self) -> usize {
        self.rank
    }
    /// The numerator rank class; `new` rejects ranks above every class.
    pub fn rank_class(&self) -> RankClass {
        RankClass::of(self.propagators, self.rank)
            .expect("Topology::new validates the rank against the supported classes")
    }
    pub fn cuts(&self) -> &[Cut] {
        &self.cuts
    }
    pub fn cuts_of_size(&self, size: usize) -> impl Iterator<Item = &Cut> {
        self.cuts
            .iter()
            .filter(move |cut| cut.partition.len() == size)
    }
}

/// Calls `found` with every sorted `partition ∪ extra` for nonempty `extra`
/// drawn in order from `complement[start..]`, at most `remaining` more lines.
fn supersets(
    partition: &[usize],
    complement: &[usize],
    start: usize,
    remaining: usize,
    extra: &mut Vec<usize>,
    found: &mut impl FnMut(&[usize]),
) {
    if remaining == 0 {
        return;
    }
    for i in start..complement.len() {
        extra.push(complement[i]);
        let mut union: Vec<usize> = partition.iter().chain(extra.iter()).copied().collect();
        union.sort_unstable();
        found(&union);
        supersets(partition, complement, i + 1, remaining - 1, extra, found);
        extra.pop();
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn six_propagator_partition_and_parent_invariants() {
        let topology = Topology::new(6, 7).unwrap();
        assert_eq!(topology.cuts.len(), 62);
        assert_eq!(topology.cuts_of_size(5).count(), 6);
        assert_eq!(topology.cuts_of_size(2).count(), 15);
        for cut in &topology.cuts {
            let mut all = cut.partition.clone();
            all.extend(&cut.complement);
            all.sort();
            assert_eq!(all, (0..6).collect::<Vec<_>>());
            assert!(cut.parents.iter().all(|&p| {
                cut.partition
                    .iter()
                    .all(|i| topology.cuts[p].partition.contains(i))
            }));
        }
        assert!(Topology::new(6, 8).is_ok());
        assert!(Topology::new(6, 9).is_err());
    }

    #[test]
    fn rank_classes_and_residue_counts() {
        for n in 1..=8 {
            for rank in 0..=n {
                assert_eq!(RankClass::of(n, rank), Some(RankClass::UpToN));
            }
            assert_eq!(RankClass::of(n, n + 1), Some(RankClass::NPlus1));
            assert_eq!(RankClass::of(n, n + 2), Some(RankClass::NPlus2));
            assert_eq!(RankClass::of(n, n + 3), None);
            for rank in 0..=n + 2 {
                let topology = Topology::new(n, rank).unwrap();
                assert_eq!(topology.rank_class(), RankClass::of(n, rank).unwrap());
            }
            assert!(matches!(Topology::new(n, n + 3), Err(Error::Rank { .. })));
        }
        assert_eq!(
            [RankClass::UpToN, RankClass::NPlus1, RankClass::NPlus2].map(RankClass::excess),
            [0, 1, 2]
        );
        let counts = |class| {
            (1..=5)
                .map(|k| residue_coefficient_count(k, class))
                .collect::<Vec<_>>()
        };
        assert_eq!(counts(RankClass::UpToN), [5, 10, 10, 5, 1]);
        assert_eq!(counts(RankClass::NPlus1), [16, 20, 15, 6, 1]);
        assert_eq!(counts(RankClass::NPlus2), [16, 35, 21, 7, 1]);
        // Bubble, triangle and box: binomial(r - n + 5, 5 - k) (theory), the
        // ordinary rank padded to Ninja's rank-n layout.
        for (k, class) in [
            (2, RankClass::NPlus2),
            (3, RankClass::NPlus2),
            (4, RankClass::NPlus2),
        ] {
            let excess = class.excess() as usize;
            let binomial = |a: usize, b: usize| (0..b).fold(1, |acc, i| acc * (a - i) / (i + 1));
            assert_eq!(
                residue_coefficient_count(k, class),
                binomial(excess + 5, 5 - k)
            );
        }
    }

    /// Parents enumerated from supersets equal the former scan of every earlier
    /// cut (same indices, ascending), so every reduction order is unchanged.
    #[test]
    fn parents_equal_the_scan_over_earlier_cuts() {
        for n in 1..=10 {
            let topology = Topology::new(n, 0).unwrap();
            let cuts = topology.cuts();
            for (index, cut) in cuts.iter().enumerate() {
                let scanned: Vec<usize> = (0..index)
                    .filter(|&parent| {
                        cuts[parent].partition.len() > cut.partition.len()
                            && cut
                                .partition
                                .iter()
                                .all(|i| cuts[parent].partition.contains(i))
                    })
                    .collect();
                assert_eq!(cut.parents, scanned, "n {n} cut {index}");
            }
        }
    }

    /// Twenty-four propagators (55454 cuts) took about five seconds per
    /// topology, and every reduction stage builds one.
    #[test]
    fn twenty_four_propagator_topology_builds_quickly() {
        let start = std::time::Instant::now();
        let topology = Topology::new(24, 0).unwrap();
        let elapsed = start.elapsed();
        assert_eq!(topology.cuts().len(), 55454);
        assert!(elapsed.as_secs() < 5, "{elapsed:?}");
    }
}
