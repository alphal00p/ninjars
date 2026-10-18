use crate::{Error, Result};
use std::collections::HashMap;

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
        if rank > propagators.saturating_add(1) {
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
        assert!(Topology::new(6, 8).is_err());
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
