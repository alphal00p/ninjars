use crate::{Error, Result};

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
        for index in 0..cuts.len() {
            cuts[index].parents = (0..index)
                .filter(|&parent| {
                    cuts[parent].partition.len() > cuts[index].partition.len()
                        && cuts[index]
                            .partition
                            .iter()
                            .all(|i| cuts[parent].partition.contains(i))
                })
                .collect();
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
}
