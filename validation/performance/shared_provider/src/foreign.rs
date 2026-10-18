//! Retained wire adapters for the Rust reducer. Both languages enter exactly
//! the same non-inlined C ABI functions; algebra stays in the shared providers.
use super::masters::{Masters, sp_master_call};
use super::{
    Complex, ExpansionKind, ExpansionRequest, Identity, LAST_ERROR, Numerator, NumeratorRequest,
    Pair, Provider, sp_call,
};
use ninjars::{Error, Laurent, MasterIntegral, MasterProvider, MasterRequest, MasterStatus};
use std::marker::PhantomData;
type C = Complex<f64>;
fn checked(code: i32) -> ninjars::Result<()> {
    if code == 0 {
        Ok(())
    } else {
        Err(Error::Unsupported(
            LAST_ERROR.with(|error| error.borrow().clone()),
        ))
    }
}
#[derive(Default)]
pub struct NumeratorWire {
    input: Vec<Pair>,
    output: Vec<Pair>,
    staged: Vec<C>,
    ids: Vec<Identity>,
    powers: Vec<i32>,
    partition: Vec<u8>,
    pub calls: u64,
    pub rows: u64,
}
pub struct ForeignNumerator<'a> {
    provider: *mut Provider,
    rank: usize,
    wire: &'a mut NumeratorWire,
    _borrow: PhantomData<&'a mut Provider>,
}
impl NumeratorWire {
    pub fn bind<'a>(&'a mut self, provider: &'a mut Provider) -> ForeignNumerator<'a> {
        ForeignNumerator {
            rank: provider.numerator.rank(),
            provider,
            wire: self,
            _borrow: PhantomData,
        }
    }
    fn start(&mut self) {
        self.input.clear();
        self.ids.clear();
    }
    fn id(&mut self, id: ninjars::SampleId) {
        self.ids.push(Identity {
            lane: id.lane as u64,
            sample: id.sample as u64,
        });
    }
    fn vector(&mut self, q: &ninjars::FourMomentum<C>) {
        self.input.extend(q.components().map(|v| Pair::from(*v)));
    }
    fn invoke(
        &mut self,
        provider: *mut Provider,
        kind: u32,
        partition: &[usize],
        columns: usize,
    ) -> ninjars::Result<()> {
        self.partition.clear();
        for &index in partition {
            self.partition.push(
                u8::try_from(index)
                    .map_err(|_| Error::Unsupported("partition index exceeds ABI range".into()))?,
            );
        }
        let len = self
            .ids
            .len()
            .checked_mul(columns)
            .ok_or_else(|| Error::Unsupported("callback output overflow".into()))?;
        self.output.resize(len, Pair::default());
        self.calls += 1;
        self.rows += self.ids.len() as u64;
        // SAFETY: bind retains the exclusive provider borrow. All wire buffers
        // are separate, live and dimensioned for the call; no aliasing casts.
        checked(unsafe {
            sp_call(
                provider,
                kind,
                self.powers.as_ptr(),
                self.powers.len(),
                self.ids.as_ptr(),
                self.ids.len(),
                self.input.as_ptr(),
                self.input.len(),
                self.partition.as_ptr(),
                self.partition.len(),
                self.output.as_mut_ptr(),
                self.output.len(),
            )
        })?;
        self.staged.extend(self.output.iter().copied().map(C::from));
        Ok(())
    }
}
impl Numerator<C> for ForeignNumerator<'_> {
    fn rank(&self) -> usize {
        self.rank
    }
    fn evaluate_batch(
        &mut self,
        requests: &[NumeratorRequest<'_, C>],
        output: &mut [C],
    ) -> ninjars::Result<()> {
        if requests.len() != output.len() {
            return Err(Error::Unsupported("wire direct output shape".into()));
        }
        let w = &mut self.wire;
        w.staged.clear();
        w.powers.clear();
        let mut offset = 0;
        while offset < requests.len() {
            let partition = requests[offset].partition;
            let end = offset
                + requests[offset..]
                    .iter()
                    .take_while(|r| r.partition == partition)
                    .count();
            w.start();
            for request in &requests[offset..end] {
                w.id(request.id);
                w.vector(request.q);
                w.input.push((*request.mu_squared).into());
            }
            w.invoke(self.provider, 0, partition, 1)?;
            offset = end;
        }
        output.copy_from_slice(&w.staged);
        Ok(())
    }
    fn expansion_batch(
        &mut self,
        kind: &ExpansionKind,
        requests: &[ExpansionRequest<'_, C>],
        output: &mut [C],
    ) -> ninjars::Result<()> {
        if requests.len().checked_mul(kind.len()) != Some(output.len()) {
            return Err(Error::Unsupported("wire expansion output shape".into()));
        }
        let w = &mut self.wire;
        w.staged.clear();
        w.powers.clear();
        let (which, vectors, parameters) = match kind {
            ExpansionKind::Mu { powers } => {
                w.powers.extend(powers);
                (1, 2, 0)
            }
            ExpansionKind::T3 { powers } => {
                w.powers
                    .extend(powers.iter().flat_map(|&(t, m)| [t, i32::from(m)]));
                (2, 3, 1)
            }
            ExpansionKind::T2 { powers } => {
                w.powers.extend(
                    powers
                        .iter()
                        .flat_map(|&(t, x, m)| [t, i32::from(x), i32::from(m)]),
                );
                (3, 4, 3)
            }
            // Rank n+2 boxes; original Ninja's callback ABI has no beta offset.
            ExpansionKind::MuShifted { .. } => {
                return Err(Error::Unsupported(
                    "the wire ABI has no shifted mu expansion (rank n+2)".into(),
                ));
            }
        };
        let mut offset = 0;
        while offset < requests.len() {
            let partition = requests[offset].partition;
            let end = offset
                + requests[offset..]
                    .iter()
                    .take_while(|r| r.partition == partition)
                    .count();
            w.start();
            for request in &requests[offset..end] {
                if request.parameters.len() != parameters
                    || !(request.vectors.len() == vectors
                        || which == 1 && request.vectors.len() == 1)
                {
                    return Err(Error::Unsupported("wire expansion input shape".into()));
                }
                w.id(request.id);
                for q in request.vectors {
                    w.vector(q);
                }
                if which == 1 && request.vectors.len() == 1 {
                    w.input.extend([Pair::default(); 4]);
                }
                w.input
                    .extend(request.parameters.iter().copied().map(Pair::from));
            }
            w.invoke(self.provider, which, partition, kind.len())?;
            offset = end;
        }
        output.copy_from_slice(&w.staged);
        Ok(())
    }
}
#[derive(Default)]
pub struct MasterWire {
    invariants: Vec<Pair>,
    masses: Vec<Pair>,
    scales: Vec<Pair>,
    output: Vec<Pair>,
    pub calls: u64,
    pub rows: u64,
}
pub struct ForeignMasters<'a> {
    handle: *mut Masters,
    wire: &'a mut MasterWire,
    _borrow: PhantomData<&'a mut Masters>,
}
impl MasterWire {
    pub fn bind<'a>(&'a mut self, master: &'a mut Masters) -> ForeignMasters<'a> {
        ForeignMasters {
            handle: master,
            wire: self,
            _borrow: PhantomData,
        }
    }
}
impl MasterProvider<C> for ForeignMasters<'_> {
    fn evaluate(
        &mut self,
        request: MasterRequest<'_, C>,
        out: &mut [Laurent<C>],
    ) -> ninjars::Result<()> {
        request.validate(out.len())?;
        let kind = match request.integral {
            MasterIntegral::A0 => 0,
            MasterIntegral::B0 => 1,
            MasterIntegral::B1 => 2,
            MasterIntegral::B11 => 3,
            MasterIntegral::B111 => 4,
            MasterIntegral::C0 => 5,
            MasterIntegral::D0 => 6,
            _ => {
                return Err(Error::Unsupported(
                    "master not in original Ninja ABI".into(),
                ));
            }
        };
        let w = &mut self.wire;
        w.invariants.clear();
        w.invariants
            .extend(request.invariants.iter().copied().map(Pair::from));
        w.masses.clear();
        w.masses
            .extend(request.masses_squared.iter().copied().map(Pair::from));
        w.scales.clear();
        w.scales
            .extend(request.scales_squared.iter().copied().map(Pair::from));
        let len = out
            .len()
            .checked_mul(3)
            .ok_or_else(|| Error::Unsupported("master wire size overflow".into()))?;
        w.output.resize(len, Pair::default());
        w.calls += 1;
        w.rows += request.rows() as u64;
        // SAFETY: bind retains the exclusive master borrow, separate from every
        // live, dimensioned wire buffer. Output is staged before caller commit.
        checked(unsafe {
            sp_master_call(
                self.handle,
                kind,
                1,
                request.rows(),
                w.invariants.as_ptr(),
                w.invariants.len(),
                w.masses.as_ptr(),
                w.masses.len(),
                w.scales.as_ptr(),
                w.scales.len(),
                w.output.as_mut_ptr(),
                w.output.len(),
            )
        })?;
        for (out, row) in out.iter_mut().zip(w.output.as_chunks::<3>().0) {
            *out = row.map(C::from);
        }
        Ok(())
    }
    fn evaluate_batch(
        &mut self,
        request: MasterRequest<'_, C>,
        out: &mut [Laurent<C>],
        statuses: &mut [MasterStatus],
    ) -> ninjars::Result<()> {
        request.validate(statuses.len())?;
        self.evaluate(request, out)?;
        statuses.fill(MasterStatus::Success);
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{control, control_vectors, fixture};
    #[test]
    fn wire_groups_mixed_partitions_preserving_permuted_lanes_and_transactionality() {
        let mut provider = fixture(0).unwrap();
        provider
            .numerator
            .set_parameter_batch(
                &[
                    C::new(2., 1.),
                    C::new(3., -1.),
                    C::new(-1., 2.),
                    C::new(1., 1.),
                ],
                2,
            )
            .unwrap();
        let mut wire = NumeratorWire::default();
        let vectors = control_vectors();
        let mu = C::new(7., 2.);
        let beta = [C::new(2., -1.)];
        let partitions = [[0, 1, 2], [0, 1, 3], [0, 1, 3]];
        let lanes = [1, 0, 1];
        let requests = (0..3)
            .map(|row| NumeratorRequest {
                id: ninjars::SampleId {
                    lane: lanes[row],
                    sample: 42,
                },
                q: &vectors[0],
                mu_squared: &mu,
                partition: &partitions[row],
            })
            .collect::<Vec<_>>();
        let mut out = [C::new(77., 0.); 3];
        wire.bind(&mut provider)
            .evaluate_batch(&requests, &mut out)
            .unwrap();
        for (row, &lane) in lanes.iter().enumerate() {
            assert_eq!(out[row], C::from(control(0, 0, lane).unwrap()[0]));
        }
        assert_eq!((wire.calls, wire.rows), (2, 3));
        let saved = out;
        assert!(
            wire.bind(&mut provider)
                .evaluate_batch(&requests[..2], &mut out)
                .is_err()
        );
        assert_eq!(out, saved);
        wire.bind(&mut provider)
            .evaluate_batch(&[], &mut [])
            .unwrap();
        let kind = ExpansionKind::T3 {
            powers: vec![
                (4, 0),
                (3, 0),
                (2, 0),
                (2, 1),
                (1, 0),
                (1, 1),
                (0, 0),
                (0, 1),
                (0, 2),
            ],
        };
        let requests = (0..3)
            .map(|row| ExpansionRequest {
                id: ninjars::SampleId {
                    lane: lanes[row],
                    sample: 42,
                },
                vectors: &vectors[..3],
                parameters: &beta,
                partition: &partitions[row],
            })
            .collect::<Vec<_>>();
        let mut out = vec![C::new(77., 0.); 27];
        wire.bind(&mut provider)
            .expansion_batch(&kind, &requests, &mut out)
            .unwrap();
        let expected = lanes
            .into_iter()
            .flat_map(|lane| control(0, 2, lane).unwrap().into_iter().map(C::from))
            .collect::<Vec<_>>();
        assert_eq!(out, expected);
        assert_eq!((wire.calls, wire.rows), (4, 6));
        let capacity = (
            wire.input.capacity(),
            wire.output.capacity(),
            wire.ids.capacity(),
        );
        wire.bind(&mut provider)
            .expansion_batch(&kind, &[], &mut [])
            .unwrap();
        wire.bind(&mut provider)
            .expansion_batch(&kind, &requests, &mut out)
            .unwrap();
        assert_eq!(out, expected);
        assert_eq!(
            capacity,
            (
                wire.input.capacity(),
                wire.output.capacity(),
                wire.ids.capacity()
            )
        );
    }
}
