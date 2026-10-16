//! Validation-only union capture and exact-key replay of scalar/tensor masters.
//! Capture uses the production Symbolica provider; frozen replay never computes.

use super::{
    Complex, FourMomentum, Pair, Provider, Result, dimensions, guard, provider, read, write,
};
use ninjars::{
    AmplitudeInput, Error, EvaluationStatus, Laurent, MasterIntegral, MasterProvider,
    MasterRequest, MasterStatus, Numerator,
};
use ninjars_symbolica::{SymbolicaAmplitudeKernels, SymbolicaCuts, SymbolicaMasters};
use std::collections::HashMap;

#[derive(Clone, Debug, Eq, PartialEq, Hash)]
struct Key {
    integral: MasterIntegral,
    bits: Vec<u64>,
}
struct Entry {
    values: Laurent<Complex<f64>>,
    sources: u8,
}
pub struct Masters {
    backend: SymbolicaMasters<f64>,
    entries: HashMap<Key, Entry>,
    frozen: bool,
    live: bool,
    backend_calls: usize,
    backend_rows: usize,
    replay_rows: usize,
    misses: usize,
}
impl Masters {
    fn new() -> Self {
        Self {
            backend: SymbolicaMasters::new(0.),
            entries: HashMap::new(),
            frozen: false,
            live: false,
            backend_calls: 0,
            backend_rows: 0,
            replay_rows: 0,
            misses: 0,
        }
    }
    fn evaluate(
        &mut self,
        request: MasterRequest<'_, Complex<f64>>,
        source: u8,
    ) -> Result<Vec<Laurent<Complex<f64>>>> {
        request
            .validate(request.rows())
            .map_err(|e| e.to_string())?;
        if source != 1 && source != 2 {
            return Err("master source must be Rust1 or originalNinja2".into());
        }
        if request
            .invariants
            .iter()
            .chain(request.masses_squared)
            .chain(request.scales_squared)
            .any(|v| !v.re.is_finite() || !v.im.is_finite())
        {
            return Err("nonfinite master key".into());
        }
        if self.live {
            // Complete shared-provider controls execute the master expressions
            // on every request, including repeated points. No key lookup or
            // insertion occurs in this mode.
            let mut values = vec![[Complex::new(0., 0.); 3]; request.rows()];
            let mut statuses = vec![MasterStatus::Success; request.rows()];
            self.backend_calls += 1;
            #[cfg(feature = "execution-diagnostics")]
            let execution_scope =
                super::execution::Scope::master(request.integral, source, request.rows());
            self.backend
                .evaluate_batch(request, &mut values, &mut statuses)
                .map_err(|e| e.to_string())?;
            #[cfg(feature = "execution-diagnostics")]
            drop(execution_scope);
            if let Some((row, status)) = statuses
                .iter()
                .enumerate()
                .find(|(_, status)| **status != MasterStatus::Success)
            {
                return Err(format!("shared live master row {row} failed: {status:?}"));
            }
            self.backend_rows += values.len();
            return Ok(values);
        }
        let ni = request.integral.invariant_count();
        let nm = request.integral.mass_count();
        let keys = (0..request.rows())
            .map(|row| Key {
                integral: request.integral,
                bits: request.invariants[row * ni..(row + 1) * ni]
                    .iter()
                    .chain(&request.masses_squared[row * nm..(row + 1) * nm])
                    .chain(&request.scales_squared[row..row + 1])
                    .flat_map(|v| [v.re.to_bits(), v.im.to_bits()])
                    .collect(),
            })
            .collect::<Vec<_>>();
        let missing = keys
            .iter()
            .enumerate()
            .filter(|(_, key)| !self.entries.contains_key(key))
            .map(|(row, _)| row)
            .collect::<Vec<_>>();
        if self.frozen && !missing.is_empty() {
            self.misses += missing.len();
            return Err(format!(
                "frozen shared-master replay miss for {:?}, rows {missing:?}",
                request.integral
            ));
        }
        if !missing.is_empty() {
            let invariants = missing
                .iter()
                .flat_map(|row| request.invariants[row * ni..(row + 1) * ni].iter().copied())
                .collect::<Vec<_>>();
            let masses = missing
                .iter()
                .flat_map(|row| {
                    request.masses_squared[row * nm..(row + 1) * nm]
                        .iter()
                        .copied()
                })
                .collect::<Vec<_>>();
            let scales = missing
                .iter()
                .map(|&row| request.scales_squared[row])
                .collect::<Vec<_>>();
            let mut values = vec![[Complex::new(0., 0.); 3]; missing.len()];
            let mut statuses = vec![MasterStatus::Success; missing.len()];
            self.backend_calls += 1;
            #[cfg(feature = "execution-diagnostics")]
            let execution_scope =
                super::execution::Scope::master(request.integral, source, missing.len());
            self.backend
                .evaluate_batch(
                    MasterRequest {
                        integral: request.integral,
                        invariants: &invariants,
                        masses_squared: &masses,
                        scales_squared: &scales,
                    },
                    &mut values,
                    &mut statuses,
                )
                .map_err(|e| e.to_string())?;
            #[cfg(feature = "execution-diagnostics")]
            drop(execution_scope);
            if let Some((row, status)) = statuses
                .iter()
                .enumerate()
                .find(|(_, s)| **s != MasterStatus::Success)
            {
                return Err(format!(
                    "shared master capture row {} failed: {status:?}",
                    missing[row]
                ));
            }
            // No table mutation occurs before every missing numerical row succeeds.
            self.backend_rows += missing.len();
            for (&row, values) in missing.iter().zip(values) {
                self.entries
                    .insert(keys[row].clone(), Entry { values, sources: 0 });
            }
        }
        if self.frozen {
            self.replay_rows += request.rows();
        }
        Ok(keys
            .iter()
            .map(|key| {
                let entry = self
                    .entries
                    .get_mut(key)
                    .expect("all missing keys inserted");
                entry.sources |= source;
                entry.values
            })
            .collect())
    }
    fn stats(&self) -> [u64; 7] {
        let count = |mask| {
            self.entries
                .values()
                .filter(|entry| entry.sources == mask)
                .count() as u64
        };
        [
            self.entries.len() as u64,
            count(3),
            count(1),
            count(2),
            self.backend_rows as u64,
            self.replay_rows as u64,
            self.misses as u64,
        ]
    }
}
fn integral(which: u32) -> Result<MasterIntegral> {
    Ok(match which {
        0 => MasterIntegral::A0,
        1 => MasterIntegral::B0,
        2 => MasterIntegral::B1,
        3 => MasterIntegral::B11,
        4 => MasterIntegral::B111,
        5 => MasterIntegral::C0,
        6 => MasterIntegral::D0,
        _ => return Err("unsupported shared master identifier".into()),
    })
}
pub(crate) unsafe fn masters<'a>(handle: *mut Masters) -> Result<&'a mut Masters> {
    if handle.is_null() || !handle.is_aligned() {
        return Err("invalid shared-master handle".into());
    }
    Ok(unsafe { &mut *handle })
}
struct RustMasters<'a>(&'a mut Masters);
impl MasterProvider<Complex<f64>> for RustMasters<'_> {
    fn evaluate(
        &mut self,
        request: MasterRequest<'_, Complex<f64>>,
        out: &mut [Laurent<Complex<f64>>],
    ) -> ninjars::Result<()> {
        request.validate(out.len())?;
        let values = self.0.evaluate(request, 1).map_err(Error::Unsupported)?;
        out.copy_from_slice(&values);
        Ok(())
    }
    fn evaluate_batch(
        &mut self,
        request: MasterRequest<'_, Complex<f64>>,
        out: &mut [Laurent<Complex<f64>>],
        statuses: &mut [MasterStatus],
    ) -> ninjars::Result<()> {
        request.validate(out.len())?;
        request.validate(statuses.len())?;
        self.evaluate(request, out)?;
        statuses.fill(MasterStatus::Success);
        Ok(())
    }
}
/// # Safety
/// `out` is a valid writable pointer slot. Calls on a returned handle serialize.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn sp_master_create(out: *mut *mut Masters) -> i32 {
    guard(|| {
        if out.is_null() || !out.is_aligned() {
            return Err("invalid master handle output".into());
        }
        unsafe { out.write(Box::into_raw(Box::new(Masters::new()))) };
        Ok(())
    })
}
/// # Safety
/// All explicit arrays are readable/writable for their lengths; handle is live
/// and exclusively borrowed. The output is committed only after all rows succeed.
#[unsafe(no_mangle)]
#[inline(never)]
pub unsafe extern "C" fn sp_master_call(
    handle: *mut Masters,
    which: u32,
    source: u32,
    rows: usize,
    invariants: *const Pair,
    ni: usize,
    masses_ptr: *const Pair,
    nm: usize,
    scales: *const Pair,
    ns: usize,
    out: *mut Pair,
    no: usize,
) -> i32 {
    guard(|| {
        #[cfg(feature = "execution-diagnostics")]
        let _execution_scope =
            super::execution::Scope::master_wire(u8::try_from(source).unwrap_or(0), rows);
        #[cfg(feature = "allocation-diagnostics")]
        let _allocation_scope = super::allocation::Scope::new(1);
        let state = unsafe { masters(handle) }?;
        let integral = integral(which)?;
        dimensions(ni, rows, integral.invariant_count(), "master invariants")?;
        dimensions(nm, rows, integral.mass_count(), "master masses")?;
        dimensions(ns, rows, 1, "master scales")?;
        dimensions(no, rows, 3, "master output")?;
        let invariants = unsafe { read(invariants, ni) }?
            .iter()
            .copied()
            .map(Complex::from)
            .collect::<Vec<_>>();
        let masses = unsafe { read(masses_ptr, nm) }?
            .iter()
            .copied()
            .map(Complex::from)
            .collect::<Vec<_>>();
        let scales = unsafe { read(scales, ns) }?
            .iter()
            .copied()
            .map(Complex::from)
            .collect::<Vec<_>>();
        let source = u8::try_from(source).map_err(|_| "master source overflow")?;
        let values = state.evaluate(
            MasterRequest {
                integral,
                invariants: &invariants,
                masses_squared: &masses,
                scales_squared: &scales,
            },
            source,
        )?;
        let values = values
            .into_iter()
            .flatten()
            .map(Pair::from)
            .collect::<Vec<_>>();
        unsafe { write(out, no, &values) }
    })
}
/// # Safety
/// The handle is live and exclusively borrowed. Freeze is irreversible.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn sp_master_freeze(handle: *mut Masters) -> i32 {
    guard(|| {
        let state = unsafe { masters(handle) }?;
        if state.live {
            return Err("live master provider cannot freeze".into());
        }
        state.frozen = true;
        Ok(())
    })
}
/// # Safety
/// Live exclusive handle. Live mode cannot replace frozen replay, and skips
/// every result-table lookup/insertion. Existing capture entries are retained
/// only for diagnostics and are never consulted by live execution.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn sp_master_set_live(handle: *mut Masters) -> i32 {
    guard(|| {
        let state = unsafe { masters(handle) }?;
        if state.frozen {
            return Err("frozen replay cannot become a live master provider".into());
        }
        state.live = true;
        Ok(())
    })
}
/// # Safety
/// Live exclusive handle and three writable u64 cells: backend calls (including
/// failed calls), successful backend rows, live-mode flag.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn sp_master_execution_stats(
    handle: *mut Masters,
    out: *mut u64,
    len: usize,
) -> i32 {
    guard(|| {
        let state = unsafe { masters(handle) }?;
        unsafe {
            write(
                out,
                len,
                &[
                    state.backend_calls as u64,
                    state.backend_rows as u64,
                    u64::from(state.live),
                ],
            )
        }
    })
}
/// # Safety
/// The handle is live/exclusive, and `out` has space for seven u64 entries.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn sp_master_stats(handle: *mut Masters, out: *mut u64, len: usize) -> i32 {
    guard(|| {
        let state = unsafe { masters(handle) }?;
        unsafe { write(out, len, &state.stats()) }
    })
}
/// # Safety
/// A created handle must be destroyed once, without outstanding calls/borrows.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn sp_master_destroy(handle: *mut Masters) -> i32 {
    guard(|| {
        if !handle.is_null() {
            if !handle.is_aligned() {
                return Err("invalid master handle".into());
            }
            drop(unsafe { Box::from_raw(handle) });
        }
        Ok(())
    })
}

/// Safe Rust amplitude using the same bound numerator and shared master table.
/// The initial control covers two-denominator ordinary/higher-rank numerators.
/// # Safety
/// Both handles are live and mutually disjoint. Every array has its declared
/// valid length; out has five cells. No call on either handle is concurrent.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn sp_rust_bubble(
    handle: *mut Provider,
    master_handle: *mut Masters,
    shifts: *const Pair,
    nv: usize,
    mass_ptr: *const Pair,
    nm: usize,
    scale: Pair,
    matrix: *const Pair,
    nmat: usize,
    out: *mut Pair,
    no: usize,
) -> i32 {
    guard(|| {
        let provider = unsafe { provider(handle) }?;
        if provider.n != 2 {
            return Err("Rust amplitude probe requires a bubble fixture".into());
        }
        dimensions(nv, 2, 4, "bubble shifts")?;
        dimensions(nm, 1, 2, "bubble masses")?;
        dimensions(nmat, 2, 2, "bubble matrix")?;
        dimensions(no, 1, 5, "amplitude output")?;
        let shifts = unsafe { read(shifts, nv) }?
            .as_chunks::<4>()
            .0
            .iter()
            .map(|row| FourMomentum::from(row.map(Complex::from)))
            .collect::<Vec<_>>();
        let masses = unsafe { read(mass_ptr, nm) }?
            .iter()
            .copied()
            .map(Complex::from)
            .collect::<Vec<_>>();
        let matrix = unsafe { read(matrix, nmat) }?
            .iter()
            .copied()
            .map(Complex::from)
            .collect::<Vec<_>>();
        let mut amplitude = SymbolicaAmplitudeKernels::new(2, provider.numerator.rank(), 0.)
            .map_err(|e| e.to_string())?
            .into_amplitude(1)
            .map_err(|e| e.to_string())?;
        let mut geometry = SymbolicaCuts::new(0.).map_err(|e| e.to_string())?;
        let mut masters = RustMasters(unsafe { masters(master_handle) }?);
        amplitude
            .evaluate_batch_with_low_cuts(
                AmplitudeInput {
                    shifts: &shifts,
                    masses_squared: &masses,
                    scales_squared: &[scale.into()],
                    sampling_scales: &[Complex::new(1., 0.)],
                    s_matrix: Some(&matrix),
                },
                &mut provider.numerator,
                &mut geometry,
                &mut masters,
            )
            .map_err(|e| e.to_string())?;
        if amplitude.statuses() != [EvaluationStatus::Success] {
            return Err(format!("Rust amplitude failed: {:?}", amplitude.statuses()));
        }
        let values = amplitude.results()[0].map(Pair::from);
        unsafe { write(out, no, &values) }
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    fn c(x: f64) -> Complex<f64> {
        Complex::new(x, 0.)
    }
    #[test]
    fn live_mode_reexecutes_repeated_rows_without_table_access() {
        let mut state = Masters::new();
        let masses = [Complex::new(2., 0.), Complex::new(2., 0.)];
        let scales = [Complex::new(1., 0.); 2];
        let request = MasterRequest {
            integral: MasterIntegral::A0,
            invariants: &[],
            masses_squared: &masses,
            scales_squared: &scales,
        };
        let expected = state.evaluate(request, 1).unwrap();
        let entries = state.entries.len();
        let calls = state.backend_calls;
        let rows = state.backend_rows;
        state.live = true;
        for _ in 0..2 {
            assert_eq!(state.evaluate(request, 2).unwrap(), expected);
        }
        assert_eq!(state.entries.len(), entries);
        assert_eq!(state.backend_calls, calls + 2);
        assert_eq!(state.backend_rows, rows + 4);
        assert_eq!(state.replay_rows, 0);
        assert!(state.entries.values().all(|entry| entry.sources == 1));
        assert_ne!(unsafe { sp_master_freeze(&mut state) }, 0);
        assert!(!state.frozen);
    }

    #[test]
    fn capture_union_replays_permuted_batches_and_rejects_changed_scale_and_order() {
        let mut state = Masters::new();
        let request = MasterRequest {
            integral: MasterIntegral::B1,
            invariants: &[c(-1.), c(-2.)],
            masses_squared: &[c(2.), c(3.), c(4.), c(5.)],
            scales_squared: &[c(1.), c(2.)],
        };
        let expected = state.evaluate(request, 1).unwrap();
        state
            .evaluate(
                MasterRequest {
                    integral: MasterIntegral::A0,
                    invariants: &[],
                    masses_squared: &[c(7.)],
                    scales_squared: &[c(3.)],
                },
                2,
            )
            .unwrap();
        state.evaluate(request, 2).unwrap();
        assert_eq!(&state.stats()[..5], &[3, 2, 0, 1, 3]);
        state.frozen = true;
        let actual = state
            .evaluate(
                MasterRequest {
                    integral: MasterIntegral::B1,
                    invariants: &[c(-2.), c(-1.), c(-2.)],
                    masses_squared: &[c(4.), c(5.), c(2.), c(3.), c(4.), c(5.)],
                    scales_squared: &[c(2.), c(1.), c(2.)],
                },
                1,
            )
            .unwrap();
        assert_eq!(actual, [expected[1], expected[0], expected[1]]);
        assert!(
            state
                .evaluate(
                    MasterRequest {
                        scales_squared: &[c(2.), c(2.)],
                        ..request
                    },
                    1
                )
                .is_err()
        );
        assert!(
            state
                .evaluate(
                    MasterRequest {
                        masses_squared: &[c(3.), c(2.), c(4.), c(5.)],
                        ..request
                    },
                    1
                )
                .is_err()
        );
        assert!(
            state
                .evaluate(
                    MasterRequest {
                        integral: MasterIntegral::B0,
                        ..request
                    },
                    1
                )
                .is_err()
        );
        assert_eq!(state.backend_rows, 3);
        assert_eq!(state.entries.len(), 3);
        assert!(
            state
                .evaluate(
                    MasterRequest {
                        integral: MasterIntegral::B0,
                        invariants: &[],
                        masses_squared: &[],
                        scales_squared: &[]
                    },
                    1
                )
                .unwrap()
                .is_empty()
        );
    }
    #[test]
    fn failed_master_batch_preserves_output_and_status() {
        let mut state = Masters::new();
        state.frozen = true;
        let request = MasterRequest {
            integral: MasterIntegral::A0,
            invariants: &[],
            masses_squared: &[c(1.)],
            scales_squared: &[c(1.)],
        };
        let mut provider = RustMasters(&mut state);
        let mut out = [[c(77.); 3]];
        let mut statuses = [MasterStatus::Success];
        assert!(
            provider
                .evaluate_batch(request, &mut out, &mut statuses)
                .is_err()
        );
        assert_eq!(out, [[c(77.); 3]]);
        assert_eq!(statuses, [MasterStatus::Success]);
    }

    /// Positive masses and strictly negative real invariants give a positive
    /// Feynman-parameter denominator. The current analytic implementation still
    /// produces a nonfinite result here; this is an outstanding evaluation bug,
    /// not an accepted physical singularity or a supported-domain exclusion.
    #[test]
    fn euclidean_c0_exceptional_point_must_be_finite() {
        let mut provider = SymbolicaMasters::new(0.);
        let mut output = [[c(77.); 3]];
        let mut statuses = [MasterStatus::Success];
        provider
            .evaluate_batch(
                MasterRequest {
                    integral: MasterIntegral::C0,
                    invariants: &[c(-1.), c(-2.), c(-3.)],
                    masses_squared: &[c(2.), c(3.), c(5.)],
                    scales_squared: &[c(7.)],
                },
                &mut output,
                &mut statuses,
            )
            .unwrap();
        assert_eq!(statuses, [MasterStatus::Success]);
        assert!(
            output[0]
                .iter()
                .all(|v| v.re.is_finite() && v.im.is_finite())
        );
        // Independent 60-digit Feynman-parameter quadrature:
        // -0.135212396225436074537371827943492306170596769774993734247205.
        assert!((output[0][0].re + 0.135_212_396_225_436_08).abs() < 1e-12);
        assert!(output[0][0].im.abs() < 1e-12);
        assert_eq!(output[0][1..], [c(0.), c(0.)]);
    }
}
