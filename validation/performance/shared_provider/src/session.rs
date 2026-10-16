//! Persistent validation coordinator; symbolic construction is outside runs.
use super::foreign::{MasterWire, NumeratorWire};
use super::masters::{Masters, masters};
use super::{
    Complex, FourMomentum, Pair, Provider, Result, dimensions, guard, provider, read, write,
};
use ninjars::{Amplitude, AmplitudeInput, EvaluationStatus, Numerator};
use ninjars_symbolica::{SymbolicaAmplitudeKernels, SymbolicaCuts, SymbolicaHighCuts};
type C = Complex<f64>;
pub struct Session {
    n: usize,
    rank: usize,
    amplitude: Amplitude<C, SymbolicaAmplitudeKernels<f64>>,
    low: SymbolicaCuts<f64>,
    high: SymbolicaHighCuts<f64>,
    numerator: NumeratorWire,
    masters: MasterWire,
    shifts: Vec<FourMomentum<C>>,
    masses: Vec<C>,
    scales: Vec<C>,
    sampling: Vec<C>,
    matrix: Vec<C>,
    output: Vec<Pair>,
}
unsafe fn session<'a>(handle: *mut Session) -> Result<&'a mut Session> {
    if handle.is_null() || !handle.is_aligned() {
        return Err("invalid session handle".into());
    }
    Ok(unsafe { &mut *handle })
}
/// # Safety
/// out must be a valid writable slot. Returned handles are used serially.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn sp_session_create(
    n: usize,
    rank: usize,
    rows: usize,
    out: *mut *mut Session,
) -> i32 {
    guard(|| {
        if out.is_null() || !out.is_aligned() {
            return Err("invalid session output".into());
        }
        let state = Session {
            n,
            rank,
            amplitude: SymbolicaAmplitudeKernels::new(n, rank, 0.)
                .and_then(|k| k.into_amplitude(rows))
                .map_err(|e| e.to_string())?,
            low: SymbolicaCuts::new(0.).map_err(|e| e.to_string())?,
            high: SymbolicaHighCuts::new(0.).map_err(|e| e.to_string())?,
            numerator: NumeratorWire::default(),
            masters: MasterWire::default(),
            shifts: Vec::new(),
            masses: Vec::new(),
            scales: Vec::new(),
            sampling: Vec::new(),
            matrix: Vec::new(),
            output: Vec::new(),
        };
        unsafe { out.write(Box::into_raw(Box::new(state))) };
        Ok(())
    })
}
/// # Safety
/// All handles are live, exclusive and mutually disjoint. Buffers are valid for
/// the declared lengths and do not overlap handles/output. No concurrent calls.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn sp_session_run(
    handle: *mut Session,
    provider_handle: *mut Provider,
    master_handle: *mut Masters,
    rows: usize,
    shifts: *const Pair,
    nv: usize,
    masses_ptr: *const Pair,
    nm: usize,
    scales: *const Pair,
    ns: usize,
    matrix: *const Pair,
    nmat: usize,
    out: *mut Pair,
    no: usize,
) -> i32 {
    guard(|| {
        let s = unsafe { session(handle) }?;
        let p = unsafe { provider(provider_handle) }?;
        let m = unsafe { masters(master_handle) }?;
        if p.n != s.n || p.numerator.rank() != s.rank {
            return Err("session/provider topology mismatch".into());
        }
        dimensions(nv, rows, s.n * 4, "session shifts")?;
        dimensions(nm, rows, s.n, "session masses")?;
        dimensions(ns, rows, 1, "session scales")?;
        dimensions(nmat, rows, s.n * s.n, "session matrix")?;
        dimensions(no, rows, 5, "session output")?;
        s.shifts.clear();
        s.shifts.extend(
            unsafe { read(shifts, nv) }?
                .as_chunks::<4>()
                .0
                .iter()
                .map(|q| FourMomentum::from(q.map(C::from))),
        );
        s.masses.clear();
        s.masses.extend(
            unsafe { read(masses_ptr, nm) }?
                .iter()
                .copied()
                .map(C::from),
        );
        s.scales.clear();
        s.scales
            .extend(unsafe { read(scales, ns) }?.iter().copied().map(C::from));
        s.matrix.clear();
        s.matrix
            .extend(unsafe { read(matrix, nmat) }?.iter().copied().map(C::from));
        s.sampling.resize(rows, C::new(1., 0.));
        s.amplitude.reset_batch(rows);
        s.amplitude
            .evaluate_batch_with_inputs(
                AmplitudeInput {
                    shifts: &s.shifts,
                    masses_squared: &s.masses,
                    scales_squared: &s.scales,
                    sampling_scales: &s.sampling,
                    s_matrix: Some(&s.matrix),
                },
                &mut s.numerator.bind(p),
                &mut s.low,
                &mut s.high,
                &mut s.masters.bind(m),
            )
            .map_err(|e| e.to_string())?;
        if s.amplitude
            .statuses()
            .iter()
            .any(|status| *status != EvaluationStatus::Success)
        {
            return Err(format!(
                "session reduction failed: {:?}",
                s.amplitude.statuses()
            ));
        }
        s.output.clear();
        s.output.extend(
            s.amplitude
                .results()
                .iter()
                .flatten()
                .copied()
                .map(Pair::from),
        );
        unsafe { write(out, no, &s.output) }
    })
}
/// # Safety
/// Live exclusive handle and four writable u64 cells.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn sp_session_stats(handle: *mut Session, out: *mut u64, len: usize) -> i32 {
    guard(|| {
        let s = unsafe { session(handle) }?;
        unsafe {
            write(
                out,
                len,
                &[
                    s.numerator.calls,
                    s.numerator.rows,
                    s.masters.calls,
                    s.masters.rows,
                ],
            )
        }
    })
}
/// # Safety
/// Destroy a created handle once, with no outstanding calls. Null is a no-op.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn sp_session_destroy(handle: *mut Session) -> i32 {
    guard(|| {
        if !handle.is_null() {
            if !handle.is_aligned() {
                return Err("invalid session handle".into());
            }
            drop(unsafe { Box::from_raw(handle) });
        }
        Ok(())
    })
}
