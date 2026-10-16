//! Persistent validation coordinator; symbolic construction is outside runs.
use super::foreign::{MasterWire, NumeratorWire};
use super::masters::{Masters, masters};
use super::{
    Complex, FourMomentum, Pair, Provider, Result, dimensions, guard, provider, read, write,
};
use ninjars::{
    Amplitude, AmplitudeInput, EvaluationStatus, Laurent, MasterProvider, MasterRequest,
    MasterStatus, Numerator,
};
use ninjars_symbolica::{
    SymbolicaAmplitudeKernels, SymbolicaCuts, SymbolicaHighCuts, SymbolicaMasters,
};
type C = Complex<f64>;
pub struct Session {
    n: usize,
    rank: usize,
    amplitude: Amplitude<C, SymbolicaAmplitudeKernels<f64>>,
    low: SymbolicaCuts<f64>,
    high: SymbolicaHighCuts<f64>,
    numerator: NumeratorWire,
    masters: MasterWire,
    production: ProductionMasters,
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
            production: ProductionMasters {
                inner: SymbolicaMasters::new(0.),
                calls: 0,
                rows: 0,
            },
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
        dimensions(no, rows, 5, "session output")?;
        let values = execute(
            unsafe { session(handle) }?,
            unsafe { provider(provider_handle) }?,
            Some(unsafe { masters(master_handle) }?),
            rows,
            Inputs {
                shifts: unsafe { read(shifts, nv) }?,
                masses: unsafe { read(masses_ptr, nm) }?,
                scales: unsafe { read(scales, ns) }?,
                matrix: unsafe { read(matrix, nmat) }?,
            },
        )?;
        unsafe { write(out, no, values) }
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

struct ProductionMasters {
    inner: SymbolicaMasters<f64>,
    calls: u64,
    rows: u64,
}
impl MasterProvider<C> for ProductionMasters {
    fn evaluate(
        &mut self,
        request: MasterRequest<'_, C>,
        output: &mut [Laurent<C>],
    ) -> ninjars::Result<()> {
        self.calls += 1;
        self.rows += request.rows() as u64;
        #[cfg(feature = "allocation-diagnostics")]
        let _allocation_scope = super::allocation::Scope::new(1);
        self.inner.evaluate(request, output)
    }
    fn evaluate_batch(
        &mut self,
        request: MasterRequest<'_, C>,
        output: &mut [Laurent<C>],
        statuses: &mut [MasterStatus],
    ) -> ninjars::Result<()> {
        self.calls += 1;
        self.rows += request.rows() as u64;
        #[cfg(feature = "allocation-diagnostics")]
        let _allocation_scope = super::allocation::Scope::new(1);
        self.inner.evaluate_batch(request, output, statuses)
    }
}
struct Inputs<'a> {
    shifts: &'a [Pair],
    masses: &'a [Pair],
    scales: &'a [Pair],
    matrix: &'a [Pair],
}
fn execute<'a>(
    s: &'a mut Session,
    p: &mut Provider,
    master: Option<&mut Masters>,
    rows: usize,
    inputs: Inputs<'_>,
) -> Result<&'a [Pair]> {
    if p.n != s.n || p.numerator.rank() != s.rank {
        return Err("session/provider topology mismatch".into());
    }
    dimensions(inputs.shifts.len(), rows, s.n * 4, "session shifts")?;
    dimensions(inputs.masses.len(), rows, s.n, "session masses")?;
    dimensions(inputs.scales.len(), rows, 1, "session scales")?;
    dimensions(inputs.matrix.len(), rows, s.n * s.n, "session matrix")?;
    s.shifts.clear();
    s.shifts.extend(
        inputs
            .shifts
            .as_chunks::<4>()
            .0
            .iter()
            .map(|q| FourMomentum::from(q.map(C::from))),
    );
    s.masses.clear();
    s.masses.extend(inputs.masses.iter().copied().map(C::from));
    s.scales.clear();
    s.scales.extend(inputs.scales.iter().copied().map(C::from));
    s.matrix.clear();
    s.matrix.extend(inputs.matrix.iter().copied().map(C::from));
    s.sampling.resize(rows, C::new(1., 0.));
    s.amplitude.reset_batch(rows);
    if let Some(m) = master {
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
    } else {
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
                &mut s.production,
            )
            .map_err(|e| e.to_string())?;
    }
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
    Ok(&s.output)
}
/// # Safety
/// Same ownership/shape contract as sp_session_run. Uses direct production
/// SymbolicaMasters, without the validation master FFI or replay table.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn sp_session_run_production(
    handle: *mut Session,
    provider_handle: *mut Provider,
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
        dimensions(no, rows, 5, "session output")?;
        let values = execute(
            unsafe { session(handle) }?,
            unsafe { provider(provider_handle) }?,
            None,
            rows,
            Inputs {
                shifts: unsafe { read(shifts, nv) }?,
                masses: unsafe { read(masses_ptr, nm) }?,
                scales: unsafe { read(scales, ns) }?,
                matrix: unsafe { read(matrix, nmat) }?,
            },
        )?;
        unsafe { write(out, no, values) }
    })
}
/// # Safety
/// Live exclusive handle and two writable cells: direct production master
/// calls and attempted rows. Successful runs must have all lane statuses good.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn sp_session_production_stats(
    handle: *mut Session,
    out: *mut u64,
    len: usize,
) -> i32 {
    guard(|| {
        let state = unsafe { session(handle) }?;
        unsafe { write(out, len, &[state.production.calls, state.production.rows]) }
    })
}
