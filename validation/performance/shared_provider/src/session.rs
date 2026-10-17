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
    kernel_info: [u64; 10],
    rank_correction_info: [u64; 4],
    native_execution_info: [u64; 2],
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
fn execution_bound(
    selection: u32,
    policy: u32,
    block_rows: usize,
) -> Result<Option<std::num::NonZeroUsize>> {
    match (policy, block_rows) {
        (0, 0) => Ok(None),
        (1, rows) if rows > 0 => {
            if !matches!(selection, 1 | 2 | 4) {
                return Err("fixed native rows require a native reduction-kernel selection".into());
            }
            if !cfg!(feature = "native-kernels") {
                return Err("fixed native rows require native-kernels feature".into());
            }
            Ok(std::num::NonZeroUsize::new(rows))
        }
        _ => Err("invalid native execution policy/bound pair".into()),
    }
}
unsafe fn session<'a>(handle: *mut Session) -> Result<&'a mut Session> {
    if handle.is_null() || !handle.is_aligned() {
        return Err("invalid session handle".into());
    }
    Ok(unsafe { &mut *handle })
}
/// Validation IDs select public factory methods; buffer dimensions never select arithmetic.
fn kernels(
    n: usize,
    rank: usize,
    selection: u32,
) -> Result<(SymbolicaAmplitudeKernels<f64>, [u64; 10], [u64; 4])> {
    if selection > 4 {
        return Err("unknown reduction-kernel selection".into());
    }
    #[cfg(not(feature = "native-kernels"))]
    if matches!(selection, 1 | 2 | 4) {
        return Err("native selection requires native-kernels feature".into());
    }
    if matches!(selection, 3 | 4) && (n, rank) != (5, 4) {
        return Err("rank correction selection requires n=5 and rank=4".into());
    }
    let kernels = SymbolicaAmplitudeKernels::new(n, rank, 0.).map_err(|e| e.to_string())?;
    let kernels = if selection == 3 {
        kernels
            .with_rank_specialized_corrections()
            .map_err(|e| e.to_string())?
    } else {
        kernels
    };
    #[cfg(feature = "native-kernels")]
    let kernels = match selection {
        0 | 3 => kernels,
        1 => kernels.with_native_denominators(),
        2 => kernels.with_native_kernels(),
        4 => kernels
            .with_native_kernels()
            .with_native_rank_specialized_corrections()
            .map_err(|e| e.to_string())?,
        _ => unreachable!(),
    };
    let mut info = [0; 10];
    info[0] = u64::from(selection);
    info[1] = u64::from(cfg!(feature = "native-kernels"));
    #[cfg(feature = "native-kernels")]
    {
        use ninjars_symbolica::{KernelBackend, KernelFamily};
        let mut seen = [false; 8];
        for entry in kernels.kernel_backends() {
            let index = match entry.family {
                KernelFamily::T2Denominators => 0,
                KernelFamily::T3Denominators => 1,
                KernelFamily::BubbleSourceProjection => 2,
                KernelFamily::TriangleBubbleCorrection => 3,
                KernelFamily::TadpoleSourceProjection => 4,
                KernelFamily::OtherReduction => 5,
                KernelFamily::Integration => 6,
                KernelFamily::Accumulation => 7,
            };
            if seen[index] {
                return Err("duplicate kernel coverage family".into());
            }
            seen[index] = true;
            info[index + 2] = match entry.backend {
                KernelBackend::SymbolicaInterpreter => 0,
                KernelBackend::NativeRust => 1,
            };
        }
        if !seen.into_iter().all(|value| value) {
            return Err("incomplete kernel coverage".into());
        }
    }
    let enabled = kernels.rank_specialized_corrections_enabled();
    #[cfg(feature = "native-kernels")]
    let alternate_native = matches!(
        kernels.rank_specialized_correction_backend(),
        Some(ninjars_symbolica::KernelBackend::NativeRust)
    );
    #[cfg(not(feature = "native-kernels"))]
    let alternate_native = false;
    let conditional = [
        n as u64,
        rank as u64,
        u64::from(enabled),
        u64::from(alternate_native),
    ];
    Ok((kernels, info, conditional))
}
/// # Safety
/// out must be a valid writable slot. Returned handles are used serially.
/// Compatibility constructor: interpreted reduction, independent of numerator mode.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn sp_session_create(
    n: usize,
    rank: usize,
    rows: usize,
    out: *mut *mut Session,
) -> i32 {
    unsafe { sp_session_create_with_kernels(n, rank, rows, 0, out) }
}
/// # Safety
/// Same ownership contract as sp_session_create. Selection is setup-only:
/// 0 interpreted, 1 native denominators, 2 generic native catalog,
/// 3 interpreted rank corrections, 4 native catalog plus native rank corrections.
/// Invalid selection/build/topology leaves the caller's output slot unchanged.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn sp_session_create_with_kernels(
    n: usize,
    rank: usize,
    rows: usize,
    selection: u32,
    out: *mut *mut Session,
) -> i32 {
    unsafe { sp_session_create_with_execution(n, rank, rows, selection, 0, 0, out) }
}
/// # Safety
/// Same ownership/output contract as sp_session_create. Policy pairs are
/// Whole=(0,0), FixedRows=(1,positive bound). Bounds count complete native
/// kernel rows, not phase-space lanes. Invalid inputs preserve the output slot.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn sp_session_create_with_execution(
    n: usize,
    rank: usize,
    rows: usize,
    selection: u32,
    policy: u32,
    block_rows: usize,
    out: *mut *mut Session,
) -> i32 {
    guard(|| {
        if out.is_null() || !out.is_aligned() {
            return Err("invalid session output".into());
        }
        let bound = execution_bound(selection, policy, block_rows)?;
        let (kernels, kernel_info, rank_correction_info) = kernels(n, rank, selection)?;
        #[cfg(feature = "native-kernels")]
        let kernels = kernels.with_native_batch_execution(match bound {
            None => ninjars_symbolica::BatchExecutionPolicy::Whole,
            Some(rows) => ninjars_symbolica::BatchExecutionPolicy::FixedRows(rows),
        });
        #[cfg(feature = "native-kernels")]
        let native_execution_info = match kernels.native_batch_execution() {
            ninjars_symbolica::BatchExecutionPolicy::Whole => [0, 0],
            ninjars_symbolica::BatchExecutionPolicy::FixedRows(rows) => [1, rows.get() as u64],
        };
        #[cfg(not(feature = "native-kernels"))]
        let native_execution_info = {
            debug_assert!(bound.is_none());
            [0, 0]
        };
        let state = Session {
            kernel_info,
            rank_correction_info,
            native_execution_info,
            n,
            rank,
            amplitude: kernels.into_amplitude(rows).map_err(|e| e.to_string())?,
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
/// Live exclusive session and ten writable cells: selection, compiled-native
/// feature, then backend flags for T2,T3,bubble-source,triangle-bubble,tadpole-source,other,
/// integration,accumulation. Zero means interpreted, one means generated Rust.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn sp_session_kernel_info(
    handle: *mut Session,
    out: *mut u64,
    len: usize,
) -> i32 {
    guard(|| unsafe { write(out, len, &session(handle)?.kernel_info) })
}
/// # Safety
/// Live exclusive session and four writable cells: propagators, rank, enabled,
/// alternate backend (0 interpreted, 1 generated Rust). The original ten-cell
/// family accessor stays unchanged and describes generic/fallback coverage.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn sp_session_rank_correction_info(
    handle: *mut Session,
    out: *mut u64,
    len: usize,
) -> i32 {
    guard(|| unsafe { write(out, len, &session(handle)?.rank_correction_info) })
}
/// # Safety
/// Live exclusive session and exactly two writable cells: configured policy
/// tag and native kernel row bound. Whole returns [0,0], FixedRows [1,N].
#[unsafe(no_mangle)]
pub unsafe extern "C" fn sp_session_native_execution_info(
    handle: *mut Session,
    out: *mut u64,
    len: usize,
) -> i32 {
    guard(|| unsafe { write(out, len, &session(handle)?.native_execution_info) })
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
        #[cfg(feature = "execution-diagnostics")]
        let _execution_scope = super::execution::Scope::master(request.integral, 3, request.rows());
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
        #[cfg(feature = "execution-diagnostics")]
        let _execution_scope = super::execution::Scope::master(request.integral, 3, request.rows());
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

#[cfg(test)]
mod kernel_tests {
    use super::*;
    #[test]
    fn native_execution_rejects_bad_pairs_and_preserves_outputs() {
        let mut handle = std::ptr::null_mut();
        assert_eq!(unsafe { sp_session_create(2, 2, 0, &mut handle) }, 0);
        let original = handle;
        for (selection, policy, bound) in [
            (1, 0, 1),
            (1, 1, 0),
            (1, 2, 1),
            (1, u32::MAX, 0),
            (0, 1, 8),
            (3, 1, 8),
            (u32::MAX, 0, 0),
        ] {
            assert_ne!(
                unsafe {
                    sp_session_create_with_execution(5, 4, 0, selection, policy, bound, &mut handle)
                },
                0
            );
            assert_eq!(handle, original);
        }
        assert_ne!(
            unsafe { sp_session_create_with_execution(5, 5, 0, 4, 1, 8, &mut handle) },
            0
        );
        assert_eq!(handle, original);
        if !cfg!(feature = "native-kernels") {
            for selection in [1, 2, 4] {
                assert_ne!(
                    unsafe {
                        sp_session_create_with_execution(5, 4, 0, selection, 1, 8, &mut handle)
                    },
                    0
                );
                assert_eq!(handle, original);
            }
        }
        assert_ne!(
            unsafe { sp_session_create_with_execution(2, 2, 0, 0, 0, 0, std::ptr::null_mut()) },
            0
        );
        let mut slots = [0usize; 3];
        let misaligned = unsafe {
            slots
                .as_mut_ptr()
                .cast::<u8>()
                .add(1)
                .cast::<*mut Session>()
        };
        assert_ne!(
            unsafe { sp_session_create_with_execution(2, 2, 0, 0, 0, 0, misaligned) },
            0
        );
        assert_eq!(slots, [0; 3]);
        let mut output = [99u64; 3];
        for length in [0, 1, 3] {
            assert_ne!(
                unsafe { sp_session_native_execution_info(handle, output.as_mut_ptr(), length) },
                0
            );
            assert_eq!(output, [99; 3]);
        }
        assert_ne!(
            unsafe { sp_session_native_execution_info(handle, std::ptr::null_mut(), 2) },
            0
        );
        assert_ne!(
            unsafe {
                sp_session_native_execution_info(std::ptr::null_mut(), output.as_mut_ptr(), 2)
            },
            0
        );
        let misaligned = unsafe { output.as_mut_ptr().cast::<u8>().add(1).cast::<u64>() };
        assert_ne!(
            unsafe { sp_session_native_execution_info(handle, misaligned, 2) },
            0
        );
        assert_eq!(output, [99; 3]);
        assert_eq!(
            unsafe { sp_session_native_execution_info(handle, output.as_mut_ptr(), 2) },
            0
        );
        assert_eq!(output, [0, 0, 99]);
        assert_eq!(unsafe { sp_session_destroy(handle) }, 0);
    }
    #[test]
    fn native_execution_old_constructors_keep_whole_and_fixed_reports_survive_reset() {
        for selection in 0..=4 {
            if matches!(selection, 1 | 2 | 4) && !cfg!(feature = "native-kernels") {
                continue;
            }
            let mut old = std::ptr::null_mut();
            assert_eq!(
                unsafe { sp_session_create_with_kernels(5, 4, 0, selection, &mut old) },
                0
            );
            let mut old_generic = [99; 10];
            let mut old_conditional = [99; 4];
            let mut old_execution = [99; 2];
            assert_eq!(
                unsafe { sp_session_kernel_info(old, old_generic.as_mut_ptr(), 10) },
                0
            );
            assert_eq!(
                unsafe { sp_session_rank_correction_info(old, old_conditional.as_mut_ptr(), 4) },
                0
            );
            assert_eq!(
                unsafe { sp_session_native_execution_info(old, old_execution.as_mut_ptr(), 2) },
                0
            );
            assert_eq!(old_execution, [0, 0]);
            for bound in [0, 1, 7, 1024] {
                if bound != 0 && !matches!(selection, 1 | 2 | 4) {
                    continue;
                }
                let mut current = std::ptr::null_mut();
                let policy = u32::from(bound != 0);
                assert_eq!(
                    unsafe {
                        sp_session_create_with_execution(
                            5,
                            4,
                            0,
                            selection,
                            policy,
                            bound,
                            &mut current,
                        )
                    },
                    0
                );
                let mut generic = [99; 10];
                let mut conditional = [99; 4];
                let mut execution = [99; 2];
                assert_eq!(
                    unsafe { sp_session_kernel_info(current, generic.as_mut_ptr(), 10) },
                    0
                );
                assert_eq!(
                    unsafe {
                        sp_session_rank_correction_info(current, conditional.as_mut_ptr(), 4)
                    },
                    0
                );
                assert_eq!(generic, old_generic);
                assert_eq!(conditional, old_conditional);
                for rows in [0, 3, 1] {
                    unsafe { session(current).unwrap() }
                        .amplitude
                        .reset_batch(rows);
                    assert_eq!(
                        unsafe {
                            sp_session_native_execution_info(current, execution.as_mut_ptr(), 2)
                        },
                        0
                    );
                    assert_eq!(execution, [u64::from(policy), bound as u64]);
                }
                assert_eq!(unsafe { sp_session_destroy(current) }, 0);
            }
            assert_eq!(unsafe { sp_session_destroy(old) }, 0);
        }
    }
    #[test]
    fn compatibility_selection_and_transactional_creation() {
        let mut handle = std::ptr::null_mut();
        assert_eq!(unsafe { sp_session_create(2, 2, 0, &mut handle) }, 0);
        let original = handle;
        let mut info = [99; 10];
        assert_eq!(
            unsafe { sp_session_kernel_info(handle, info.as_mut_ptr(), 10) },
            0
        );
        assert_eq!(info[0], 0);
        assert!(info[2..].iter().all(|v| *v == 0));
        assert_eq!(info[1], u64::from(cfg!(feature = "native-kernels")));
        for selection in [3, 4, 5, u32::MAX] {
            assert_ne!(
                unsafe { sp_session_create_with_kernels(2, 2, 0, selection, &mut handle) },
                0
            );
            assert_eq!(handle, original);
        }
        let before = info;
        // Reject the previous seven-family buffer without a partial write.
        assert_ne!(
            unsafe { sp_session_kernel_info(handle, info.as_mut_ptr(), 9) },
            0
        );
        assert_eq!(info, before);
        unsafe { session(handle).unwrap() }.amplitude.reset_batch(3);
        assert_eq!(
            unsafe { sp_session_kernel_info(handle, info.as_mut_ptr(), 10) },
            0
        );
        assert_eq!(info, before);
        assert_eq!(unsafe { sp_session_destroy(handle) }, 0);
    }
    #[test]
    fn explicit_catalog_coverage_or_feature_off_refusal() {
        for selection in [1, 2] {
            let mut handle = std::ptr::null_mut();
            let status = unsafe { sp_session_create_with_kernels(5, 4, 0, selection, &mut handle) };
            if cfg!(feature = "native-kernels") {
                assert_eq!(status, 0);
                let mut info = [99; 10];
                assert_eq!(
                    unsafe { sp_session_kernel_info(handle, info.as_mut_ptr(), 10) },
                    0
                );
                assert_eq!(
                    info,
                    [
                        u64::from(selection),
                        1,
                        1,
                        1,
                        u64::from(selection == 2),
                        u64::from(selection == 2),
                        u64::from(selection == 2),
                        0,
                        0,
                        0
                    ]
                );
                assert_eq!(unsafe { sp_session_destroy(handle) }, 0);
            } else {
                assert_ne!(status, 0);
                assert!(handle.is_null());
            }
        }
    }
    #[test]
    fn rank_corrections_report_separately_and_preserve_old_coverage() {
        for selection in [3, 4] {
            let mut handle = std::ptr::null_mut();
            let status = unsafe { sp_session_create_with_kernels(5, 4, 0, selection, &mut handle) };
            if selection == 4 && !cfg!(feature = "native-kernels") {
                assert_ne!(status, 0);
                assert!(handle.is_null());
                continue;
            }
            assert_eq!(status, 0);
            let mut generic = [99; 10];
            let mut alternate = [99; 4];
            assert_eq!(
                unsafe { sp_session_kernel_info(handle, generic.as_mut_ptr(), 10) },
                0
            );
            assert_eq!(
                unsafe { sp_session_rank_correction_info(handle, alternate.as_mut_ptr(), 4) },
                0
            );
            assert_eq!(generic[0], u64::from(selection));
            assert_eq!(generic[1], u64::from(cfg!(feature = "native-kernels")));
            assert_eq!(&generic[2..7], &[u64::from(selection == 4); 5]);
            assert_eq!(&generic[7..], &[0; 3]);
            assert_eq!(alternate, [5, 4, 1, u64::from(selection == 4)]);
            let before = alternate;
            assert_ne!(
                unsafe { sp_session_rank_correction_info(handle, alternate.as_mut_ptr(), 3) },
                0
            );
            assert_eq!(alternate, before);
            let original = handle;
            for (n, rank) in [(2, 2), (5, 5), (6, 4)] {
                assert_ne!(
                    unsafe { sp_session_create_with_kernels(n, rank, 0, selection, &mut handle) },
                    0
                );
                assert_eq!(handle, original);
            }
            unsafe { session(handle).unwrap() }.amplitude.reset_batch(3);
            assert_eq!(
                unsafe { sp_session_rank_correction_info(handle, alternate.as_mut_ptr(), 4) },
                0
            );
            assert_eq!(alternate, before);
            assert_eq!(unsafe { sp_session_destroy(handle) }, 0);
        }
    }
}
