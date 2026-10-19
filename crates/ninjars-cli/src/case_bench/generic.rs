//! `case-bench --engines generic`: the same per-point breakdown as the fast
//! engine for the generic engine (`SymbolicaAmplitudeKernels`) in binary64 with
//! oneloopmaster's native masters, the configuration `case-check --engines
//! generic --precision f64` checks.
//!
//! The numerator is `case-check`'s [`TensorGenericNumerator`], which expands
//! the dense tensor numerically. Per batch width:
//! - `reduction`: recorded numerator outputs and master values replayed at
//!   O(1) cost (pure reduction);
//! - `reduction_and_numerator`: the live numerator with replayed masters (the
//!   difference is the numerator time);
//! - `masters`: the recorded master requests replayed through
//!   `OneLoopNativeMasters`, without reduction;
//! - `total`: numerator, reduction and native masters together.
//!
//! Inputs (complex shifts, masses, scales, the S-matrix and Ninja's auxiliary
//! box sampling scale) are built once per width, outside the timed region,
//! exactly as `case-check`'s generic engine builds them.

use std::{hint::black_box, time::Duration};

use ninjars::{
    AmplitudeInput, Complex, EvaluationStatus, ExpansionKind, ExpansionRequest, FourMomentum,
    Laurent, MasterIntegral, MasterProvider, MasterRequest, MasterStatus, Numerator,
    NumeratorRequest,
};
use ninjars_symbolica::{
    OneLoopNativeMasters, SymbolicaAmplitudeKernels, SymbolicaCuts, SymbolicaHighCuts,
};
use serde_json::{Value, json};

use super::{Args, Case, Lane, Result};
use crate::{case_check::harness::TensorGenericNumerator, numerator_bench::measure};

type Cf = Complex<f64>;

/// Records every numerator output in call order.
struct RecordingNumerator<'a, N> {
    inner: &'a mut N,
    outputs: Vec<Vec<Cf>>,
}

impl<N: Numerator<Cf>> Numerator<Cf> for RecordingNumerator<'_, N> {
    fn rank(&self) -> usize {
        self.inner.rank()
    }

    fn evaluate_batch(
        &mut self,
        requests: &[NumeratorRequest<'_, Cf>],
        output: &mut [Cf],
    ) -> ninjars::Result<()> {
        self.inner.evaluate_batch(requests, output)?;
        self.outputs.push(output.to_vec());
        Ok(())
    }

    fn expansion_batch(
        &mut self,
        kind: &ExpansionKind,
        requests: &[ExpansionRequest<'_, Cf>],
        output: &mut [Cf],
    ) -> ninjars::Result<()> {
        self.inner.expansion_batch(kind, requests, output)?;
        self.outputs.push(output.to_vec());
        Ok(())
    }
}

/// Replays recorded numerator outputs (one copy per call).
struct ReplayNumerator {
    rank: usize,
    outputs: Vec<Vec<Cf>>,
    next: usize,
}

impl ReplayNumerator {
    fn take(&mut self, output: &mut [Cf]) -> ninjars::Result<()> {
        let recorded = self
            .outputs
            .get(self.next)
            .ok_or_else(|| ninjars::Error::Unsupported("numerator replay exhausted".to_string()))?;
        if recorded.len() != output.len() {
            return Err(ninjars::Error::Unsupported(
                "numerator replay out of order".to_string(),
            ));
        }
        output.clone_from_slice(recorded);
        self.next += 1;
        Ok(())
    }
}

impl Numerator<Cf> for ReplayNumerator {
    fn rank(&self) -> usize {
        self.rank
    }

    fn evaluate_batch(
        &mut self,
        _requests: &[NumeratorRequest<'_, Cf>],
        output: &mut [Cf],
    ) -> ninjars::Result<()> {
        self.take(output)
    }

    fn expansion_batch(
        &mut self,
        _kind: &ExpansionKind,
        _requests: &[ExpansionRequest<'_, Cf>],
        output: &mut [Cf],
    ) -> ninjars::Result<()> {
        self.take(output)
    }
}

/// One recorded master request with its outputs and row statuses.
struct MasterCall {
    integral: MasterIntegral,
    invariants: Vec<Cf>,
    masses: Vec<Cf>,
    scales: Vec<Cf>,
    output: Vec<Laurent<Cf>>,
    statuses: Vec<MasterStatus>,
    batched: bool,
}

/// Records every master request (arguments, outputs, statuses) in call order.
struct RecordingMasters<'a> {
    inner: &'a mut OneLoopNativeMasters,
    calls: Vec<MasterCall>,
}

impl MasterProvider<Cf> for RecordingMasters<'_> {
    fn evaluate(
        &mut self,
        request: MasterRequest<'_, Cf>,
        output: &mut [Laurent<Cf>],
    ) -> ninjars::Result<()> {
        self.inner.evaluate(request, output)?;
        self.calls.push(MasterCall {
            integral: request.integral,
            invariants: request.invariants.to_vec(),
            masses: request.masses_squared.to_vec(),
            scales: request.scales_squared.to_vec(),
            output: output.to_vec(),
            statuses: Vec::new(),
            batched: false,
        });
        Ok(())
    }

    fn evaluate_batch(
        &mut self,
        request: MasterRequest<'_, Cf>,
        output: &mut [Laurent<Cf>],
        statuses: &mut [MasterStatus],
    ) -> ninjars::Result<()> {
        self.inner.evaluate_batch(request, output, statuses)?;
        self.calls.push(MasterCall {
            integral: request.integral,
            invariants: request.invariants.to_vec(),
            masses: request.masses_squared.to_vec(),
            scales: request.scales_squared.to_vec(),
            output: output.to_vec(),
            statuses: statuses.to_vec(),
            batched: true,
        });
        Ok(())
    }
}

/// Replays recorded master outputs and statuses (one copy per call).
struct ReplayMasters<'a> {
    calls: &'a [MasterCall],
    next: usize,
}

impl ReplayMasters<'_> {
    fn take(&mut self, output: &mut [Laurent<Cf>]) -> ninjars::Result<&MasterCall> {
        let call = self
            .calls
            .get(self.next)
            .ok_or_else(|| ninjars::Error::Unsupported("master replay exhausted".to_string()))?;
        if call.output.len() != output.len() {
            return Err(ninjars::Error::Unsupported(
                "master replay out of order".to_string(),
            ));
        }
        output.clone_from_slice(&call.output);
        self.next += 1;
        Ok(call)
    }
}

impl MasterProvider<Cf> for ReplayMasters<'_> {
    fn evaluate(
        &mut self,
        _request: MasterRequest<'_, Cf>,
        output: &mut [Laurent<Cf>],
    ) -> ninjars::Result<()> {
        self.take(output).map(|_| ())
    }

    fn evaluate_batch(
        &mut self,
        _request: MasterRequest<'_, Cf>,
        output: &mut [Laurent<Cf>],
        statuses: &mut [MasterStatus],
    ) -> ninjars::Result<()> {
        let call = self.take(output)?;
        statuses.copy_from_slice(&call.statuses);
        Ok(())
    }
}

fn real(x: f64) -> Cf {
    Cf::new(x, 0.)
}

fn minkowski(a: &[f64; 4], b: &[f64; 4]) -> f64 {
    let d = [a[0] - b[0], a[1] - b[1], a[2] - b[2], a[3] - b[3]];
    d[0] * d[0] - d[1] * d[1] - d[2] * d[2] - d[3] * d[3]
}

/// Batched inputs of `lanes` (case-check's generic-engine conventions).
struct Inputs {
    shifts: Vec<FourMomentum<Cf>>,
    masses: Vec<Cf>,
    scales: Vec<Cf>,
    sampling: Vec<Cf>,
    s_matrix: Option<Vec<Cf>>,
}

impl Inputs {
    fn view(&self) -> AmplitudeInput<'_, Cf> {
        AmplitudeInput {
            shifts: &self.shifts,
            masses_squared: &self.masses,
            scales_squared: &self.scales,
            sampling_scales: &self.sampling,
            s_matrix: self.s_matrix.as_deref(),
        }
    }
}

fn inputs(n: usize, lanes: &[&Lane]) -> Inputs {
    let shifts = lanes
        .iter()
        .flat_map(|lane| lane.shifts.iter().map(|v| FourMomentum::from(v.map(real))))
        .collect();
    let masses = lanes
        .iter()
        .flat_map(|lane| lane.masses.iter().map(|m| Cf::new(m.re, m.im)))
        .collect();
    let scales = lanes.iter().map(|lane| real(lane.mu2)).collect();
    // Ninja's auxiliary box mu^2 sampling scale.
    let sampling = lanes
        .iter()
        .map(|lane| {
            let invariants = lane.s_matrix.iter().flatten().map(|x| x.abs());
            let masses = lane.masses.iter().map(|m| m.re.abs());
            real(invariants.chain(masses).fold(1., f64::max))
        })
        .collect();
    // Supplied S-matrices; missing ones (when another lane has one) are the
    // invariants of the shifts.
    let s_matrix = lanes.iter().any(|lane| lane.s_matrix.is_some()).then(|| {
        let mut matrix = Vec::with_capacity(lanes.len() * n * n);
        for lane in lanes {
            match &lane.s_matrix {
                Some(s) => matrix.extend(s.iter().map(|&x| real(x))),
                None => {
                    for i in 0..n {
                        for j in 0..n {
                            matrix.push(real(minkowski(&lane.shifts[i], &lane.shifts[j])));
                        }
                    }
                }
            }
        }
        matrix
    });
    Inputs {
        shifts,
        masses,
        scales,
        sampling,
        s_matrix,
    }
}

/// Time every width of one case with the generic engine.
pub(super) fn time_case(
    args: &Args,
    case: &Case,
    masters: &mut OneLoopNativeMasters,
    min_block: Duration,
) -> Result<Vec<Value>> {
    let (n, rank) = (case.n, case.rank);
    let mut low = SymbolicaCuts::new(0_f64)?;
    let mut high = SymbolicaHighCuts::new(0_f64)?;
    let mut amplitude = SymbolicaAmplitudeKernels::new(n, rank, 0_f64)?.into_amplitude(1)?;
    let mut widths = Vec::new();
    for &width in &args.widths {
        if width == 0 {
            return Err("timing widths must be positive".into());
        }
        let lanes: Vec<&Lane> = (0..width)
            .map(|i| &case.lanes[i % case.lanes.len()])
            .collect();
        let data = inputs(n, &lanes);
        let mut numerator = TensorGenericNumerator::new(rank, case.mu_powers, &0_f64)?;
        numerator.set_coefficients(
            lanes
                .iter()
                .flat_map(|lane| lane.tensor.iter().map(|z| Cf::new(z.re, z.im)))
                .collect(),
            width,
        )?;

        let mut recording_numerator = RecordingNumerator {
            inner: &mut numerator,
            outputs: Vec::new(),
        };
        let mut recorder = RecordingMasters {
            inner: masters,
            calls: Vec::new(),
        };
        amplitude.reset_batch(width);
        amplitude.evaluate_batch_with_inputs(
            data.view(),
            &mut recording_numerator,
            &mut low,
            &mut high,
            &mut recorder,
        )?;
        let failed = amplitude
            .statuses()
            .iter()
            .filter(|s| **s != EvaluationStatus::Success)
            .count();
        let mut failure_kinds = std::collections::BTreeMap::<String, usize>::new();
        for status in amplitude
            .statuses()
            .iter()
            .filter(|s| **s != EvaluationStatus::Success)
        {
            *failure_kinds.entry(format!("{status:?}")).or_default() += 1;
        }
        let live = amplitude.results().to_vec();
        let recorded_outputs = recording_numerator.outputs;
        let calls = recorder.calls;
        let mut replay_numerator = ReplayNumerator {
            rank,
            outputs: recorded_outputs,
            next: 0,
        };

        let reduction = measure(args.blocks, min_block, width, || {
            replay_numerator.next = 0;
            let mut replay = ReplayMasters {
                calls: &calls,
                next: 0,
            };
            amplitude.reset_batch(width);
            amplitude.evaluate_batch_with_inputs(
                data.view(),
                &mut replay_numerator,
                &mut low,
                &mut high,
                &mut replay,
            )?;
            black_box(amplitude.results());
            Ok(())
        })?;
        // The replays must reproduce the live evaluation bit for bit.
        let mut replay_bitwise = amplitude.results() == live.as_slice();
        let reduction_and_numerator = measure(args.blocks, min_block, width, || {
            let mut replay = ReplayMasters {
                calls: &calls,
                next: 0,
            };
            amplitude.reset_batch(width);
            amplitude.evaluate_batch_with_inputs(
                data.view(),
                &mut numerator,
                &mut low,
                &mut high,
                &mut replay,
            )?;
            black_box(amplitude.results());
            Ok(())
        })?;
        replay_bitwise &= amplitude.results() == live.as_slice();
        let mut outputs: Vec<Vec<Laurent<Cf>>> =
            calls.iter().map(|call| call.output.clone()).collect();
        let mut statuses: Vec<Vec<MasterStatus>> = calls
            .iter()
            .map(|call| vec![MasterStatus::Success; call.scales.len()])
            .collect();
        let masters_only = measure(args.blocks, min_block, width, || {
            for ((call, out), status) in calls
                .iter()
                .zip(outputs.iter_mut())
                .zip(statuses.iter_mut())
            {
                let request = MasterRequest {
                    integral: call.integral,
                    invariants: &call.invariants,
                    masses_squared: &call.masses,
                    scales_squared: &call.scales,
                };
                if call.batched {
                    masters.evaluate_batch(request, out, status)?;
                } else {
                    masters.evaluate(request, out)?;
                }
            }
            black_box(&outputs);
            Ok(())
        })?;
        let total = measure(args.blocks, min_block, width, || {
            amplitude.reset_batch(width);
            amplitude.evaluate_batch_with_inputs(
                data.view(),
                &mut numerator,
                &mut low,
                &mut high,
                masters,
            )?;
            black_box(amplitude.results());
            Ok(())
        })?;
        let median = |v: &Value| v["median_ns_per_point"].as_f64().unwrap_or(f64::NAN);
        let numerator_ns = median(&reduction_and_numerator) - median(&reduction);
        let sum = numerator_ns + median(&reduction) + median(&masters_only);
        let mut families = std::collections::BTreeMap::<String, usize>::new();
        for call in &calls {
            *families.entry(format!("{:?}", call.integral)).or_default() += call.scales.len();
        }
        widths.push(json!({
            "width": width,
            "failed_lanes": failed,
            "failure_kinds": failure_kinds,
            "replay_bitwise": replay_bitwise,
            "reduction": reduction,
            "reduction_and_numerator": reduction_and_numerator,
            "masters": masters_only,
            "total": total,
            "summary_ns_per_point": {
                "numerator": numerator_ns,
                "reduction": median(&reduction),
                "masters": median(&masters_only),
                "total": median(&total),
                "sum_of_components": sum,
                "sum_over_total": sum / median(&total),
            },
            "master_rows_per_batch": families,
        }));
    }
    Ok(widths)
}
