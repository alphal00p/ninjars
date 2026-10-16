//! Shared numerical physical-input boundary. Static graph construction is setup;
//! every prepare call computes wavefunctions, dependent model coefficients and
//! routed propagators from raw momenta, helicities and free model parameters.
use super::{CallScratch, Pair, Provider, Result, dimensions, guard, read, write};
use feynkit_graph::ExternalState;
use feynkit_kinematics::{FourMomentum, Helicity, WavefunctionKind};
use ninjars_graph::{
    CoefficientEvaluator, Kinematics, Model, NumeratorMode, PolynomialCoefficientPlan,
    SymJitCoefficientEvaluator, prepare_dot, symbols,
};
use ninjars_symbolica::PolynomialNumerator;
use std::{collections::BTreeMap, fs};
use symbolica::{
    atom::{Atom, AtomCore},
    domains::{
        atom::AtomField,
        float::{Complex, Float, RealLike, SingleFloat},
        rational::Rational,
    },
    evaluate::BatchExpressionEvaluator,
    id::{Pattern, Replacement},
    parser::ParseSettings,
    symbol,
};
type C = Complex<f64>;
const MODEL_NAMES: [&str; 7] = ["MT", "ymt", "MH", "Gf", "aS", "aEWM1", "MZ"];
fn atom(text: &str) -> Atom {
    Atom::parse(text, "shared_physical", ParseSettings::default()).unwrap()
}
fn error(e: impl std::fmt::Display) -> String {
    e.to_string()
}
fn output<T>(ptr: *mut T, len: usize) -> Result<()> {
    if len > 0
        && (ptr.is_null()
            || !ptr.is_aligned()
            || len > isize::MAX as usize / std::mem::size_of::<T>())
    {
        Err("invalid physical output pointer/length".into())
    } else {
        Ok(())
    }
}
enum Coefficients {
    Interpreter(CoefficientEvaluator<f64>),
    Jit(SymJitCoefficientEvaluator),
}
impl Coefficients {
    fn evaluate(&mut self, input: &[C], out: &mut [C], rows: usize) -> Result<()> {
        match self {
            Self::Interpreter(e) => e.evaluate_batch(input, out, rows).map_err(error),
            Self::Jit(e) => e.evaluate_batch(input, out, rows).map_err(error),
        }
    }
}
pub struct Physical {
    provider: Box<Provider>,
    plan: PolynomialCoefficientPlan,
    coefficients: Coefficients,
    propagation: BatchExpressionEvaluator<C>,
    sources: Vec<usize>,
    waves: Vec<(usize, WavefunctionKind)>,
    scalar_legs: Vec<usize>,
    input: Vec<C>,
    heads: Vec<C>,
    propagators: Vec<C>,
    packed: Vec<Pair>,
    bound_input: Vec<C>,
    bound_heads: Vec<C>,
    rows: usize,
    calls: u64,
    total_rows: u64,
    wave_calls: u64,
}
impl Physical {
    fn new(path: &str, mode: u32) -> Result<Self> {
        let model = Model::standard_model();
        let graph = prepare_dot(
            model.clone(),
            &fs::read_to_string(path).map_err(error)?,
            &Kinematics::new(),
            NumeratorMode::Explicit,
        )
        .map_err(error)?
        .with_color_dimension_invariants()
        .map_err(error)?;
        if graph.quadratics.len() != 5 {
            return Err("this physical probe requires the verified ggHHH pentagon".into());
        }
        let q = ["q0", "q1", "q2", "q3", "mu2"].map(atom);
        let loop_components =
            FourMomentum::from([q[0].clone(), q[1].clone(), q[2].clone(), q[3].clone()]);
        let model_parameters =
            MODEL_NAMES.map(|name| atom(&format!("shared_physical::model_{name}")));
        let mut overrides: BTreeMap<_, _> = MODEL_NAMES
            .iter()
            .zip(&model_parameters)
            .map(|(name, value)| (symbol!(format!("UFO::{name}")), value.clone()))
            .collect();
        overrides.insert(symbol!("UFO::WT"), Atom::Zero);
        overrides.insert(symbol!("UFO::WH"), Atom::Zero);
        let replacements: Vec<_> = model
            .scalar_bindings(None, &overrides)
            .map_err(error)?
            .into_iter()
            .map(|(key, value)| {
                Replacement::new(Pattern::Literal(Atom::var(key)), Pattern::Literal(value))
            })
            .collect();
        let mut carriers = Vec::new();
        let mut sources = Vec::new();
        let mut waves = Vec::new();
        let mut scalar_legs = Vec::new();
        let mut seen = [false; 5];
        for (position, edge) in graph
            .diagram
            .loop_momentum_basis()
            .external_edges
            .iter()
            .enumerate()
        {
            let data = graph
                .diagram
                .edges()
                .find(|(id, _, _)| id == edge)
                .ok_or("missing external edge")?
                .2;
            let external = data.external.as_ref().ok_or("missing external metadata")?;
            let leg = external.index;
            if leg >= 5 || seen[leg] {
                return Err("external indices must be unique0..4".into());
            }
            seen[leg] = true;
            let pdg = model.particle_by_id(data.particle).map_err(error)?.pdg_code;
            if (leg < 2 && (pdg != 21 || external.state != ExternalState::Incoming))
                || (leg >= 2 && (pdg != 25 || external.state != ExternalState::Outgoing))
            {
                return Err("expected incoming gluons0,1 and outgoing Higgs2,3,4".into());
            }
            let momentum = symbols::external_momentum().call(position);
            if graph.family.external_momenta().contains(&momentum) {
                carriers.push(momentum);
                sources.push(leg);
            }
            if pdg == 25 {
                scalar_legs.push(leg);
                continue;
            }
            let matches: Vec<_> = [
                (symbols::epsilon(), WavefunctionKind::Epsilon),
                (symbols::epsilonbar(), WavefunctionKind::EpsilonBar),
            ]
            .into_iter()
            .filter(|(head, _)| {
                graph
                    .numerator
                    .pattern_match(
                        &head
                            .call((edge.0, atom("shared_physical::rep_")))
                            .to_pattern(),
                        None,
                        None,
                    )
                    .next()
                    .is_some()
            })
            .collect();
            if matches.len() != 1 {
                return Err("one vector wavefunction carrier required per gluon".into());
            }
            waves.push((leg, matches[0].1, matches[0].0.call(edge.0)));
        }
        if seen.iter().any(|v| !*v) {
            return Err("missing physical external leg".into());
        }
        carriers.extend(waves.iter().map(|(_, _, carrier)| carrier.clone()));
        let mut parameters: Vec<_> = (0..4 * carriers.len())
            .map(|i| atom(&format!("shared_physical::p{i}")))
            .collect();
        let symbolic: BTreeMap<_, _> = carriers
            .into_iter()
            .zip(
                parameters
                    .as_chunks::<4>()
                    .0
                    .iter()
                    .cloned()
                    .map(FourMomentum::from),
            )
            .collect();
        parameters.extend(model_parameters);
        let kinematics = graph
            .component_kinematics(&loop_components, &symbolic)
            .map_err(error)?;
        let expression = kinematics
            .apply(&(&graph.numerator * &graph.overall_factor))
            .replace_multiple(&replacements)
            .expand();
        let raw = PolynomialCoefficientPlan::new(expression, q.clone(), parameters.clone())
            .map_err(error)?;
        // Public factor/map_coeff/flatten, poly/factor implementation/examples,
        // and FeynKit usages were audited in the coefficient-factor experiment.
        // This exact composition is temporary until the optional public method
        // lands; no parallel polynomial or algebra implementation is introduced.
        let factored = raw
            .polynomial()
            .map_coeff(
                |coefficient| {
                    let factored = coefficient.factor();
                    assert!((&factored - coefficient).expand().is_zero());
                    factored
                },
                AtomField {
                    statistical_zero_test: false,
                    ..AtomField::new()
                },
            )
            .flatten(false);
        let plan = PolynomialCoefficientPlan::new(factored, q.clone(), parameters.clone())
            .map_err(error)?;
        if plan.rank() != 4 {
            return Err(format!(
                "verified top pentagon must have weighted degree4, got{}",
                plan.rank()
            ));
        }
        let coefficients = match mode {
            0 => Coefficients::Interpreter(plan.compile(0.).map_err(error)?),
            1 => Coefficients::Jit(plan.compile_symjit_o2().map_err(error)?),
            _ => return Err("coefficient mode must be interpreter0 or SymJIT O2=1".into()),
        };
        let shifts = graph.shifts(&symbolic).map_err(error)?;
        let mut propagation_heads: Vec<_> = shifts
            .iter()
            .flat_map(|p| p.components().map(Clone::clone))
            .chain(graph.squared_masses())
            .collect();
        for left in &shifts {
            for right in &shifts {
                let delta = left.clone() - right.clone();
                propagation_heads.push(delta.dot(&delta));
            }
        }
        for value in &mut propagation_heads {
            *value = value.replace_multiple(&replacements);
        }
        let propagation = Atom::evaluator_multiple(&propagation_heads, &parameters)
            .cores(1)
            .build()
            .map_err(error)?
            .map_coeff_with_prec(
                &|v| C::new(0_f64.from_rational(&v.re), 0_f64.from_rational(&v.im)),
                53,
            )
            .into_batch_evaluator();
        let provider = Box::new(Provider {
            numerator: PolynomialNumerator::new(
                plan.template().clone(),
                q,
                plan.coefficient_symbols().to_vec(),
                0.,
            )
            .map_err(error)?,
            n: 5,
            parameters: plan.coefficient_symbols().len(),
            scratch: CallScratch::default(),
        });
        Ok(Self {
            provider,
            plan,
            coefficients,
            propagation,
            sources,
            waves: waves
                .into_iter()
                .map(|(leg, kind, _)| (leg, kind))
                .collect(),
            scalar_legs,
            input: Vec::new(),
            heads: Vec::new(),
            propagators: Vec::new(),
            packed: Vec::new(),
            bound_input: Vec::new(),
            bound_heads: Vec::new(),
            rows: 0,
            calls: 0,
            total_rows: 0,
            wave_calls: 0,
        })
    }
    fn prepare(
        &mut self,
        momenta: &[f64],
        helicities: &[i8],
        model: &[f64],
        scales: &[f64],
        rows: usize,
    ) -> Result<()> {
        dimensions(momenta.len(), rows, 20, "physical momenta")?;
        dimensions(helicities.len(), rows, 5, "physical helicities")?;
        dimensions(model.len(), rows, 7, "physical model")?;
        dimensions(scales.len(), rows, 1, "physical scales")?;
        if momenta
            .iter()
            .chain(model)
            .chain(scales)
            .any(|x| !x.is_finite())
        {
            return Err("nonfinite physical input".into());
        }
        if scales.iter().chain(model).any(|x| *x <= 0.) {
            return Err("this zero-width probe requires positive model inputs/scales".into());
        }
        self.input.clear();
        for row in 0..rows {
            let p: Vec<_> = momenta[row * 20..(row + 1) * 20]
                .as_chunks::<4>()
                .0
                .iter()
                .copied()
                .map(FourMomentum::from)
                .collect();
            if p.iter().any(|p| p.temporal.value <= 0.) {
                return Err(format!("nonpositive physical energy at row{row}"));
            }
            if helicities[row * 5..row * 5 + 2]
                .iter()
                .any(|h| ![-1, 1].contains(h))
            {
                return Err(format!("massless vector helicity must be +/-1 at row{row}"));
            }
            let conservation = p[0] + p[1] - p[2] - p[3] - p[4];
            let energy_scale = p
                .iter()
                .map(|p| p.temporal.value.abs())
                .fold(1_f64, f64::max);
            if conservation
                .components()
                .iter()
                .any(|x| x.abs() > 1e-10 * energy_scale)
            {
                return Err(format!("physical momentum conservation failed at row{row}"));
            }
            for (leg, momentum) in p.iter().enumerate() {
                let mass2 = if leg < 2 {
                    0.
                } else {
                    model[7 * row + 2].powi(2)
                };
                if (momentum.mass_squared() - mass2).abs() > 1e-10 * energy_scale.powi(2) {
                    return Err(format!(
                        "physical on-shell check failed at row{row} leg{leg}"
                    ));
                }
            }
            for &leg in &self.sources {
                self.input
                    .extend(p[leg].components().map(|x| C::new(*x, 0.)));
            }
            for &(leg, kind) in &self.waves {
                self.input.extend_from_slice(
                    p[leg]
                        .wavefunction(
                            kind,
                            Helicity::try_from(helicities[5 * row + leg]).map_err(error)?,
                        )
                        .map_err(error)?
                        .components(),
                );
            }
            for &leg in &self.scalar_legs {
                let value = p[leg]
                    .wavefunction(
                        WavefunctionKind::Scalar,
                        Helicity::try_from(helicities[5 * row + leg]).map_err(error)?,
                    )
                    .map_err(error)?;
                if value.components() != [C::new(1., 0.)] {
                    return Err("unexpected scalar wavefunction convention".into());
                }
            }
            self.input
                .extend(model[row * 7..(row + 1) * 7].iter().map(|&x| C::new(x, 0.)));
        }
        self.heads
            .resize(rows * self.provider.parameters, C::new(0., 0.));
        self.coefficients
            .evaluate(&self.input, &mut self.heads, rows)?;
        self.propagators.resize(rows * 50, C::new(0., 0.));
        self.propagation
            .evaluate_batch(&self.input, &mut self.propagators, rows)
            .map_err(error)?;
        if self
            .propagators
            .iter()
            .any(|z| !z.re.is_finite() || !z.im.is_finite() || z.im != 0.)
        {
            return Err("nonfinite/nonreal zero-width propagator preparation".into());
        }
        self.packed.clear();
        for (row, prop) in self.propagators.chunks(50).enumerate() {
            self.packed.extend(prop.iter().copied().map(Pair::from));
            self.packed.push(Pair {
                re: scales[row],
                im: 0.,
            });
        }
        // The public provider binding is the commit point; no caller-visible
        // output/binding changes before every preparation stage succeeds.
        self.provider
            .numerator
            .set_parameter_batch(&self.heads, rows)
            .map_err(error)?;
        std::mem::swap(&mut self.bound_input, &mut self.input);
        std::mem::swap(&mut self.bound_heads, &mut self.heads);
        self.rows = rows;
        self.calls += 1;
        self.total_rows += rows as u64;
        self.wave_calls += (rows * 5) as u64;
        Ok(())
    }
}
unsafe fn physical<'a>(handle: *mut Physical) -> Result<&'a mut Physical> {
    if handle.is_null() || !handle.is_aligned() {
        Err("invalid physical handle".into())
    } else {
        Ok(unsafe { &mut *handle })
    }
}
/// # Safety
/// Valid UTF-8 path bytes and writable output slot. This handle owns its provider.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn sp_physical_create(
    path: *const u8,
    len: usize,
    mode: u32,
    out: *mut *mut Physical,
) -> i32 {
    guard(|| {
        output(out, 1)?;
        let path = std::str::from_utf8(unsafe { read(path, len) }?).map_err(error)?;
        let state = Physical::new(path, mode)?;
        unsafe { out.write(Box::into_raw(Box::new(state))) };
        Ok(())
    })
}
/// # Safety
/// Handle live/exclusive, out writable. Returned borrowed provider must never
/// be passed to sp_destroy and is invalidated by sp_physical_destroy. Neither
/// preparation nor reduction may run concurrently with another use of handle.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn sp_physical_provider(
    handle: *mut Physical,
    out: *mut *mut Provider,
) -> i32 {
    guard(|| {
        output(out, 1)?;
        let state = unsafe { physical(handle) }?;
        unsafe { out.write(&mut *state.provider) };
        Ok(())
    })
}
/// # Safety
/// Handle live/exclusive; eight writable u64 slots.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn sp_physical_info(handle: *mut Physical, out: *mut u64, len: usize) -> i32 {
    guard(|| {
        let p = unsafe { physical(handle) }?;
        unsafe {
            write(
                out,
                len,
                &[
                    5,
                    p.plan.rank() as u64,
                    5,
                    7,
                    p.provider.parameters as u64,
                    51,
                    p.plan.parameters().len() as u64,
                    5,
                ],
            )
        }
    })
}
/// # Safety
/// Handle and all readable/writable arrays are mutually disjoint and valid for
/// their lengths. Output is row-major [20 shift cells,5 masses,25 S-matrix,scale].
/// Inputs are [5 physical positive-energy four-momenta], [5 helicities], model
/// [MT,ymt,MH,Gf,aS,aEWM1,MZ], scales. Whole-batch preparation is transactional.
#[unsafe(no_mangle)]
#[inline(never)]
pub unsafe extern "C" fn sp_physical_prepare(
    handle: *mut Physical,
    rows: usize,
    momenta: *const f64,
    np: usize,
    helicities: *const i8,
    nh: usize,
    model: *const f64,
    nmodel: usize,
    scales: *const f64,
    ns: usize,
    out: *mut Pair,
    no: usize,
) -> i32 {
    guard(|| {
        dimensions(no, rows, 51, "physical output")?;
        output(out, no)?;
        let state = unsafe { physical(handle) }?;
        state.prepare(
            unsafe { read(momenta, np) }?,
            unsafe { read(helicities, nh) }?,
            unsafe { read(model, nmodel) }?,
            unsafe { read(scales, ns) }?,
            rows,
        )?;
        unsafe { write(out, no, &state.packed) }
    })
}
/// # Safety
/// Handle live/exclusive; correctly-sized writable coefficient output.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn sp_physical_coefficients(
    handle: *mut Physical,
    out: *mut Pair,
    len: usize,
) -> i32 {
    guard(|| {
        let s = unsafe { physical(handle) }?;
        let values: Vec<_> = s.bound_heads.iter().copied().map(Pair::from).collect();
        unsafe { write(out, len, &values) }
    })
}
/// # Safety
/// Handle live/exclusive; writable [max scaled error,index,actualre,actualim,
/// expectedre,expectedim] buffer. Audit only, never part of timed preparation.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn sp_physical_float256_audit(
    handle: *mut Physical,
    out: *mut f64,
    len: usize,
) -> i32 {
    guard(|| {
        let s = unsafe { physical(handle) }?;
        let proto = Float::new(256);
        let promote = |x: f64| proto.from_rational(&Rational::try_from(x).unwrap());
        let inputs: Vec<_> = s
            .bound_input
            .iter()
            .map(|z| Complex::new(promote(z.re), promote(z.im)))
            .collect();
        let mut values = vec![Complex::new(Float::new(256), Float::new(256)); s.bound_heads.len()];
        s.plan
            .compile(proto)
            .map_err(error)?
            .evaluate_batch(&inputs, &mut values, s.rows)
            .map_err(error)?;
        let mut worst = [0.; 6];
        for (index, (actual, expected)) in s.bound_heads.iter().zip(values).enumerate() {
            let expected = C::new(expected.re.to_f64(), expected.im.to_f64());
            if !actual.re.is_finite()
                || !actual.im.is_finite()
                || !expected.re.is_finite()
                || !expected.im.is_finite()
            {
                return Err("nonfinite physical coefficient oracle".into());
            }
            let scaled = (actual.re - expected.re).hypot(actual.im - expected.im)
                / 1_f64.max(expected.re.hypot(expected.im));
            if scaled > worst[0] {
                worst = [
                    scaled,
                    index as f64,
                    actual.re,
                    actual.im,
                    expected.re,
                    expected.im,
                ];
            }
        }
        unsafe { write(out, len, &worst) }
    })
}
/// # Safety
/// Handle live/exclusive; three writable stats entries: calls,rows,wavefunctions.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn sp_physical_stats(
    handle: *mut Physical,
    out: *mut u64,
    len: usize,
) -> i32 {
    guard(|| {
        let s = unsafe { physical(handle) }?;
        unsafe { write(out, len, &[s.calls, s.total_rows, s.wave_calls]) }
    })
}
/// # Safety
/// Destroy once with no outstanding borrowed provider or calls. Null is a no-op.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn sp_physical_destroy(handle: *mut Physical) -> i32 {
    guard(|| {
        if !handle.is_null() {
            if !handle.is_aligned() {
                return Err("invalid physical handle".into());
            }
            drop(unsafe { Box::from_raw(handle) });
        }
        Ok(())
    })
}
