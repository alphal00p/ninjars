//! Convenient Symbolica backends for the integrated NinjaRS amplitude.

use crate::{
    SymbolicaBoxKernels, SymbolicaBubbleKernels, SymbolicaDualBoxKernels,
    SymbolicaDualBubbleKernels, SymbolicaDualIntegrationKernels, SymbolicaDualPentagonKernels,
    SymbolicaDualTadpoleKernels, SymbolicaDualTriangleKernels, SymbolicaIntegrationKernels,
    SymbolicaPentagonKernels, SymbolicaTadpoleKernels, SymbolicaTriangleKernels,
    kernel::{CompiledDualKernel, CompiledKernel, ExactKernel},
};
use ninjars::{
    Amplitude, AmplitudeKernels, BoxCoefficientMode, BoxMuMode, BoxReductionOptions, Complex,
    Error, Real, Result, SingleFloat, Topology,
};
use symbolica::{atom::Atom, domains::dual::DualNumber, evaluate::EvaluationDomain, symbol};

fn accumulation(n: usize, rank: usize) -> Result<ExactKernel> {
    let topology = Topology::new(n, rank)?;
    if n > 6 {
        return Err(Error::Unsupported(
            "integrated reduction currently supports at most six propagators".into(),
        ));
    }
    let rows = topology.cuts().len() + 1;
    let parameters: Vec<_> = (0..rows * 5)
        .map(|i| Atom::var(symbol!(format!("ninjars_accumulation::p{i}"))))
        .collect();
    let expressions = (0..5)
        .map(|component| {
            (0..rows).fold(Atom::Zero, |sum, row| {
                sum + &parameters[row * 5 + component]
            })
        })
        .collect();
    Ok(ExactKernel {
        parameters,
        expressions,
    })
}

/// Exact reduction and integration kernels for complex f64, DoubleFloat or Float.
/// Construct once, then call `into_amplitude(lanes)` to allocate the coordinator.
pub struct SymbolicaAmplitudeKernels<T> {
    n: usize,
    rank: usize,
    prototype: T,
    box_options: BoxReductionOptions,
    accumulation: CompiledKernel<T>,
}
impl<T: Real + SingleFloat> SymbolicaAmplitudeKernels<T>
where
    Complex<T>: Real + EvaluationDomain,
{
    pub fn new(propagators: usize, rank: usize, prototype: T) -> Result<Self> {
        let accumulation = accumulation(propagators, rank)?.compile(&prototype)?;
        Ok(Self {
            n: propagators,
            rank,
            prototype,
            box_options: BoxReductionOptions {
                coefficients: BoxCoefficientMode::Integrated,
                mu_extraction: BoxMuMode::Expansion,
            },
            accumulation,
        })
    }
    pub fn into_amplitude(self, lanes: usize) -> Result<Amplitude<Complex<T>, Self>> {
        let zero = Complex::new(self.prototype.zero(), self.prototype.zero());
        Amplitude::new(self, zero, lanes)
    }
    pub fn with_box_options(mut self, options: BoxReductionOptions) -> Self {
        self.box_options = options;
        self
    }
}
impl<T: Real + SingleFloat> AmplitudeKernels<Complex<T>> for SymbolicaAmplitudeKernels<T>
where
    Complex<T>: Real + EvaluationDomain,
{
    type Boxes = SymbolicaBoxKernels<T>;
    type Pentagons = SymbolicaPentagonKernels<T>;
    type Triangles = SymbolicaTriangleKernels<T>;
    type Bubbles = SymbolicaBubbleKernels<T>;
    type Tadpoles = SymbolicaTadpoleKernels<T>;
    type Integration = SymbolicaIntegrationKernels<T>;
    fn propagators(&self) -> usize {
        self.n
    }
    fn rank(&self) -> usize {
        self.rank
    }
    fn box_options(&self) -> BoxReductionOptions {
        self.box_options
    }
    fn box_kernels(&self) -> Result<Self::Boxes> {
        SymbolicaBoxKernels::new(self.n, self.rank, self.box_options, self.prototype.clone())
    }
    fn pentagon_kernels(&self) -> Result<Self::Pentagons> {
        SymbolicaPentagonKernels::new(self.n, self.rank, self.prototype.clone())
    }
    fn triangle_kernels(&self) -> Result<Self::Triangles> {
        SymbolicaTriangleKernels::new(self.n, self.rank, self.prototype.clone())
    }
    fn bubble_kernels(&self) -> Result<Self::Bubbles> {
        SymbolicaBubbleKernels::new(self.n, self.rank, self.prototype.clone())
    }
    fn tadpole_kernels(&self) -> Result<Self::Tadpoles> {
        SymbolicaTadpoleKernels::new(self.n, self.rank, self.prototype.clone())
    }
    fn integration_kernels(&self, cut_size: usize) -> Result<Self::Integration> {
        SymbolicaIntegrationKernels::new(cut_size, self.rank > self.n, self.prototype.clone())
    }
    fn accumulate_batch(
        &mut self,
        input: &[Complex<T>],
        output: &mut [Complex<T>],
        rows: usize,
    ) -> Result<()> {
        self.accumulation.evaluate_batch(input, output, rows)
    }
    fn values_are_finite(&self, values: &[Complex<T>]) -> bool {
        values.iter().all(|v| v.re.is_finite() && v.im.is_finite())
    }
    fn values_equal(&self, left: &Complex<T>, right: &Complex<T>) -> bool {
        left == right
    }
}

/// The same kernels transformed by Symbolica for a static or dynamic Dual shape.
pub struct SymbolicaDualAmplitudeKernels<T, D> {
    n: usize,
    rank: usize,
    prototype: T,
    dual: D,
    box_options: BoxReductionOptions,
    accumulation: CompiledDualKernel<T, D>,
}
impl<T: Real + SingleFloat, D: Real + DualNumber<Complex<T>>> SymbolicaDualAmplitudeKernels<T, D>
where
    Complex<T>: Real + EvaluationDomain,
{
    pub fn new(propagators: usize, rank: usize, prototype: T, dual: &D) -> Result<Self> {
        let accumulation = accumulation(propagators, rank)?.compile_dual(&prototype, dual)?;
        Ok(Self {
            n: propagators,
            rank,
            prototype,
            dual: dual.zero(),
            box_options: BoxReductionOptions {
                coefficients: BoxCoefficientMode::Integrated,
                mu_extraction: BoxMuMode::Expansion,
            },
            accumulation,
        })
    }
    pub fn into_amplitude(self, lanes: usize) -> Result<Amplitude<D, Self>> {
        let zero = self.dual.zero();
        Amplitude::new(self, zero, lanes)
    }
    pub fn with_box_options(mut self, options: BoxReductionOptions) -> Self {
        self.box_options = options;
        self
    }
}
impl<T: Real + SingleFloat, D: Real + DualNumber<Complex<T>>> AmplitudeKernels<D>
    for SymbolicaDualAmplitudeKernels<T, D>
where
    Complex<T>: Real + EvaluationDomain,
{
    type Boxes = SymbolicaDualBoxKernels<T, D>;
    type Pentagons = SymbolicaDualPentagonKernels<T, D>;
    type Triangles = SymbolicaDualTriangleKernels<T, D>;
    type Bubbles = SymbolicaDualBubbleKernels<T, D>;
    type Tadpoles = SymbolicaDualTadpoleKernels<T, D>;
    type Integration = SymbolicaDualIntegrationKernels<T, D>;
    fn propagators(&self) -> usize {
        self.n
    }
    fn rank(&self) -> usize {
        self.rank
    }
    fn box_options(&self) -> BoxReductionOptions {
        self.box_options
    }
    fn box_kernels(&self) -> Result<Self::Boxes> {
        SymbolicaDualBoxKernels::new(
            self.n,
            self.rank,
            self.box_options,
            self.prototype.clone(),
            &self.dual,
        )
    }
    fn pentagon_kernels(&self) -> Result<Self::Pentagons> {
        SymbolicaDualPentagonKernels::new(self.n, self.rank, self.prototype.clone(), &self.dual)
    }
    fn triangle_kernels(&self) -> Result<Self::Triangles> {
        SymbolicaDualTriangleKernels::new(self.n, self.rank, self.prototype.clone(), &self.dual)
    }
    fn bubble_kernels(&self) -> Result<Self::Bubbles> {
        SymbolicaDualBubbleKernels::new(self.n, self.rank, self.prototype.clone(), &self.dual)
    }
    fn tadpole_kernels(&self) -> Result<Self::Tadpoles> {
        SymbolicaDualTadpoleKernels::new(self.n, self.rank, self.prototype.clone(), &self.dual)
    }
    fn integration_kernels(&self, cut_size: usize) -> Result<Self::Integration> {
        SymbolicaDualIntegrationKernels::new(
            cut_size,
            self.rank > self.n,
            self.prototype.clone(),
            &self.dual,
        )
    }
    fn accumulate_batch(&mut self, input: &[D], output: &mut [D], rows: usize) -> Result<()> {
        self.accumulation.evaluate_batch(input, output, rows)
    }
    fn values_are_finite(&self, values: &[D]) -> bool {
        values.iter().all(|v| {
            v.values()
                .iter()
                .all(|v| v.re.is_finite() && v.im.is_finite())
        })
    }
    fn values_equal(&self, left: &D, right: &D) -> bool {
        left.get_shape() == right.get_shape() && left.values() == right.values()
    }
}
