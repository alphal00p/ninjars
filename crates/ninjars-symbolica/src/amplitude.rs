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

#[cfg(feature = "native-kernels")]
type StageFactory<T, K> = fn(usize, usize, T) -> Result<K>;
#[cfg(feature = "native-kernels")]
type DualStageFactory<T, D, K> = fn(usize, usize, T, &D) -> Result<K>;

/// Exact reduction and integration kernels for complex f64, DoubleFloat or Float.
/// Construct once, then call `into_amplitude(lanes)` to allocate the coordinator.
pub struct SymbolicaAmplitudeKernels<T> {
    #[cfg(feature = "native-kernels")]
    native_bubble_corrections: bool,
    #[cfg(feature = "native-kernels")]
    native_tadpole_projections: bool,
    #[cfg(feature = "native-kernels")]
    native_triangle: Option<StageFactory<T, SymbolicaTriangleKernels<T>>>,
    #[cfg(feature = "native-kernels")]
    native_bubble: Option<StageFactory<T, SymbolicaBubbleKernels<T>>>,
    #[cfg(feature = "native-kernels")]
    native_tadpole: Option<StageFactory<T, SymbolicaTadpoleKernels<T>>>,

    n: usize,
    rank: usize,
    rank_specialized_corrections: bool,
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
            #[cfg(feature = "native-kernels")]
            native_triangle: None,
            #[cfg(feature = "native-kernels")]
            native_bubble_corrections: false,
            #[cfg(feature = "native-kernels")]
            native_tadpole_projections: false,
            #[cfg(feature = "native-kernels")]
            native_bubble: None,
            #[cfg(feature = "native-kernels")]
            native_tadpole: None,
            n: propagators,
            rank,
            rank_specialized_corrections: false,
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

    /// Opt into guarded n=5/r=4 correction bodies in the bubble/tadpole stages.
    /// These alternates use the Symbolica interpreter; native selections remain
    /// available as generic fallbacks. Either builder order preserves both.
    /// With `native-kernels`, consult both backend reports when enabling this.
    pub fn with_rank_specialized_corrections(mut self) -> Result<Self> {
        crate::rank_specialization::require_supported(self.n, self.rank)?;
        self.rank_specialized_corrections = true;
        Ok(self)
    }

    pub fn rank_specialized_corrections_enabled(&self) -> bool {
        self.rank_specialized_corrections
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
        #[cfg(feature = "native-kernels")]
        if let Some(factory) = self.native_triangle {
            return factory(self.n, self.rank, self.prototype.clone());
        }
        SymbolicaTriangleKernels::new(self.n, self.rank, self.prototype.clone())
    }
    fn bubble_kernels(&self) -> Result<Self::Bubbles> {
        #[cfg(feature = "native-kernels")]
        let kernels = if let Some(factory) = self.native_bubble {
            factory(self.n, self.rank, self.prototype.clone())?
        } else {
            SymbolicaBubbleKernels::new(self.n, self.rank, self.prototype.clone())?
        };
        #[cfg(not(feature = "native-kernels"))]
        let kernels = SymbolicaBubbleKernels::new(self.n, self.rank, self.prototype.clone())?;
        if self.rank_specialized_corrections {
            kernels.with_rank_specialized_corrections()
        } else {
            Ok(kernels)
        }
    }
    fn tadpole_kernels(&self) -> Result<Self::Tadpoles> {
        #[cfg(feature = "native-kernels")]
        let kernels = if let Some(factory) = self.native_tadpole {
            factory(self.n, self.rank, self.prototype.clone())?
        } else {
            SymbolicaTadpoleKernels::new(self.n, self.rank, self.prototype.clone())?
        };
        #[cfg(not(feature = "native-kernels"))]
        let kernels = SymbolicaTadpoleKernels::new(self.n, self.rank, self.prototype.clone())?;
        if self.rank_specialized_corrections {
            kernels.with_rank_specialized_corrections()
        } else {
            Ok(kernels)
        }
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
    #[cfg(feature = "native-kernels")]
    native_tadpole_projections: bool,
    #[cfg(feature = "native-kernels")]
    native_triangle: Option<DualStageFactory<T, D, SymbolicaDualTriangleKernels<T, D>>>,
    #[cfg(feature = "native-kernels")]
    native_bubble: Option<DualStageFactory<T, D, SymbolicaDualBubbleKernels<T, D>>>,
    #[cfg(feature = "native-kernels")]
    native_tadpole: Option<DualStageFactory<T, D, SymbolicaDualTadpoleKernels<T, D>>>,

    n: usize,
    rank: usize,
    rank_specialized_corrections: bool,
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
            #[cfg(feature = "native-kernels")]
            native_tadpole_projections: false,
            #[cfg(feature = "native-kernels")]
            native_triangle: None,
            #[cfg(feature = "native-kernels")]
            native_bubble: None,
            #[cfg(feature = "native-kernels")]
            native_tadpole: None,
            n: propagators,
            rank,
            rank_specialized_corrections: false,
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

    /// See [`SymbolicaAmplitudeKernels::with_rank_specialized_corrections`].
    /// The guard checks every higher/mixed derivative component.
    pub fn with_rank_specialized_corrections(mut self) -> Result<Self> {
        crate::rank_specialization::require_supported(self.n, self.rank)?;
        self.rank_specialized_corrections = true;
        Ok(self)
    }

    pub fn rank_specialized_corrections_enabled(&self) -> bool {
        self.rank_specialized_corrections
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
        #[cfg(feature = "native-kernels")]
        if let Some(factory) = self.native_triangle {
            return factory(self.n, self.rank, self.prototype.clone(), &self.dual);
        }
        SymbolicaDualTriangleKernels::new(self.n, self.rank, self.prototype.clone(), &self.dual)
    }
    fn bubble_kernels(&self) -> Result<Self::Bubbles> {
        #[cfg(feature = "native-kernels")]
        let kernels = if let Some(factory) = self.native_bubble {
            factory(self.n, self.rank, self.prototype.clone(), &self.dual)?
        } else {
            SymbolicaDualBubbleKernels::new(self.n, self.rank, self.prototype.clone(), &self.dual)?
        };
        #[cfg(not(feature = "native-kernels"))]
        let kernels =
            SymbolicaDualBubbleKernels::new(self.n, self.rank, self.prototype.clone(), &self.dual)?;
        if self.rank_specialized_corrections {
            kernels.with_rank_specialized_corrections()
        } else {
            Ok(kernels)
        }
    }
    fn tadpole_kernels(&self) -> Result<Self::Tadpoles> {
        #[cfg(feature = "native-kernels")]
        let kernels = if let Some(factory) = self.native_tadpole {
            factory(self.n, self.rank, self.prototype.clone(), &self.dual)?
        } else {
            SymbolicaDualTadpoleKernels::new(self.n, self.rank, self.prototype.clone(), &self.dual)?
        };
        #[cfg(not(feature = "native-kernels"))]
        let kernels = SymbolicaDualTadpoleKernels::new(
            self.n,
            self.rank,
            self.prototype.clone(),
            &self.dual,
        )?;
        if self.rank_specialized_corrections {
            kernels.with_rank_specialized_corrections()
        } else {
            Ok(kernels)
        }
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

#[cfg(feature = "native-kernels")]
impl<T: oneloop::rust_codegen::runtime::NativeFloat> SymbolicaAmplitudeKernels<T>
where
    Complex<T>: Real + EvaluationDomain,
{
    /// Opt into the partial native catalog. Only T2/T3 denominator expansions
    /// change backend; callbacks, geometry, masters and other kernels do not.
    pub fn with_native_denominators(mut self) -> Self {
        self.native_bubble_corrections = false;
        self.native_tadpole_projections = false;
        self.native_triangle = Some(SymbolicaTriangleKernels::new_with_native_denominators);
        self.native_bubble = Some(SymbolicaBubbleKernels::new_with_native_denominators);
        self.native_tadpole = Some(SymbolicaTadpoleKernels::new_with_native_denominators);
        self
    }

    /// Select all currently catalogued native kernels. This includes T2/T3
    /// denominators and ordinary bubble source projections and triangle-parent
    /// corrections, plus tadpole parent-axis projections. Other reduction,
    /// geometry and integration stay interpreted.
    pub fn with_native_kernels(mut self) -> Self {
        self = self.with_native_denominators();
        self.native_bubble = Some(SymbolicaBubbleKernels::new_with_native_kernels);
        self.native_bubble_corrections = true;
        self.native_tadpole = Some(SymbolicaTadpoleKernels::new_with_native_kernels);
        self.native_tadpole_projections = true;
        self
    }
}

#[cfg(feature = "native-kernels")]
impl<T: Real + SingleFloat> SymbolicaAmplitudeKernels<T>
where
    Complex<T>: Real + EvaluationDomain,
{
    /// Generic/fallback per-family selection, not the conditional alternate.
    /// When specialization is enabled, also consult
    /// [`Self::rank_specialized_correction_backend`].
    pub fn kernel_backends(&self) -> [crate::KernelBackendSelection; 8] {
        crate::native_kernels::coverage(
            self.native_triangle.is_some(),
            self.native_bubble_corrections,
            self.native_tadpole_projections,
        )
    }

    /// Backend of guarded correction alternates; generic fallback selections
    /// remain in [`Self::kernel_backends`]. Actual selection depends on inputs.
    pub fn rank_specialized_correction_backend(&self) -> Option<crate::KernelBackend> {
        self.rank_specialized_corrections
            .then_some(crate::KernelBackend::SymbolicaInterpreter)
    }
}

#[cfg(feature = "native-kernels")]
impl<T: oneloop::rust_codegen::runtime::NativeFloat, D: Real + DualNumber<Complex<T>>>
    SymbolicaDualAmplitudeKernels<T, D>
where
    Complex<T>: Real + EvaluationDomain,
{
    /// Opt into the partial native catalog. Only T2/T3 denominator expansions
    /// change backend; callbacks, geometry, masters and other kernels do not.
    pub fn with_native_denominators(mut self) -> Self {
        self.native_tadpole_projections = false;
        self.native_triangle = Some(SymbolicaDualTriangleKernels::new_with_native_denominators);
        self.native_bubble = Some(SymbolicaDualBubbleKernels::new_with_native_denominators);
        self.native_tadpole = Some(SymbolicaDualTadpoleKernels::new_with_native_denominators);
        self
    }

    /// Select available Dual denominators and tadpole parent-axis projections
    /// for the supported ordered shape. Bubble corrections stay interpreted.
    pub fn with_native_kernels(mut self) -> Self {
        self = self.with_native_denominators();
        self.native_tadpole = Some(SymbolicaDualTadpoleKernels::new_with_native_kernels);
        self.native_tadpole_projections = true;
        self
    }
}

#[cfg(feature = "native-kernels")]
impl<T: Real + SingleFloat, D: Real + DualNumber<Complex<T>>> SymbolicaDualAmplitudeKernels<T, D>
where
    Complex<T>: Real + EvaluationDomain,
{
    /// Generic/fallback per-family selection, not the conditional alternate.
    /// When specialization is enabled, also consult
    /// [`Self::rank_specialized_correction_backend`].
    pub fn kernel_backends(&self) -> [crate::KernelBackendSelection; 8] {
        let supported = crate::native_kernels::supported_shape(&self.dual.get_shape());
        crate::native_kernels::coverage(
            self.native_triangle.is_some() && supported,
            false,
            self.native_tadpole_projections && supported,
        )
    }

    /// Backend of guarded correction alternates; see [`Self::kernel_backends`]
    /// for the separate generic fallback selections.
    pub fn rank_specialized_correction_backend(&self) -> Option<crate::KernelBackend> {
        self.rank_specialized_corrections
            .then_some(crate::KernelBackend::SymbolicaInterpreter)
    }
}
