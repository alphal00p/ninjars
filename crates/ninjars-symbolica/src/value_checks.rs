//! Static dispatch for the numerical validity checks, and the precision-aware
//! singularity test shared by every stage kernel ([`Negligible`]).
//!
//! Ordinary carriers have no runtime derivative shape. Keeping that fact in
//! the executor's type lets the compiler remove the shape check and inline
//! Symbolica's finite/zero predicates. Dual checks still inspect every
//! component and compare the complete ordered shape with the prototype.

use std::marker::PhantomData;

use ninjars::{Complex, FloatLike, FourMomentum, Real, SingleFloat};
use symbolica::domains::{dual::DualNumber, rational::Rational};

pub(crate) trait ValueChecks<V> {
    /// Real carrier of the primal component.
    type Real: Real + SingleFloat;
    fn finite(value: &V) -> bool;
    fn nonzero(value: &V) -> bool;
    fn shape_matches(value: &V, prototype: &V) -> bool;
    /// Taxicab magnitude `|Re| + |Im|` of the primal component, as Ninja's
    /// `taxicab_norm`.
    fn magnitude(value: &V) -> Self::Real;
}

pub(crate) struct OrdinaryChecks;

impl<T: Real + SingleFloat> ValueChecks<Complex<T>> for OrdinaryChecks
where
    Complex<T>: Real,
{
    type Real = T;

    #[inline]
    fn finite(value: &Complex<T>) -> bool {
        value.is_finite()
    }

    #[inline]
    fn nonzero(value: &Complex<T>) -> bool {
        !value.is_fully_zero()
    }

    #[inline]
    fn shape_matches(_: &Complex<T>, _: &Complex<T>) -> bool {
        // Float tracks significant bits per component. Cancellation changes
        // that count, but not the scalar carrier or derivative shape.
        true
    }

    #[inline]
    fn magnitude(value: &Complex<T>) -> T {
        value.re.norm() + value.im.norm()
    }
}

pub(crate) struct DualChecks<T>(PhantomData<fn() -> T>);

/// Integration checks include every derivative component when comparing
/// invariants or testing zero weights. Pivots retain primal-only `nonzero`.
pub(crate) trait IntegrationChecks<V>: ValueChecks<V> {
    fn equal(left: &V, right: &V) -> bool;
    fn is_zero(value: &V) -> bool;
    fn valid_scale(value: &V) -> bool;
}

impl<T: Real + SingleFloat> IntegrationChecks<Complex<T>> for OrdinaryChecks
where
    Complex<T>: Real,
{
    #[inline]
    fn equal(left: &Complex<T>, right: &Complex<T>) -> bool {
        left == right
    }

    #[inline]
    fn is_zero(value: &Complex<T>) -> bool {
        value.re.is_zero() && value.im.is_zero()
    }

    #[inline]
    fn valid_scale(value: &Complex<T>) -> bool {
        value.im.is_zero()
            && value.re.real_cmp(&value.re.zero()) == Some(std::cmp::Ordering::Greater)
    }
}

impl<T: Real + SingleFloat, D: DualNumber<Complex<T>>> IntegrationChecks<D> for DualChecks<T>
where
    Complex<T>: Real,
{
    #[inline]
    fn equal(left: &D, right: &D) -> bool {
        left.get_shape() == right.get_shape() && left.values() == right.values()
    }

    #[inline]
    fn is_zero(value: &D) -> bool {
        value.values().iter().all(OrdinaryChecks::is_zero)
    }

    #[inline]
    fn valid_scale(value: &D) -> bool {
        OrdinaryChecks::valid_scale(&value.values()[0])
    }
}

impl<T: Real + SingleFloat, D: DualNumber<Complex<T>>> ValueChecks<D> for DualChecks<T>
where
    Complex<T>: Real,
{
    type Real = T;

    #[inline]
    fn finite(value: &D) -> bool {
        value.values().iter().all(|component| component.is_finite())
    }

    #[inline]
    fn nonzero(value: &D) -> bool {
        !value.values()[0].is_fully_zero()
    }

    #[inline]
    fn shape_matches(value: &D, prototype: &D) -> bool {
        value.values().len() == prototype.values().len()
            && value.get_shape() == prototype.get_shape()
    }

    #[inline]
    fn magnitude(value: &D) -> T {
        OrdinaryChecks::magnitude(&value.values()[0])
    }
}

/// Precision-aware test for a quantity that vanishes at a singular point.
///
/// Whether such a quantity comes out as exactly zero or as rounding noise of
/// order `ε · scale` depends on the precision and on the operation order, so
/// an exact-zero test accepts or rejects the same singular point depending on
/// the carrier. A value is negligible when its taxicab magnitude is at most
/// `2^10 ε · scale`, where `ε = 2^(1-p)` is the carrier's machine epsilon
/// (Ninja's `REAL_EPS = 10³ ε`) and `scale` bounds the magnitudes of the terms
/// the value was computed from. Carriers with the binary64 exponent range
/// (`f64`, `DoubleFloat`) also reject magnitudes below Ninja's absolute
/// floating-point threshold `REAL_MIN = 10^20 DBL_MIN`. Dual carriers test
/// their primal component.
pub(crate) struct Negligible<T> {
    tolerance: T,
    floor: T,
}

impl<T: Real + SingleFloat> Negligible<T> {
    pub(crate) fn new(prototype: &T) -> Self {
        let precision = prototype.get_precision();
        let tolerance = prototype
            .from_i64(2)
            .pow(u64::from(precision.saturating_sub(11)))
            .inv();
        let floor = if prototype.fixed_precision() {
            let minimum = Rational::try_from(f64::MIN_POSITIVE).expect("normal binary64");
            prototype.from_rational(&minimum) * prototype.from_i64(10).pow(20)
        } else {
            prototype.zero()
        };
        Self { tolerance, floor }
    }

    /// `magnitude` is negligible relative to `scale` (non-finite inputs are).
    #[inline]
    pub(crate) fn test(&self, magnitude: &T, scale: &T) -> bool {
        let greater = |a: &T, b: &T| a.real_cmp(b) == Some(std::cmp::Ordering::Greater);
        !(greater(magnitude, &(self.tolerance.clone() * scale)) && !greater(&self.floor, magnitude))
    }

    /// `value` is negligible relative to `scale`.
    #[inline]
    pub(crate) fn value<V, C: ValueChecks<V, Real = T>>(&self, value: &V, scale: &T) -> bool {
        self.test(&C::magnitude(value), scale)
    }
}

/// Scale of `(q + shift)² - m² - mu²`: `Σ_μ (|q_μ| + |shift_μ|)² + |m²| + |mu²|`.
pub(crate) fn denominator_scale<V, C: ValueChecks<V>>(
    q: &FourMomentum<V>,
    shift: &FourMomentum<V>,
    mass_squared: &V,
    mu_squared: &V,
) -> C::Real {
    let mut scale = C::magnitude(mass_squared) + C::magnitude(mu_squared);
    for (a, b) in q.components().into_iter().zip(shift.components()) {
        let term = C::magnitude(a) + C::magnitude(b);
        scale += term.clone() * term;
    }
    scale
}

/// Scale of `2 v.(a - b)`: `2 Σ_μ |v_μ| (|a_μ| + |b_μ|)`.
pub(crate) fn difference_dot_scale<V, C: ValueChecks<V>>(
    v: &FourMomentum<V>,
    a: &FourMomentum<V>,
    b: &FourMomentum<V>,
) -> C::Real {
    let mut scale: Option<C::Real> = None;
    for ((v, a), b) in v
        .components()
        .into_iter()
        .zip(a.components())
        .zip(b.components())
    {
        let term = C::magnitude(v) * (C::magnitude(a) + C::magnitude(b));
        scale = Some(match scale {
            Some(sum) => sum + term,
            None => term,
        });
    }
    let scale = scale.unwrap();
    scale.clone() + scale
}

/// Scale of `v.v`: `Σ_μ |v_μ|²`.
pub(crate) fn square_scale<V, C: ValueChecks<V>>(v: &FourMomentum<V>) -> C::Real {
    let mut scale: Option<C::Real> = None;
    for component in v.components() {
        let term = C::magnitude(component);
        let term = term.clone() * term;
        scale = Some(match scale {
            Some(sum) => sum + term,
            None => term,
        });
    }
    scale.unwrap()
}
