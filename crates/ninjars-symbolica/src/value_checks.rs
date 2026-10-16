//! Static dispatch for the existing numerical validity checks.
//!
//! Ordinary carriers have no runtime derivative shape. Keeping that fact in
//! the executor's type lets the compiler remove the shape check and inline
//! Symbolica's finite/zero predicates. Dual checks still inspect every
//! component and compare the complete ordered shape with the prototype.

use std::marker::PhantomData;

use ninjars::{Complex, FloatLike, Real, SingleFloat};
use symbolica::domains::dual::DualNumber;

pub(crate) trait ValueChecks<V> {
    fn finite(value: &V) -> bool;
    fn nonzero(value: &V) -> bool;
    fn shape_matches(value: &V, prototype: &V) -> bool;
}

pub(crate) struct OrdinaryChecks;

impl<T: Real + SingleFloat> ValueChecks<Complex<T>> for OrdinaryChecks
where
    Complex<T>: Real,
{
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
        true
    }
}

pub(crate) struct DualChecks<T>(PhantomData<fn() -> T>);

impl<T: Real + SingleFloat, D: DualNumber<Complex<T>>> ValueChecks<D> for DualChecks<T>
where
    Complex<T>: Real,
{
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
}
