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
        // Float tracks significant bits per component. Cancellation changes
        // that count, but not the scalar carrier or derivative shape.
        true
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
