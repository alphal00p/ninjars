//! Plain binary64 complex arithmetic and Minkowski products for the fast engine.
//!
//! `C64` deliberately implements textbook (Fortran-rule) complex multiplication
//! and division without NaN/infinity recovery or rescaling, matching the
//! arithmetic original Ninja performs when compiled with `-fcx-fortran-rules`.
//! It has the same `repr(C)` layout as Symbolica's `Complex<f64>`.

use std::ops::{Add, AddAssign, Div, DivAssign, Mul, MulAssign, Neg, Sub, SubAssign};

use symbolica::domains::float::Complex;

#[derive(Clone, Copy, Debug, Default, PartialEq)]
#[repr(C)]
pub struct C64 {
    pub re: f64,
    pub im: f64,
}

/// Real four-momentum `(E, x, y, z)`, metric `(+,-,-,-)`.
pub type RMom = [f64; 4];
/// Complex four-momentum `(E, x, y, z)`, metric `(+,-,-,-)`.
pub type CMom = [C64; 4];

impl C64 {
    pub const ZERO: Self = Self { re: 0.0, im: 0.0 };
    pub const ONE: Self = Self { re: 1.0, im: 0.0 };
    pub const I: Self = Self { re: 0.0, im: 1.0 };

    #[inline(always)]
    pub const fn new(re: f64, im: f64) -> Self {
        Self { re, im }
    }
    #[inline(always)]
    pub const fn real(re: f64) -> Self {
        Self { re, im: 0.0 }
    }
    #[inline(always)]
    pub fn conj(self) -> Self {
        Self::new(self.re, -self.im)
    }
    #[inline(always)]
    pub fn norm_sqr(self) -> f64 {
        self.re * self.re + self.im * self.im
    }
    #[inline(always)]
    pub fn abs(self) -> f64 {
        self.re.hypot(self.im)
    }
    /// Taxicab norm `|re| + |im|`, as used by Ninja's stability checks.
    #[inline(always)]
    pub fn taxicab(self) -> f64 {
        self.re.abs() + self.im.abs()
    }
    #[inline(always)]
    pub fn is_finite(self) -> bool {
        self.re.is_finite() && self.im.is_finite()
    }
    #[inline(always)]
    pub fn is_zero(self) -> bool {
        self.re == 0.0 && self.im == 0.0
    }
    #[inline(always)]
    pub fn inv(self) -> Self {
        let n = 1.0 / self.norm_sqr();
        Self::new(self.re * n, -self.im * n)
    }
    #[inline(always)]
    pub fn scale(self, s: f64) -> Self {
        Self::new(self.re * s, self.im * s)
    }
    /// Principal square root (branch cut on the negative real axis), as
    /// `std::sqrt(std::complex<double>)`.
    #[inline]
    pub fn sqrt(self) -> Self {
        if self.im == 0.0 {
            return if self.re >= 0.0 {
                Self::new(self.re.sqrt(), self.im)
            } else {
                Self::new(0.0, (-self.re).sqrt().copysign(self.im))
            };
        }
        let r = self.abs();
        let t = ((r + self.re.abs()) * 0.5).sqrt();
        if self.re >= 0.0 {
            Self::new(t, self.im / (2.0 * t))
        } else {
            Self::new(self.im.abs() / (2.0 * t), t.copysign(self.im))
        }
    }
    #[inline(always)]
    pub fn from_symbolica(value: Complex<f64>) -> Self {
        Self::new(value.re, value.im)
    }
    #[inline(always)]
    pub fn to_symbolica(self) -> Complex<f64> {
        Complex::new(self.re, self.im)
    }
}

impl From<f64> for C64 {
    #[inline(always)]
    fn from(re: f64) -> Self {
        Self::real(re)
    }
}

impl Add for C64 {
    type Output = Self;
    #[inline(always)]
    fn add(self, o: Self) -> Self {
        Self::new(self.re + o.re, self.im + o.im)
    }
}
impl Sub for C64 {
    type Output = Self;
    #[inline(always)]
    fn sub(self, o: Self) -> Self {
        Self::new(self.re - o.re, self.im - o.im)
    }
}
impl Mul for C64 {
    type Output = Self;
    #[inline(always)]
    fn mul(self, o: Self) -> Self {
        Self::new(
            self.re * o.re - self.im * o.im,
            self.re * o.im + self.im * o.re,
        )
    }
}
impl Div for C64 {
    type Output = Self;
    #[inline(always)]
    fn div(self, o: Self) -> Self {
        let n = 1.0 / o.norm_sqr();
        Self::new(
            (self.re * o.re + self.im * o.im) * n,
            (self.im * o.re - self.re * o.im) * n,
        )
    }
}
impl Neg for C64 {
    type Output = Self;
    #[inline(always)]
    fn neg(self) -> Self {
        Self::new(-self.re, -self.im)
    }
}
impl Add<f64> for C64 {
    type Output = Self;
    #[inline(always)]
    fn add(self, o: f64) -> Self {
        Self::new(self.re + o, self.im)
    }
}
impl Sub<f64> for C64 {
    type Output = Self;
    #[inline(always)]
    fn sub(self, o: f64) -> Self {
        Self::new(self.re - o, self.im)
    }
}
impl Mul<f64> for C64 {
    type Output = Self;
    #[inline(always)]
    fn mul(self, o: f64) -> Self {
        Self::new(self.re * o, self.im * o)
    }
}
impl Div<f64> for C64 {
    type Output = Self;
    #[inline(always)]
    fn div(self, o: f64) -> Self {
        let r = 1.0 / o;
        Self::new(self.re * r, self.im * r)
    }
}
impl Add<C64> for f64 {
    type Output = C64;
    #[inline(always)]
    fn add(self, o: C64) -> C64 {
        C64::new(self + o.re, o.im)
    }
}
impl Sub<C64> for f64 {
    type Output = C64;
    #[inline(always)]
    fn sub(self, o: C64) -> C64 {
        C64::new(self - o.re, -o.im)
    }
}
impl Mul<C64> for f64 {
    type Output = C64;
    #[inline(always)]
    fn mul(self, o: C64) -> C64 {
        C64::new(self * o.re, self * o.im)
    }
}
impl Div<C64> for f64 {
    type Output = C64;
    // A real numerator: one complex reciprocal scaled by `self`.
    #[allow(clippy::suspicious_arithmetic_impl)]
    #[inline(always)]
    fn div(self, o: C64) -> C64 {
        o.inv() * self
    }
}

macro_rules! assign_ops {
    ($($trait:ident $method:ident $op:tt $rhs:ty;)*) => {$(
        impl $trait<$rhs> for C64 {
            #[inline(always)]
            fn $method(&mut self, o: $rhs) {
                *self = *self $op o;
            }
        }
    )*};
}
assign_ops! {
    AddAssign add_assign + C64; SubAssign sub_assign - C64;
    MulAssign mul_assign * C64; DivAssign div_assign / C64;
    AddAssign add_assign + f64; SubAssign sub_assign - f64;
    MulAssign mul_assign * f64; DivAssign div_assign / f64;
}

/// Minkowski product of two real momenta.
#[inline(always)]
pub fn mp_rr(a: &RMom, b: &RMom) -> f64 {
    a[0] * b[0] - a[1] * b[1] - a[2] * b[2] - a[3] * b[3]
}

/// Minkowski product of a complex and a real momentum.
#[inline(always)]
pub fn mp_cr(a: &CMom, b: &RMom) -> C64 {
    a[0] * b[0] - a[1] * b[1] - a[2] * b[2] - a[3] * b[3]
}

/// Minkowski product of two complex momenta.
#[inline(always)]
pub fn mp_cc(a: &CMom, b: &CMom) -> C64 {
    a[0] * b[0] - a[1] * b[1] - a[2] * b[2] - a[3] * b[3]
}

#[inline(always)]
pub fn to_complex(a: &RMom) -> CMom {
    [
        C64::real(a[0]),
        C64::real(a[1]),
        C64::real(a[2]),
        C64::real(a[3]),
    ]
}
