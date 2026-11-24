//! Double-double arithmetic for the fast engine's degenerate-triangle
//! re-reduction ([`super::engine::FastAmplitude`], `dd_stages`).
//!
//! [`Dd`] is an unevaluated sum `hi + lo` of two binary64 numbers with
//! `|lo| <= ulp(hi) / 2` (about 106 bits), with the textbook error-free
//! transformations (Knuth's two-sum, Dekker's split product: no fused
//! multiply-add, which baseline x86-64 lacks). [`Cd`] is a complex number of
//! two [`Dd`] with the same Fortran-rule arithmetic as [`super::C64`].
//! Mixed operations with `f64` promote the `f64` exactly.

use std::cmp::Ordering;
use std::ops::{Add, AddAssign, Div, DivAssign, Mul, MulAssign, Neg, Sub, SubAssign};

use super::complex::C64;

#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub struct Dd {
    pub hi: f64,
    pub lo: f64,
}

#[inline(always)]
fn two_sum(a: f64, b: f64) -> (f64, f64) {
    let s = a + b;
    let bb = s - a;
    (s, (a - (s - bb)) + (b - bb))
}

#[inline(always)]
fn quick_two_sum(a: f64, b: f64) -> (f64, f64) {
    let s = a + b;
    (s, b - (s - a))
}

/// Dekker's split: `a = hi + lo` with 26-bit halves (inputs below 2^996).
#[inline(always)]
fn split(a: f64) -> (f64, f64) {
    const SPLITTER: f64 = 134_217_729.0; // 2^27 + 1
    let t = SPLITTER * a;
    let hi = t - (t - a);
    (hi, a - hi)
}

#[inline(always)]
fn two_prod(a: f64, b: f64) -> (f64, f64) {
    let p = a * b;
    let (ah, al) = split(a);
    let (bh, bl) = split(b);
    (p, ((ah * bh - p) + ah * bl + al * bh) + al * bl)
}

impl Dd {
    pub const ZERO: Self = Self { hi: 0., lo: 0. };
    pub const ONE: Self = Self { hi: 1., lo: 0. };

    #[inline(always)]
    pub const fn from_f64(x: f64) -> Self {
        Self { hi: x, lo: 0. }
    }
    #[inline(always)]
    pub fn to_f64(self) -> f64 {
        self.hi + self.lo
    }
    #[inline(always)]
    pub fn abs(self) -> Self {
        if self.hi < 0. || (self.hi == 0. && self.lo < 0.) {
            -self
        } else {
            self
        }
    }
    #[inline(always)]
    pub fn is_finite(self) -> bool {
        self.hi.is_finite() && self.lo.is_finite()
    }
    pub fn sqrt(self) -> Self {
        if self.hi <= 0. {
            return if self.hi == 0. {
                Self::ZERO
            } else {
                Self::from_f64(f64::NAN)
            };
        }
        let q = self.hi.sqrt();
        let (p, e) = two_prod(q, q);
        // r = (self - q²) / (2 q)
        let r = ((self.hi - p) - e + self.lo) / (2. * q);
        let (s, t) = quick_two_sum(q, r);
        Self { hi: s, lo: t }
    }
    #[inline(always)]
    pub fn max(self, other: Self) -> Self {
        if self < other { other } else { self }
    }
    #[inline(always)]
    pub fn powi(self, n: i32) -> Self {
        let mut r = Self::ONE;
        for _ in 0..n.unsigned_abs() {
            r = r * self;
        }
        if n < 0 { Self::ONE / r } else { r }
    }
}

impl From<f64> for Dd {
    #[inline(always)]
    fn from(x: f64) -> Self {
        Self::from_f64(x)
    }
}

impl Neg for Dd {
    type Output = Self;
    #[inline(always)]
    fn neg(self) -> Self {
        Self {
            hi: -self.hi,
            lo: -self.lo,
        }
    }
}

impl Add for Dd {
    type Output = Self;
    #[inline(always)]
    fn add(self, b: Self) -> Self {
        let (s, e) = two_sum(self.hi, b.hi);
        let (t, f) = two_sum(self.lo, b.lo);
        let (s, e) = quick_two_sum(s, e + t);
        let (hi, lo) = quick_two_sum(s, e + f);
        Self { hi, lo }
    }
}

impl Sub for Dd {
    type Output = Self;
    #[inline(always)]
    fn sub(self, b: Self) -> Self {
        self + (-b)
    }
}

impl Mul for Dd {
    type Output = Self;
    #[inline(always)]
    fn mul(self, b: Self) -> Self {
        let (p, e) = two_prod(self.hi, b.hi);
        let (hi, lo) = quick_two_sum(p, e + (self.hi * b.lo + self.lo * b.hi));
        Self { hi, lo }
    }
}

impl Div for Dd {
    type Output = Self;
    #[inline(always)]
    fn div(self, b: Self) -> Self {
        let q1 = self.hi / b.hi;
        let r = self - b * q1;
        let q2 = r.hi / b.hi;
        let r = r - b * q2;
        let q3 = r.hi / b.hi;
        let (hi, lo) = quick_two_sum(q1, q2);
        Self { hi, lo } + Self::from_f64(q3)
    }
}

impl Add<f64> for Dd {
    type Output = Self;
    #[inline(always)]
    fn add(self, b: f64) -> Self {
        let (s, e) = two_sum(self.hi, b);
        let (hi, lo) = quick_two_sum(s, e + self.lo);
        Self { hi, lo }
    }
}
impl Sub<f64> for Dd {
    type Output = Self;
    #[inline(always)]
    fn sub(self, b: f64) -> Self {
        self + (-b)
    }
}
impl Mul<f64> for Dd {
    type Output = Self;
    #[inline(always)]
    fn mul(self, b: f64) -> Self {
        let (p, e) = two_prod(self.hi, b);
        let (hi, lo) = quick_two_sum(p, e + self.lo * b);
        Self { hi, lo }
    }
}
impl Div<f64> for Dd {
    type Output = Self;
    #[inline(always)]
    fn div(self, b: f64) -> Self {
        self / Self::from_f64(b)
    }
}
impl Add<Dd> for f64 {
    type Output = Dd;
    #[inline(always)]
    fn add(self, b: Dd) -> Dd {
        b + self
    }
}
impl Sub<Dd> for f64 {
    type Output = Dd;
    #[inline(always)]
    fn sub(self, b: Dd) -> Dd {
        (-b) + self
    }
}
impl Mul<Dd> for f64 {
    type Output = Dd;
    #[inline(always)]
    fn mul(self, b: Dd) -> Dd {
        b * self
    }
}
impl Div<Dd> for f64 {
    type Output = Dd;
    #[inline(always)]
    fn div(self, b: Dd) -> Dd {
        Dd::from_f64(self) / b
    }
}

macro_rules! assign_ops {
    ($t:ty, $rhs:ty) => {
        impl AddAssign<$rhs> for $t {
            #[inline(always)]
            fn add_assign(&mut self, b: $rhs) {
                *self = *self + b;
            }
        }
        impl SubAssign<$rhs> for $t {
            #[inline(always)]
            fn sub_assign(&mut self, b: $rhs) {
                *self = *self - b;
            }
        }
        impl MulAssign<$rhs> for $t {
            #[inline(always)]
            fn mul_assign(&mut self, b: $rhs) {
                *self = *self * b;
            }
        }
        impl DivAssign<$rhs> for $t {
            #[inline(always)]
            fn div_assign(&mut self, b: $rhs) {
                *self = *self / b;
            }
        }
    };
}
assign_ops!(Dd, Dd);
assign_ops!(Dd, f64);

impl PartialOrd for Dd {
    #[inline(always)]
    fn partial_cmp(&self, b: &Self) -> Option<Ordering> {
        match self.hi.partial_cmp(&b.hi)? {
            Ordering::Equal => self.lo.partial_cmp(&b.lo),
            o => Some(o),
        }
    }
}
impl PartialEq<f64> for Dd {
    #[inline(always)]
    fn eq(&self, b: &f64) -> bool {
        self.hi == *b && self.lo == 0.
    }
}
impl PartialOrd<f64> for Dd {
    #[inline(always)]
    fn partial_cmp(&self, b: &f64) -> Option<Ordering> {
        self.partial_cmp(&Dd::from_f64(*b))
    }
}

impl std::iter::Sum for Dd {
    fn sum<I: Iterator<Item = Dd>>(iter: I) -> Dd {
        iter.fold(Dd::ZERO, |a, b| a + b)
    }
}

/// Complex double-double.
#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub struct Cd {
    pub re: Dd,
    pub im: Dd,
}

/// Real four-momentum in double-double.
pub type RMomD = [Dd; 4];
/// Complex four-momentum in double-double.
pub type CMomD = [Cd; 4];

impl Cd {
    pub const ZERO: Self = Self {
        re: Dd::ZERO,
        im: Dd::ZERO,
    };
    pub const ONE: Self = Self {
        re: Dd::ONE,
        im: Dd::ZERO,
    };

    #[inline(always)]
    pub fn new(re: impl Into<Dd>, im: impl Into<Dd>) -> Self {
        Self {
            re: re.into(),
            im: im.into(),
        }
    }
    #[inline(always)]
    pub fn real(re: impl Into<Dd>) -> Self {
        Self {
            re: re.into(),
            im: Dd::ZERO,
        }
    }
    #[inline(always)]
    pub fn from_c64(z: C64) -> Self {
        Self::new(z.re, z.im)
    }
    #[inline(always)]
    pub fn to_c64(self) -> C64 {
        C64::new(self.re.to_f64(), self.im.to_f64())
    }
    #[inline(always)]
    pub fn conj(self) -> Self {
        Self {
            re: self.re,
            im: -self.im,
        }
    }
    #[inline(always)]
    pub fn norm_sqr(self) -> Dd {
        self.re * self.re + self.im * self.im
    }
    /// The modulus, scaled by an exact power of two before squaring so that it
    /// neither underflows nor overflows for any finite parts.
    #[inline]
    pub fn abs(self) -> Dd {
        let m = self.re.abs().max(self.im.abs());
        if m.hi == 0. || !m.hi.is_finite() {
            return m;
        }
        let e = (m.hi.log2().floor() as i32).clamp(-1000, 1000);
        let (down, up) = (pow2(-e), pow2(e));
        let (x, y) = (self.re * down, self.im * down);
        (x * x + y * y).sqrt() * up
    }
    #[inline(always)]
    pub fn taxicab(self) -> Dd {
        self.re.abs() + self.im.abs()
    }
    #[inline(always)]
    pub fn is_finite(self) -> bool {
        self.re.is_finite() && self.im.is_finite()
    }
    #[inline(always)]
    pub fn is_zero(self) -> bool {
        self.re == 0. && self.im == 0.
    }
    #[inline(always)]
    pub fn inv(self) -> Self {
        let n = Dd::ONE / self.norm_sqr();
        Self {
            re: self.re * n,
            im: -self.im * n,
        }
    }
    #[inline(always)]
    pub fn scale(self, s: impl Into<Dd> + Copy) -> Self {
        let s = s.into();
        Self {
            re: self.re * s,
            im: self.im * s,
        }
    }
    /// Principal square root (branch cut on the negative real axis), as
    /// [`C64::sqrt`].
    pub fn sqrt(self) -> Self {
        if self.im == 0. {
            return if self.re >= 0. {
                Self {
                    re: self.re.sqrt(),
                    im: self.im,
                }
            } else {
                let r = (-self.re).sqrt();
                Self {
                    re: Dd::ZERO,
                    im: if self.im.hi.is_sign_negative() { -r } else { r },
                }
            };
        }
        let r = self.abs();
        if self.re >= 0. {
            let t = ((r + self.re) * 0.5).sqrt();
            Self {
                re: t,
                im: self.im / (t * 2.),
            }
        } else {
            let t = ((r - self.re) * 0.5).sqrt();
            let t = if self.im < 0. { -t } else { t };
            Self {
                re: self.im / (t * 2.),
                im: t,
            }
        }
    }
}

impl Neg for Cd {
    type Output = Self;
    #[inline(always)]
    fn neg(self) -> Self {
        Self {
            re: -self.re,
            im: -self.im,
        }
    }
}
impl Add for Cd {
    type Output = Self;
    #[inline(always)]
    fn add(self, b: Self) -> Self {
        Self {
            re: self.re + b.re,
            im: self.im + b.im,
        }
    }
}
impl Sub for Cd {
    type Output = Self;
    #[inline(always)]
    fn sub(self, b: Self) -> Self {
        Self {
            re: self.re - b.re,
            im: self.im - b.im,
        }
    }
}
impl Mul for Cd {
    type Output = Self;
    #[inline(always)]
    fn mul(self, b: Self) -> Self {
        Self {
            re: self.re * b.re - self.im * b.im,
            im: self.re * b.im + self.im * b.re,
        }
    }
}
impl Div for Cd {
    type Output = Self;
    #[inline(always)]
    fn div(self, b: Self) -> Self {
        self * b.inv()
    }
}

macro_rules! scalar_ops {
    ($s:ty) => {
        impl Add<$s> for Cd {
            type Output = Cd;
            #[inline(always)]
            fn add(self, b: $s) -> Cd {
                Cd {
                    re: self.re + b,
                    im: self.im,
                }
            }
        }
        impl Sub<$s> for Cd {
            type Output = Cd;
            #[inline(always)]
            fn sub(self, b: $s) -> Cd {
                Cd {
                    re: self.re - b,
                    im: self.im,
                }
            }
        }
        impl Mul<$s> for Cd {
            type Output = Cd;
            #[inline(always)]
            fn mul(self, b: $s) -> Cd {
                Cd {
                    re: self.re * b,
                    im: self.im * b,
                }
            }
        }
        impl Div<$s> for Cd {
            type Output = Cd;
            #[inline(always)]
            fn div(self, b: $s) -> Cd {
                let inv = Dd::ONE / Dd::from(b);
                Cd {
                    re: self.re * inv,
                    im: self.im * inv,
                }
            }
        }
        impl Add<Cd> for $s {
            type Output = Cd;
            #[inline(always)]
            fn add(self, b: Cd) -> Cd {
                b + self
            }
        }
        impl Sub<Cd> for $s {
            type Output = Cd;
            #[inline(always)]
            fn sub(self, b: Cd) -> Cd {
                (-b) + self
            }
        }
        impl Mul<Cd> for $s {
            type Output = Cd;
            #[inline(always)]
            fn mul(self, b: Cd) -> Cd {
                b * self
            }
        }
        impl Div<Cd> for $s {
            type Output = Cd;
            #[inline(always)]
            fn div(self, b: Cd) -> Cd {
                b.inv() * self
            }
        }
        assign_ops!(Cd, $s);
    };
}
scalar_ops!(Dd);
scalar_ops!(f64);
assign_ops!(Cd, Cd);

impl std::iter::Sum for Cd {
    fn sum<I: Iterator<Item = Cd>>(iter: I) -> Cd {
        iter.fold(Cd::ZERO, |a, b| a + b)
    }
}

/// Minkowski product of two complex vectors.
#[inline(always)]
pub fn mp_cc(a: &CMomD, b: &CMomD) -> Cd {
    a[0] * b[0] - a[1] * b[1] - a[2] * b[2] - a[3] * b[3]
}
/// Minkowski product of a complex and a real vector.
#[inline(always)]
pub fn mp_cr(a: &CMomD, b: &RMomD) -> Cd {
    a[0] * b[0] - a[1] * b[1] - a[2] * b[2] - a[3] * b[3]
}
/// Minkowski product of two real vectors.
#[inline(always)]
pub fn mp_rr(a: &RMomD, b: &RMomD) -> Dd {
    a[0] * b[0] - a[1] * b[1] - a[2] * b[2] - a[3] * b[3]
}

#[cfg(test)]
mod tests {
    use super::*;

    /// The two-mass Källén difference without cancellation (the
    /// expanded binary64 form understated it by 1.4e14), equal masses
    /// exactly degenerate, and scale invariance of the modulus and of δ.
    #[test]
    fn kallen_delta_and_modulus_are_stable() {
        let (a, b) = (1.002, 1.0020000000000018);
        let want = (b - a) / b;
        let got = kallen_delta(Dd::from(a), Dd::from(b), Dd::ZERO);
        assert!((got - want).abs() <= 1e-12 * want, "{got:e} {want:e}");
        assert_eq!(
            kallen_delta(Dd::from(1.), Dd::from(1.), Dd::ZERO),
            f64::MIN_POSITIVE
        );
        let d = kallen_delta(Dd::from(3.), Dd::from(-2.), Dd::from(5.));
        let s = kallen_delta(
            Dd::from(3. * pow2(-700)),
            Dd::from(-2. * pow2(-700)),
            Dd::from(5. * pow2(-700)),
        );
        assert_eq!(d, s);
        let z = Cd::new(Dd::from(3.), Dd::from(4.));
        for e in [-700, 0, 700] {
            let w = Cd::new(Dd::from(3. * pow2(e)), Dd::from(4. * pow2(e)));
            assert_eq!(w.abs().hi, z.abs().hi * pow2(e));
        }
    }

    #[test]
    fn double_double_arithmetic_is_exact_to_about_106_bits() {
        let third = Dd::ONE / 3.;
        let back = third * 3.;
        assert!((back - 1.).abs().hi < 1e-31);
        let two = Dd::from(2.);
        let r = two.sqrt();
        assert!((r * r - 2.).abs().hi < 1e-31);
        // (1 + 2^-60) - 1 is exact in double-double, lost in binary64.
        let x = Dd::ONE + 2f64.powi(-60);
        assert_eq!((x - 1.).to_f64(), 2f64.powi(-60));
        let z = Cd::new(3., -4.);
        let s = z.sqrt();
        assert!(((s * s) - z).abs().hi < 1e-30);
        assert!(((z * z.inv()) - Cd::ONE).abs().hi < 1e-31);
    }
}

/// `2^e` for `-1022 <= e <= 1023`, exactly.
#[inline]
pub(crate) fn pow2(e: i32) -> f64 {
    f64::from_bits(((e + 1023) as u64) << 52)
}

/// A triangle's normalised Källén function `sqrt|λ(a, b, c)| / max(|a|, |b|,
/// |c|)` (the square root of its Gram determinant over its scale), from
/// `λ = (a - b)² - c (2 (a + b) - c)` with `c` the smallest in modulus and
/// every value first divided by the largest: no cancellation of the expanded
/// form (two-mass triangles: `|a - b| / max`), no underflow or overflow.
/// 1 for a triangle without a scale.
pub(crate) fn kallen_delta(a: Dd, b: Dd, c: Dd) -> f64 {
    let mut v = [a, b, c];
    v.sort_by(|x, y| y.abs().hi.total_cmp(&x.abs().hi));
    let big = v[0].abs();
    if big.hi == 0. || !big.hi.is_finite() {
        return 1.;
    }
    let [a, b, c] = v.map(|x| x / big);
    let lambda = (a - b) * (a - b) - c * ((a + b) * 2. - c);
    lambda.abs().sqrt().to_f64().clamp(f64::MIN_POSITIVE, 1.)
}
