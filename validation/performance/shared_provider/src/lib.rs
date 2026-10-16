//! Validation-only, dimensioned C ABI around the production Rust numerator.
//! No timing claims: packed buffers are retained; some request/provider work
//! still allocates, so this is not an allocation-free path.

use ninjars::{
    Complex, ExpansionKind, ExpansionRequest, FourMomentum, Numerator, NumeratorRequest, SampleId,
};
use ninjars_symbolica::PolynomialNumerator;
mod foreign;
mod masters;
mod session;

use std::{
    cell::RefCell,
    panic::{AssertUnwindSafe, catch_unwind},
    slice,
};
use symbolica::{atom::Atom, parse};

#[repr(C)]
#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub struct Pair {
    pub re: f64,
    pub im: f64,
}
impl From<Pair> for Complex<f64> {
    fn from(v: Pair) -> Self {
        Self::new(v.re, v.im)
    }
}
impl From<Complex<f64>> for Pair {
    fn from(v: Complex<f64>) -> Self {
        Self { re: v.re, im: v.im }
    }
}
#[repr(C)]
#[derive(Clone, Copy, Debug)]
pub struct Identity {
    pub lane: u64,
    pub sample: u64,
}
pub struct Provider {
    numerator: PolynomialNumerator<f64>,
    n: usize,
    parameters: usize,
    scratch: CallScratch,
}
#[derive(Default)]
struct CallScratch {
    kinds: Vec<(u32, Vec<i32>, ExpansionKind)>,
    ids: Vec<SampleId>,
    partition: Vec<usize>,
    data: Vec<Complex<f64>>,
    vectors: Vec<FourMomentum<Complex<f64>>>,
    output: Vec<Complex<f64>>,
    packed: Vec<Pair>,
    bindings: Vec<Complex<f64>>,
}

thread_local! { static LAST_ERROR: RefCell<String> = const {RefCell::new(String::new())}; }
const SUCCESS: i32 = 0;
const ERROR: i32 = 1;
const PANIC: i32 = 2;
type Result<T> = std::result::Result<T, String>;

fn guard(f: impl FnOnce() -> Result<()>) -> i32 {
    let result = catch_unwind(AssertUnwindSafe(f));
    let (code, message) = match result {
        Ok(Ok(())) => (SUCCESS, String::new()),
        Ok(Err(e)) => (ERROR, e),
        Err(_) => (
            PANIC,
            "caught Rust panic; no unwinding crossed the C ABI".into(),
        ),
    };
    LAST_ERROR.with(|error| *error.borrow_mut() = message);
    code
}
fn variables() -> [Atom; 5] {
    [
        parse!("q0"),
        parse!("q1"),
        parse!("q2"),
        parse!("q3"),
        parse!("mu2"),
    ]
}
fn fixture(number: u32) -> Result<Provider> {
    let (n, expression, parameters) = match number {
        0 => (
            4,
            parse!("a*q0^2+mu2*q1+b*mu2^2+(a-b)*q3+5"),
            vec![parse!("a"), parse!("b")],
        ),
        1 => (
            4,
            parse!("a*q0^2+mu2*q1+b*mu2^2*q0+(a-b)*q3+5"),
            vec![parse!("a"), parse!("b")],
        ),
        3 => (2, parse!("a*q0^2+b*mu2+5"), vec![parse!("a"), parse!("b")]),
        4 => (
            2,
            parse!("a*q0^3+b*mu2*q0+5"),
            vec![parse!("a"), parse!("b")],
        ),
        2 => (3, parse!("a*mu2"), vec![parse!("a")]),
        5 => (2, parse!("q0^2"), vec![]),
        6 => (2, parse!("mu2"), vec![]),
        _ => return Err("unknown fixed probe fixture".into()),
    };
    let parameter_count = parameters.len();
    Ok(Provider {
        numerator: PolynomialNumerator::new(expression, variables(), parameters, 0.)
            .map_err(|e| e.to_string())?,
        n,
        parameters: parameter_count,
        scratch: CallScratch::default(),
    })
}
fn dimensions(actual: usize, rows: usize, columns: usize, label: &str) -> Result<()> {
    let expected = rows
        .checked_mul(columns)
        .ok_or_else(|| format!("{label} size overflow"))?;
    if actual != expected {
        return Err(format!("{label}: {actual} cells, expected {expected}"));
    }
    Ok(())
}
unsafe fn read<'a, T>(ptr: *const T, len: usize) -> Result<&'a [T]> {
    if len == 0 {
        return Ok(&[]);
    }
    if ptr.is_null() || !ptr.is_aligned() || len > isize::MAX as usize / std::mem::size_of::<T>() {
        return Err("invalid input pointer or byte length".into());
    }
    // SAFETY: the ABI caller guarantees this checked nonnull/aligned allocation
    // is readable for len elements and remains live until this call returns.
    Ok(unsafe { slice::from_raw_parts(ptr, len) })
}
unsafe fn write<T: Copy>(ptr: *mut T, len: usize, values: &[T]) -> Result<()> {
    if len != values.len() {
        return Err("output length mismatch".into());
    }
    if len == 0 {
        return Ok(());
    }
    if ptr.is_null() || !ptr.is_aligned() || len > isize::MAX as usize / std::mem::size_of::<T>() {
        return Err("invalid output pointer or byte length".into());
    }
    // SAFETY: all source work has finished; the ABI caller guarantees writable
    // storage, disjoint from the live provider, for the declared output length.
    unsafe { slice::from_raw_parts_mut(ptr, len) }.copy_from_slice(values);
    Ok(())
}
unsafe fn provider<'a>(handle: *mut Provider) -> Result<&'a mut Provider> {
    if handle.is_null() || !handle.is_aligned() {
        return Err("invalid provider handle".into());
    }
    // SAFETY: only sp_create handles are accepted by the documented ABI. Calls
    // must be serialized and sp_destroy invalidates the handle exactly once.
    Ok(unsafe { &mut *handle })
}
fn kind(which: u32, powers: &[i32]) -> Result<Option<ExpansionKind>> {
    Ok(match which {
        0 if powers.is_empty() => None,
        1 => Some(ExpansionKind::Mu {
            powers: powers.to_vec(),
        }),
        2 if powers.len().is_multiple_of(2) => Some(ExpansionKind::T3 {
            powers: powers
                .as_chunks::<2>()
                .0
                .iter()
                .map(|p| {
                    Ok((
                        p[0],
                        u16::try_from(p[1]).map_err(|_| "invalid mu exponent")?,
                    ))
                })
                .collect::<Result<_>>()?,
        }),
        3 if powers.len().is_multiple_of(3) => Some(ExpansionKind::T2 {
            powers: powers
                .as_chunks::<3>()
                .0
                .iter()
                .map(|p| {
                    Ok((
                        p[0],
                        u16::try_from(p[1]).map_err(|_| "invalid x exponent")?,
                        u16::try_from(p[2]).map_err(|_| "invalid mu exponent")?,
                    ))
                })
                .collect::<Result<_>>()?,
        }),
        _ => return Err("invalid callback kind or exponent tuple length".into()),
    })
}
fn call<'a>(
    provider: &'a mut Provider,
    which: u32,
    powers: &[i32],
    ids: &[Identity],
    data: &[Pair],
    partition: &[u8],
    output_len: usize,
) -> Result<&'a [Pair]> {
    if partition
        .iter()
        .any(|&index| usize::from(index) >= provider.n)
        || partition
            .iter()
            .enumerate()
            .any(|(i, p)| partition[..i].contains(p))
    {
        return Err("invalid cut partition".into());
    }
    let columns = match which {
        0 => 5,
        1 => 8,
        2 => 13,
        3 => 19,
        _ => return Err("invalid callback kind".into()),
    };
    let k = if which == 0 {
        if !powers.is_empty() {
            return Err("direct callback has powers".into());
        }
        None
    } else {
        let index = provider
            .scratch
            .kinds
            .iter()
            .position(|(w, p, _)| *w == which && p == powers);
        Some(if let Some(index) = index {
            index
        } else {
            let parsed = kind(which, powers)?.ok_or("missing expansion kind")?;
            provider
                .scratch
                .kinds
                .push((which, powers.to_vec(), parsed));
            provider.scratch.kinds.len() - 1
        })
    };
    let s = &mut provider.scratch;
    let kind = k.map(|index| &s.kinds[index].2);
    dimensions(data.len(), ids.len(), columns, "callback inputs")?;
    dimensions(
        output_len,
        ids.len(),
        kind.map_or(1, ExpansionKind::len),
        "callback outputs",
    )?;
    if data.iter().any(|v| !v.re.is_finite() || !v.im.is_finite()) {
        return Err("nonfinite callback input".into());
    }
    s.ids.clear();
    for id in ids {
        s.ids.push(SampleId {
            lane: usize::try_from(id.lane).map_err(|_| "lane overflow")?,
            sample: usize::try_from(id.sample).map_err(|_| "sample overflow")?,
        });
    }
    s.partition.clear();
    s.partition
        .extend(partition.iter().map(|&v| usize::from(v)));
    s.data.clear();
    s.data.extend(data.iter().copied().map(Complex::from));
    s.output.resize(output_len, Complex::new(0., 0.));
    let nv = match which {
        0 => 1,
        1 => 2,
        2 => 3,
        _ => 4,
    };
    s.vectors.clear();
    s.vectors.extend(s.data.chunks(columns).flat_map(|row| {
        row[..nv * 4]
            .as_chunks::<4>()
            .0
            .iter()
            .map(|v| FourMomentum::from_args(v[0], v[1], v[2], v[3]))
    }));
    if let Some(kind) = kind {
        let request = |row: usize| ExpansionRequest {
            id: s.ids[row],
            partition: &s.partition,
            vectors: &s.vectors[row * nv..(row + 1) * nv],
            parameters: &s.data[row * columns + nv * 4..(row + 1) * columns],
        };
        if s.ids.len() == 1 {
            provider
                .numerator
                .expansion_batch(kind, &[request(0)], &mut s.output)
                .map_err(|e| e.to_string())?;
        } else {
            let requests = (0..s.ids.len()).map(request).collect::<Vec<_>>();
            provider
                .numerator
                .expansion_batch(kind, &requests, &mut s.output)
                .map_err(|e| e.to_string())?;
        }
    } else {
        let request = |row: usize| NumeratorRequest {
            id: s.ids[row],
            partition: &s.partition,
            q: &s.vectors[row],
            mu_squared: &s.data[row * columns + 4],
        };
        if s.ids.len() == 1 {
            provider
                .numerator
                .evaluate_batch(&[request(0)], &mut s.output)
                .map_err(|e| e.to_string())?;
        } else {
            let requests = (0..s.ids.len()).map(request).collect::<Vec<_>>();
            provider
                .numerator
                .evaluate_batch(&requests, &mut s.output)
                .map_err(|e| e.to_string())?;
        }
    }
    if s.output
        .iter()
        .any(|v| !v.re.is_finite() || !v.im.is_finite())
    {
        return Err("nonfinite numerator output".into());
    }
    s.packed.clear();
    s.packed.extend(s.output.iter().copied().map(Pair::from));
    Ok(&s.packed)
}

/// # Safety
/// `out` must be a writable, aligned pointer slot, with no concurrent access.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn sp_create(number: u32, out: *mut *mut Provider) -> i32 {
    guard(|| {
        if out.is_null() || !out.is_aligned() {
            return Err("invalid handle output".into());
        }
        let handle = Box::into_raw(Box::new(fixture(number)?));
        unsafe { out.write(handle) };
        Ok(())
    })
}
/// # Safety
/// `handle` must be live and exclusively borrowed; outputs are writable slots.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn sp_info(handle: *mut Provider, out: *mut u64, len: usize) -> i32 {
    guard(|| {
        let provider = unsafe { provider(handle) }?;
        unsafe {
            write(
                out,
                len,
                &[
                    provider.n as u64,
                    provider.numerator.rank() as u64,
                    provider.parameters as u64,
                ],
            )
        }
    })
}
/// # Safety
/// `handle` must be live/exclusive; data describes a readable array of `len` pairs.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn sp_bind(
    handle: *mut Provider,
    data: *const Pair,
    len: usize,
    rows: usize,
) -> i32 {
    guard(|| {
        let provider = unsafe { provider(handle) }?;
        dimensions(len, rows, provider.parameters, "parameter batch")?;
        let data = unsafe { read(data, len) }?;
        if data.iter().any(|v| !v.re.is_finite() || !v.im.is_finite()) {
            return Err("nonfinite parameter".into());
        }
        provider.scratch.bindings.clear();
        provider
            .scratch
            .bindings
            .extend(data.iter().copied().map(Complex::from));
        provider
            .numerator
            .set_parameter_batch(&provider.scratch.bindings, rows)
            .map_err(|e| e.to_string())
    })
}
/// # Safety
/// All arrays must remain valid for their explicit lengths; output is writable
/// and disjoint from the provider. Calls on one handle must not overlap.
#[unsafe(no_mangle)]
#[inline(never)]
pub unsafe extern "C" fn sp_call(
    handle: *mut Provider,
    which: u32,
    powers: *const i32,
    powers_len: usize,
    ids: *const Identity,
    rows: usize,
    data: *const Pair,
    data_len: usize,
    partition: *const u8,
    partition_len: usize,
    out: *mut Pair,
    out_len: usize,
) -> i32 {
    guard(|| {
        let provider = unsafe { provider(handle) }?;
        let powers = unsafe { read(powers, powers_len) }?;
        let ids = unsafe { read(ids, rows) }?;
        let data = unsafe { read(data, data_len) }?;
        let partition = unsafe { read(partition, partition_len) }?;
        let values = call(provider, which, powers, ids, data, partition, out_len)?;
        unsafe { write(out, out_len, values) }
    })
}
/// # Safety
/// A handle created by sp_create must be destroyed exactly once, with no calls
/// or borrows in flight. Null is accepted as a no-op.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn sp_destroy(handle: *mut Provider) -> i32 {
    guard(|| {
        if !handle.is_null() {
            if !handle.is_aligned() {
                return Err("invalid provider handle".into());
            }
            drop(unsafe { Box::from_raw(handle) });
        }
        Ok(())
    })
}
/// Copy the thread-local error message, including a NUL if capacity is nonzero.
/// # Safety
/// `out` must be writable for capacity bytes (or null when capacity is zero).
#[unsafe(no_mangle)]
pub unsafe extern "C" fn sp_error(out: *mut u8, capacity: usize) -> usize {
    catch_unwind(AssertUnwindSafe(|| {
        LAST_ERROR.with(|message| {
            let message = message.borrow();
            let needed = message.len() + 1;
            if capacity != 0 && !out.is_null() {
                let n = message.len().min(capacity - 1);
                unsafe {
                    std::ptr::copy_nonoverlapping(message.as_ptr(), out, n);
                    out.add(n).write(0)
                };
            }
            needed
        })
    }))
    .unwrap_or(0)
}
/// Deliberate validation-only panic to verify the no-unwind boundary.
#[unsafe(no_mangle)]
pub extern "C" fn sp_probe_panic() -> i32 {
    guard(|| panic!("intentional shared-provider boundary control"))
}

// Independent safe-Rust control for the fixed probe data and the original
// callback order. Explicit exponent lists do not share the C++ order builder.
fn control(number: u32, which: u32, lane: usize) -> Result<Vec<Pair>> {
    let mut provider = fixture(number)?;
    let params = [
        Complex::new(2., 1.),
        Complex::new(3., -1.),
        Complex::new(-1., 2.),
        Complex::new(1., 1.),
    ];
    provider
        .numerator
        .set_parameter_batch(&params, 2)
        .map_err(|e| e.to_string())?;
    let vectors = control_vectors();
    let beta = [
        Complex::new(2., -1.),
        Complex::new(3., 1.),
        Complex::new(5., 0.),
    ];
    let mu = Complex::new(7., 2.);
    let id = SampleId { lane, sample: 42 };
    let mut output;
    if which == 0 {
        output = vec![Complex::new(0., 0.)];
        provider
            .numerator
            .evaluate_batch(
                &[NumeratorRequest {
                    id,
                    q: &vectors[0],
                    mu_squared: &mu,
                    partition: &[],
                }],
                &mut output,
            )
            .map_err(|e| e.to_string())?;
    } else {
        let (mut kind, nv, np) = match which {
            1 => (
                ExpansionKind::Mu {
                    powers: if number == 0 { vec![4] } else { vec![5, 4] },
                },
                2,
                0,
            ),
            2 => (
                ExpansionKind::T3 {
                    powers: vec![
                        (4, 0),
                        (3, 0),
                        (2, 0),
                        (2, 1),
                        (1, 0),
                        (1, 1),
                        (0, 0),
                        (0, 1),
                        (0, 2),
                    ],
                },
                3,
                1,
            ),
            3 => (
                ExpansionKind::T2 {
                    powers: vec![
                        (4, 0, 0),
                        (3, 0, 0),
                        (3, 1, 0),
                        (2, 0, 0),
                        (2, 0, 1),
                        (2, 1, 0),
                        (2, 2, 0),
                        (1, 0, 0),
                        (1, 0, 1),
                        (1, 1, 0),
                        (1, 1, 1),
                        (1, 2, 0),
                        (1, 3, 0),
                    ],
                },
                4,
                3,
            ),
            _ => return Err("invalid control kind".into()),
        };
        if number == 1 {
            match &mut kind {
                ExpansionKind::T3 { powers } => {
                    for (t, _) in powers {
                        *t += 1;
                    }
                }
                ExpansionKind::T2 { powers } => {
                    for (t, _, _) in powers {
                        *t += 1;
                    }
                }
                ExpansionKind::Mu { .. } => {}
            }
        }
        let mut vectors = vectors;
        if which == 1 && number == 0 {
            vectors[1] = FourMomentum::from_args(
                Complex::new(0., 0.),
                Complex::new(0., 0.),
                Complex::new(0., 0.),
                Complex::new(0., 0.),
            );
        }
        output = vec![Complex::new(0., 0.); kind.len()];
        provider
            .numerator
            .expansion_batch(
                &kind,
                &[ExpansionRequest {
                    id,
                    partition: &[],
                    vectors: &vectors[..nv],
                    parameters: &beta[..np],
                }],
                &mut output,
            )
            .map_err(|e| e.to_string())?;
    }
    Ok(output.into_iter().map(Pair::from).collect())
}
fn control_vectors() -> [FourMomentum<Complex<f64>>; 4] {
    [
        [(1., 1.), (2., 0.), (3., -1.), (4., 0.)],
        [(2., 0.), (-1., 1.), (3., 0.), (-2., 0.)],
        [(3., 0.), (1., 0.), (-2., 1.), (4., 0.)],
        [(1., 0.), (2., -1.), (-3., 0.), (-1., 0.)],
    ]
    .map(|v| {
        let [a, b, c, d] = v.map(|(re, im)| Complex::new(re, im));
        FourMomentum::from_args(a, b, c, d)
    })
}
/// # Safety
/// `out` is writable for len pairs. Only fixed fixtures 0 and 1 are supported.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn sp_control(
    number: u32,
    which: u32,
    lane: usize,
    out: *mut Pair,
    len: usize,
) -> i32 {
    guard(|| {
        if number > 1 {
            return Err("control requires fixture0 or1".into());
        }
        let values = control(number, which, lane)?;
        unsafe { write(out, len, &values) }
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn batched_direct_matches_safe_calls_and_errors_preserve_output() {
        let mut handle = std::ptr::null_mut();
        assert_eq!(unsafe { sp_create(0, &mut handle) }, SUCCESS);
        let params = [
            Pair { re: 2., im: 1. },
            Pair { re: 3., im: -1. },
            Pair { re: -1., im: 2. },
            Pair { re: 1., im: 1. },
        ];
        assert_eq!(
            unsafe { sp_bind(handle, params.as_ptr(), params.len(), 2) },
            SUCCESS
        );
        let ids = [
            Identity {
                lane: 1,
                sample: 42,
            },
            Identity {
                lane: 0,
                sample: 42,
            },
            Identity {
                lane: 1,
                sample: 42,
            },
        ];
        let mut row = control_vectors()[0]
            .components()
            .map(|v| Pair::from(*v))
            .to_vec();
        row.push(Pair { re: 7., im: 2. });
        let data = row.repeat(3);
        let mut out = [Pair::default(); 3];
        assert_eq!(
            unsafe {
                sp_call(
                    handle,
                    0,
                    std::ptr::null(),
                    0,
                    ids.as_ptr(),
                    3,
                    data.as_ptr(),
                    data.len(),
                    std::ptr::null(),
                    0,
                    out.as_mut_ptr(),
                    3,
                )
            },
            SUCCESS
        );
        assert_eq!(
            out,
            [
                control(0, 0, 1).unwrap()[0],
                control(0, 0, 0).unwrap()[0],
                control(0, 0, 1).unwrap()[0]
            ]
        );
        let saved = out;
        assert_eq!(
            unsafe {
                sp_call(
                    handle,
                    0,
                    std::ptr::null(),
                    0,
                    ids.as_ptr(),
                    3,
                    data.as_ptr(),
                    data.len() - 1,
                    std::ptr::null(),
                    0,
                    out.as_mut_ptr(),
                    3,
                )
            },
            ERROR
        );
        assert_eq!(out, saved);
        let invalid_ids = [Identity {
            lane: 99,
            sample: 0,
        }; 3];
        assert_eq!(
            unsafe {
                sp_call(
                    handle,
                    0,
                    std::ptr::null(),
                    0,
                    invalid_ids.as_ptr(),
                    3,
                    data.as_ptr(),
                    data.len(),
                    std::ptr::null(),
                    0,
                    out.as_mut_ptr(),
                    3,
                )
            },
            ERROR
        );
        assert_eq!(out, saved);
        assert_eq!(
            unsafe {
                sp_call(
                    handle,
                    0,
                    std::ptr::null(),
                    0,
                    std::ptr::null(),
                    0,
                    std::ptr::null(),
                    0,
                    std::ptr::null(),
                    0,
                    std::ptr::null_mut(),
                    0,
                )
            },
            SUCCESS
        );
        assert_eq!(sp_probe_panic(), PANIC);
        assert_eq!(
            unsafe {
                sp_call(
                    handle,
                    0,
                    std::ptr::null(),
                    0,
                    ids.as_ptr(),
                    3,
                    data.as_ptr(),
                    data.len(),
                    std::ptr::null(),
                    0,
                    out.as_mut_ptr(),
                    3,
                )
            },
            SUCCESS
        );
        assert_eq!(out, saved);
        assert_eq!(unsafe { sp_destroy(handle) }, SUCCESS);
    }
    #[test]
    fn batched_laurent_responses_match_safe_rust_for_all_orders() {
        for number in 0..=1 {
            let mut provider = fixture(number).unwrap();
            provider
                .numerator
                .set_parameter_batch(
                    &[
                        Complex::new(2., 1.),
                        Complex::new(3., -1.),
                        Complex::new(-1., 2.),
                        Complex::new(1., 1.),
                    ],
                    2,
                )
                .unwrap();
            let ids = [
                Identity {
                    lane: 1,
                    sample: 42,
                },
                Identity {
                    lane: 0,
                    sample: 42,
                },
                Identity {
                    lane: 1,
                    sample: 42,
                },
            ];
            let rank = if number == 0 { 4 } else { 5 };
            let powers = [
                if number == 0 { vec![4] } else { vec![5, 4] },
                vec![
                    rank,
                    0,
                    rank - 1,
                    0,
                    rank - 2,
                    0,
                    rank - 2,
                    1,
                    rank - 3,
                    0,
                    rank - 3,
                    1,
                    rank - 4,
                    0,
                    rank - 4,
                    1,
                    rank - 4,
                    2,
                ],
                vec![
                    rank,
                    0,
                    0,
                    rank - 1,
                    0,
                    0,
                    rank - 1,
                    1,
                    0,
                    rank - 2,
                    0,
                    0,
                    rank - 2,
                    0,
                    1,
                    rank - 2,
                    1,
                    0,
                    rank - 2,
                    2,
                    0,
                    rank - 3,
                    0,
                    0,
                    rank - 3,
                    0,
                    1,
                    rank - 3,
                    1,
                    0,
                    rank - 3,
                    1,
                    1,
                    rank - 3,
                    2,
                    0,
                    rank - 3,
                    3,
                    0,
                ],
            ];
            for which in 1..=3 {
                let mut vectors = control_vectors();
                if which == 1 && number == 0 {
                    vectors[1] = FourMomentum::from_args(
                        Complex::new(0., 0.),
                        Complex::new(0., 0.),
                        Complex::new(0., 0.),
                        Complex::new(0., 0.),
                    );
                }
                let nv = which as usize + 1;
                let mut data = vectors[..nv]
                    .iter()
                    .flat_map(|v| v.components().map(|v| Pair::from(*v)))
                    .collect::<Vec<_>>();
                if which == 2 {
                    data.push(Pair { re: 2., im: -1. });
                }
                if which == 3 {
                    data.extend([
                        Pair { re: 2., im: -1. },
                        Pair { re: 3., im: 1. },
                        Pair { re: 5., im: 0. },
                    ]);
                }
                let columns = control(number, which, 0).unwrap().len();
                let actual = call(
                    &mut provider,
                    which,
                    &powers[which as usize - 1],
                    &ids,
                    &data.repeat(3),
                    &[],
                    columns * 3,
                )
                .unwrap();
                let expected = [1, 0, 1]
                    .into_iter()
                    .flat_map(|lane| control(number, which, lane).unwrap())
                    .collect::<Vec<_>>();
                assert_eq!(actual, expected);
                assert!(
                    call(
                        &mut provider,
                        which,
                        &powers[which as usize - 1],
                        &[],
                        &[],
                        &[],
                        0
                    )
                    .unwrap()
                    .is_empty()
                );
            }
        }
    }
}
