//! syn-n6-r6: exact (high-precision generic engine) Laurent coefficients vs
//! the binary64 fast engine (native oneloopmaster masters) and original Ninja.
//!
//! usage: n6r6hp CASE.json NINJA.json MASTERS.json [--percall] [--lanes 0,1,2]

#[path = "../../../../../crates/ninjars-symbolica/tests/common/ninja_reference.rs"]
#[allow(dead_code)]
mod ninja_reference;

use std::collections::HashMap;
use std::time::Instant;

use ninja_reference::{Batch, Case, MasterCall, NinjaMasters, parse_case, parse_ninja, values};
use ninjars::fast::{
    C64, FastAmplitude, FastMaster, FastMasters, FastOptions, FastResult, FastStatus,
    TensorNumerator,
};
use ninjars::{
    Amplitude, AmplitudeInput, Complex, EvaluationStatus, FourMomentum, Laurent, MasterIntegral,
    MasterProvider, MasterRequest, MasterStatus,
};
use ninjars_symbolica::{
    OneLoopNativeMasters, PolynomialNumerator, SymbolicaAmplitudeKernels, SymbolicaCuts,
    SymbolicaHighCuts, SymbolicaMasters,
};
use oneloop::NativeFloat;
use serde_json::{Value, json};
use symbolica::{
    atom::Atom,
    domains::{
        float::{DoubleFloat, Float, FloatLike, Real},
        rational::Rational,
    },
    evaluate::EvaluationDomain,
    parser::ParseSettings,
};

type R<T> = Result<T, Box<dyn std::error::Error>>;

fn to_t<T: NativeFloat>(proto: &T, x: f64) -> T {
    proto.from_rational(&Rational::try_from(x).unwrap())
}
fn ct<T: NativeFloat>(proto: &T, z: C64) -> Complex<T> {
    Complex::new(to_t(proto, z.re), to_t(proto, z.im))
}
fn c64<T: NativeFloat>(z: &Complex<T>) -> C64 {
    C64::new(z.re.to_f64(), z.im.to_f64())
}
fn parse(text: &str) -> Atom {
    Atom::parse(text, "n6r6hp", ParseSettings::default()).unwrap()
}

// ---------------------------------------------------------------- lookup
fn close(a: C64, b: C64) -> bool {
    (a - b).abs() <= 1e-12 * a.abs().max(b.abs()).max(1e-300)
}
fn find(calls: &[MasterCall], family: FastMaster, args: &[C64]) -> Option<usize> {
    calls
        .iter()
        .position(|c| c.family == family && c.args.iter().zip(args).all(|(&x, &y)| close(x, y)))
}
#[derive(Clone, Copy, Debug)]
enum Hit {
    Scaleless,
    Call(usize, usize),
}
fn lookup(calls: &[MasterCall], integral: MasterIntegral, args: &[C64]) -> Option<Hit> {
    let mu2 = *args.last()?;
    let scaleless = match integral {
        MasterIntegral::A0 | MasterIntegral::An { .. } => args[0].is_zero(),
        MasterIntegral::B0 | MasterIntegral::B1 | MasterIntegral::B11 | MasterIntegral::B111 => {
            args[..3].iter().all(|z| z.is_zero())
        }
        _ => false,
    };
    if scaleless {
        return Some(Hit::Scaleless);
    }
    let with = |mut a: Vec<C64>, m: Vec<C64>| {
        a.extend(m);
        a.push(mu2);
        a
    };
    match integral {
        MasterIntegral::D0 => {
            let (mut p, mut m) = (args[..6].to_vec(), args[6..10].to_vec());
            let mut candidates = Vec::new();
            for _ in 0..4 {
                candidates.push(with(p.clone(), m.clone()));
                candidates.push(with(
                    vec![p[3], p[2], p[1], p[0], p[4], p[5]],
                    vec![m[0], m[3], m[2], m[1]],
                ));
                p = vec![p[1], p[2], p[3], p[0], p[5], p[4]];
                m = vec![m[1], m[2], m[3], m[0]];
            }
            candidates
                .iter()
                .find_map(|a| find(calls, FastMaster::D0, a).map(|i| Hit::Call(i, 0)))
        }
        MasterIntegral::C0 => {
            let (mut p, mut m) = (args[..3].to_vec(), args[3..6].to_vec());
            let mut candidates = Vec::new();
            for _ in 0..3 {
                candidates.push(with(p.clone(), m.clone()));
                candidates.push(with(vec![p[2], p[1], p[0]], vec![m[0], m[2], m[1]]));
                p = vec![p[1], p[2], p[0]];
                m = vec![m[1], m[2], m[0]];
            }
            candidates
                .iter()
                .find_map(|a| find(calls, FastMaster::C0, a).map(|i| Hit::Call(i, 0)))
        }
        MasterIntegral::B0 | MasterIntegral::B1 | MasterIntegral::B11 | MasterIntegral::B111 => {
            let index = match integral {
                MasterIntegral::B0 => 0,
                MasterIntegral::B1 => 1,
                MasterIntegral::B11 => 2,
                _ => 3,
            };
            let direct = [args[0], args[1], args[2], mu2];
            let swapped = [args[0], args[2], args[1], mu2];
            for family in [FastMaster::BubbleRank3, FastMaster::BubbleRank2] {
                if index >= family.outputs() {
                    continue;
                }
                if let Some(i) = find(calls, family, &direct) {
                    return Some(Hit::Call(i, index));
                }
                if index == 0
                    && let Some(i) = find(calls, family, &swapped)
                {
                    return Some(Hit::Call(i, 0));
                }
            }
            None
        }
        MasterIntegral::A0 | MasterIntegral::An { rank: 0 | 1 } => {
            find(calls, FastMaster::A0, &[args[0], mu2]).map(|i| Hit::Call(i, 0))
        }
        _ => None,
    }
}

// ---------------------------------------------------------------- hybrid masters
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
enum Src {
    Hp,
    Ninja,
    Native,
    Qp,
}

/// OneLOop quadruple-precision values per Ninja call index (k = 0 output only).
type QpTable = HashMap<usize, [(String, String); 3]>;

struct Hybrid<'a, T: NativeFloat>
where
    Complex<T>: EvaluationDomain,
{
    proto: T,
    hp: &'a mut SymbolicaMasters<T>,
    cache: &'a mut HashMap<String, Laurent<Complex<T>>>,
    ninja: &'a [MasterCall],
    native: &'a [MasterCall],
    policy: &'a dyn Fn(MasterIntegral, Option<usize>) -> Src,
    log: Vec<(MasterIntegral, Vec<C64>, Option<usize>, Src)>,
    qp: &'a QpTable,
    qp_parse: &'a dyn Fn(&str) -> T,
}

impl<T: NativeFloat> Hybrid<'_, T>
where
    Complex<T>: EvaluationDomain,
{
    fn table(&self, calls: &[MasterCall], integral: MasterIntegral, args: &[C64]) -> ninjars::Result<Laurent<Complex<T>>> {
        let zero = Complex::new(self.proto.zero(), self.proto.zero());
        match lookup(calls, integral, args) {
            Some(Hit::Scaleless) => Ok([zero.clone(), zero.clone(), zero]),
            Some(Hit::Call(i, k)) => Ok(calls[i].values[k].map(|z| ct(&self.proto, z))),
            None => Err(ninjars::Error::Unsupported(format!(
                "no table master for {integral:?} {args:?}"
            ))),
        }
    }
}

impl<T: NativeFloat> MasterProvider<Complex<T>> for Hybrid<'_, T>
where
    Complex<T>: EvaluationDomain,
{
    fn evaluate(
        &mut self,
        request: MasterRequest<'_, Complex<T>>,
        output: &mut [Laurent<Complex<T>>],
    ) -> ninjars::Result<()> {
        let mut statuses = vec![MasterStatus::Success; request.rows()];
        self.evaluate_batch(request, output, &mut statuses)
    }

    fn evaluate_batch(
        &mut self,
        request: MasterRequest<'_, Complex<T>>,
        output: &mut [Laurent<Complex<T>>],
        statuses: &mut [MasterStatus],
    ) -> ninjars::Result<()> {
        let ni = request.integral.invariant_count();
        let nm = request.integral.mass_count();
        for row in 0..request.rows() {
            let inv = &request.invariants[row * ni..(row + 1) * ni];
            let ms = &request.masses_squared[row * nm..(row + 1) * nm];
            let sc = &request.scales_squared[row..row + 1];
            let args: Vec<C64> = inv.iter().chain(ms).chain(sc).map(c64).collect();
            let hit = lookup(self.ninja, request.integral, &args);
            let idx = match hit {
                Some(Hit::Call(i, _)) => Some(i),
                _ => None,
            };
            let mut src = (self.policy)(request.integral, idx);
            if src == Src::Qp {
                match hit {
                    Some(Hit::Call(i, 0)) if self.qp.contains_key(&i) => {}
                    _ => src = Src::Hp,
                }
            }
            self.log.push((request.integral, args.clone(), idx, src));
            let value = match src {
                Src::Hp => {
                    let key = format!("{:?}{:?}", request.integral, args);
                    if let Some(v) = self.cache.get(&key) {
                        v.clone()
                    } else {
                        let zero = Complex::new(self.proto.zero(), self.proto.zero());
                        let mut out = vec![[zero.clone(), zero.clone(), zero]];
                        self.hp.evaluate(
                            MasterRequest {
                                integral: request.integral,
                                invariants: inv,
                                masses_squared: ms,
                                scales_squared: sc,
                            },
                            &mut out,
                        )?;
                        self.cache.insert(key, out[0].clone());
                        out.pop().unwrap()
                    }
                }
                Src::Ninja => self.table(self.ninja, request.integral, &args)?,
                Src::Native => self.table(self.native, request.integral, &args)?,
                Src::Qp => {
                    let i = idx.unwrap();
                    let q = &self.qp[&i];
                    let f = |x: &str| (self.qp_parse)(x);
                    [
                        Complex::new(f(&q[0].0), f(&q[0].1)),
                        Complex::new(f(&q[1].0), f(&q[1].1)),
                        Complex::new(f(&q[2].0), f(&q[2].1)),
                    ]
                }
            };
            output[row] = value;
            statuses[row] = MasterStatus::Success;
        }
        Ok(())
    }
}

// ---------------------------------------------------------------- generic engine
struct Generic<T: NativeFloat>
where
    Complex<T>: EvaluationDomain + Real,
{
    proto: T,
    numerator: PolynomialNumerator<T>,
    low: SymbolicaCuts<T>,
    high: SymbolicaHighCuts<T>,
    amplitude: Amplitude<Complex<T>, SymbolicaAmplitudeKernels<T>>,
    hp: SymbolicaMasters<T>,
    cache: HashMap<String, Laurent<Complex<T>>>,
}

impl<T: NativeFloat> Generic<T>
where
    Complex<T>: EvaluationDomain + Real,
{
    fn new(n: usize, rank: usize, proto: T) -> R<Self> {
        let monomials = ninjars::fast::tensor_layout::monomials(rank);
        let names = ["q0", "q1", "q2", "q3"];
        let mut text = String::new();
        for (k, m) in monomials.iter().enumerate() {
            if k > 0 {
                text.push('+');
            }
            text.push_str(&format!("p{k}"));
            for (mu, &e) in m.iter().enumerate() {
                if e > 0 {
                    text.push_str(&format!("*{}^{e}", names[mu]));
                }
            }
        }
        let parameters: Vec<Atom> = (0..monomials.len()).map(|k| parse(&format!("p{k}"))).collect();
        let variables = ["q0", "q1", "q2", "q3", "mu2"].map(parse);
        let numerator = PolynomialNumerator::new(parse(&text), variables, parameters, proto.clone())?;
        Ok(Self {
            numerator,
            low: SymbolicaCuts::new(proto.clone())?,
            high: SymbolicaHighCuts::new(proto.clone())?,
            amplitude: SymbolicaAmplitudeKernels::new(n, rank, proto.clone())?.into_amplitude(1)?,
            hp: SymbolicaMasters::new(proto.clone()),
            cache: HashMap::new(),
            proto,
        })
    }

    /// `smat_mode`: 0 = case S-matrix, 1 = none (momenta differences).
    fn eval(
        &mut self,
        case: &Case,
        index: usize,
        ninja: &[MasterCall],
        native: &[MasterCall],
        policy: &dyn Fn(MasterIntegral, Option<usize>) -> Src,
        cut_stop: usize,
        smat_mode: usize,
        qp: &QpTable,
        qp_parse: &dyn Fn(&str) -> T,
    ) -> R<(Vec<Complex<T>>, Vec<(MasterIntegral, Vec<C64>, Option<usize>, Src)>)> {
        let n = case.n;
        let p = &self.proto;
        let point = &case.points[index];
        let coefficients: Vec<Complex<T>> = point.tensor.iter().map(|&z| ct(p, z)).collect();
        self.numerator.set_parameter_batch(&coefficients, 1)?;
        let shifts: Vec<FourMomentum<Complex<T>>> = point
            .shifts
            .iter()
            .map(|v| FourMomentum::from(v.map(|x| ct(p, C64::real(x)))))
            .collect();
        let masses: Vec<Complex<T>> = point.masses.iter().map(|&m| ct(p, m)).collect();
        let matrix: Option<Vec<Complex<T>>> = match (smat_mode, &point.smat) {
            (0, Some(s)) => Some(s.iter().map(|&x| ct(p, C64::real(x))).collect()),
            _ => None,
        };
        let sampling = point
            .smat
            .as_ref()
            .map(|s| s.iter().map(|x| x.abs()).fold(1., f64::max))
            .unwrap_or(1.)
            .max(point.masses.iter().map(|z| z.re.abs()).fold(1., f64::max));
        let _ = n;
        let mut masters = Hybrid {
            proto: p.clone(),
            hp: &mut self.hp,
            cache: &mut self.cache,
            ninja,
            native,
            policy,
            log: Vec::new(),
            qp,
            qp_parse,
        };
        self.amplitude.set_cut_stop(cut_stop);
        self.amplitude.reset_batch(1);
        self.amplitude.evaluate_batch_with_inputs(
            AmplitudeInput {
                shifts: &shifts,
                masses_squared: &masses,
                scales_squared: &[ct(p, C64::real(point.mu2))],
                sampling_scales: &[ct(p, C64::real(sampling))],
                s_matrix: matrix.as_deref(),
            },
            &mut self.numerator,
            &mut self.low,
            &mut self.high,
            &mut masters,
        )?;
        if self.amplitude.statuses()[0] != EvaluationStatus::Success {
            return Err(format!("generic engine failed: {:?}", self.amplitude.statuses()[0]).into());
        }
        Ok((self.amplitude.results()[0].to_vec(), masters.log))
    }

    /// High-precision values of one recorded (fast-layout) master call.
    fn hp_call(&mut self, call: &MasterCall) -> R<Vec<[Complex<T>; 3]>> {
        let p = self.proto.clone();
        let a: Vec<Complex<T>> = call.args.iter().map(|&z| ct(&p, z)).collect();
        let (integrals, ni, nm): (Vec<MasterIntegral>, usize, usize) = match call.family {
            FastMaster::D0 => (vec![MasterIntegral::D0], 6, 4),
            FastMaster::C0 => (vec![MasterIntegral::C0], 3, 3),
            FastMaster::BubbleRank2 => (
                vec![MasterIntegral::B0, MasterIntegral::B1, MasterIntegral::B11],
                1,
                2,
            ),
            FastMaster::BubbleRank3 => (
                vec![
                    MasterIntegral::B0,
                    MasterIntegral::B1,
                    MasterIntegral::B11,
                    MasterIntegral::B111,
                ],
                1,
                2,
            ),
            FastMaster::A0 => (vec![MasterIntegral::A0], 0, 1),
            FastMaster::BubbleRank4 => (
                vec![
                    MasterIntegral::B0,
                    MasterIntegral::B1,
                    MasterIntegral::B11,
                    MasterIntegral::B111,
                    MasterIntegral::B1111,
                ],
                1,
                2,
            ),
        };
        let mut out = Vec::new();
        for integral in integrals {
            let zero = Complex::new(p.zero(), p.zero());
            let mut o = vec![[zero.clone(), zero.clone(), zero]];
            self.hp.evaluate(
                MasterRequest {
                    integral,
                    invariants: &a[..ni],
                    masses_squared: &a[ni..ni + nm],
                    scales_squared: &a[ni + nm..ni + nm + 1],
                },
                &mut o,
            )?;
            out.push(o.pop().unwrap());
        }
        Ok(out)
    }
}

// ---------------------------------------------------------------- fast engine
struct Rec<M> {
    inner: M,
    calls: Vec<MasterCall>,
}
impl<M: FastMasters> FastMasters for Rec<M> {
    fn evaluate(
        &mut self,
        family: FastMaster,
        args: &[C64],
        out: &mut [[C64; 3]],
        ok: &mut [bool],
    ) -> ninjars::Result<()> {
        self.inner.evaluate(family, args, out, ok)?;
        let (a, w) = (family.arity(), family.outputs());
        for (row, chunk) in args.chunks_exact(a).enumerate() {
            self.calls.push(MasterCall {
                family,
                args: chunk.to_vec(),
                values: if ok[row] {
                    out[row * w..(row + 1) * w].to_vec()
                } else {
                    vec![[C64::new(f64::NAN, f64::NAN); 3]; w]
                },
            });
        }
        Ok(())
    }
}

fn fast_eval(
    amp: &mut FastAmplitude,
    case: &Case,
    index: usize,
    masters: &mut impl FastMasters,
) -> R<([C64; 5], FastStatus)> {
    let mut batch = Batch::new(case, &[index]);
    let mut tensor = std::mem::replace(&mut batch.tensor, TensorNumerator::new(case.rank, 0));
    let mut results = [FastResult::default()];
    let mut statuses = [FastStatus::NotEvaluated];
    amp.evaluate(&batch.input(), &mut tensor, masters, &mut results, &mut statuses)?;
    Ok((values(&results[0]), statuses[0]))
}

// ---------------------------------------------------------------- output helpers
fn cj(z: C64) -> Value {
    json!([z.re, z.im])
}
fn v5(v: &[C64]) -> Value {
    Value::Array(v.iter().map(|&z| cj(z)).collect())
}
fn hp_strings<T: NativeFloat + std::fmt::LowerExp>(v: &[Complex<T>]) -> Value {
    Value::Array(
        v.iter()
            .map(|z| json!([format!("{:.40e}", z.re), format!("{:.40e}", z.im), z.re.get_precision(), z.im.get_precision()]))
            .collect(),
    )
}
fn family_of(i: MasterIntegral) -> &'static str {
    match i {
        MasterIntegral::D0 => "D0",
        MasterIntegral::C0 => "C0",
        MasterIntegral::B0 | MasterIntegral::B1 | MasterIntegral::B11 | MasterIntegral::B111 | MasterIntegral::DB0 => "B",
        MasterIntegral::A0 | MasterIntegral::An { .. } => "A",
        _ => "other",
    }
}

fn main() -> R<()> {
    let child = std::thread::Builder::new()
        .stack_size(1 << 30)
        .spawn(run)
        .unwrap();
    child.join().unwrap().map_err(|e| -> Box<dyn std::error::Error> { e.to_string().into() })
}

fn run() -> Result<(), String> {
    real_run().map_err(|e| e.to_string())
}

fn fortran_number(s: &str) -> String {
    match s.split_once(['E', 'e']) {
        Some((m, e)) => format!("{m}e{}", e.parse::<i32>().unwrap()),
        None => s.to_string(),
    }
}

fn load_qp(dir: &str) -> R<HashMap<usize, QpTable>> {
    let meta: Value = serde_json::from_str(&std::fs::read_to_string(format!("{dir}/meta.json"))?)?;
    let text = std::fs::read_to_string(format!("{dir}/out_full.txt"))?;
    let qps: Vec<Vec<String>> = text
        .lines()
        .map(|l| l.split_whitespace().map(String::from).collect::<Vec<_>>())
        .filter(|v| v.first().is_some_and(|s| s == "qp"))
        .collect();
    let mut out: HashMap<usize, QpTable> = HashMap::new();
    for (m, q) in meta.as_array().unwrap().iter().zip(&qps) {
        let lane = m[0].as_u64().unwrap() as usize;
        let call = m[1].as_u64().unwrap() as usize;
        let e = |k: usize| (fortran_number(&q[1 + 2 * k]), fortran_number(&q[2 + 2 * k]));
        out.entry(lane).or_default().insert(call, [e(0), e(1), e(2)]);
    }
    Ok(out)
}

fn qp64(q: &[(String, String); 3]) -> [C64; 3] {
    let f = |s: &str| s.parse::<f64>().unwrap();
    [
        C64::new(f(&q[0].0), f(&q[0].1)),
        C64::new(f(&q[1].0), f(&q[1].1)),
        C64::new(f(&q[2].0), f(&q[2].1)),
    ]
}

fn c0_eval<T: NativeFloat>(m: &mut SymbolicaMasters<T>, proto: &T, a: &[C64]) -> R<Value>
where
    Complex<T>: EvaluationDomain,
{
    let v: Vec<Complex<T>> = a.iter().map(|&z| ct(proto, z)).collect();
    let zero = Complex::new(proto.zero(), proto.zero());
    let mut o = vec![[zero.clone(), zero.clone(), zero]];
    m.evaluate(
        MasterRequest {
            integral: MasterIntegral::C0,
            invariants: &v[..3],
            masses_squared: &v[3..6],
            scales_squared: &v[6..7],
        },
        &mut o,
    )?;
    Ok(json!({"value": cj(c64(&o[0][0])), "bits": [o[0][0].re.get_precision(), o[0][0].im.get_precision()]}))
}

fn probe(case_calls: &[(usize, usize, MasterCall)], qp_all: &HashMap<usize, QpTable>) -> R<()> {
    // All (call, labeling) argument rows.
    let mut items: Vec<(usize, usize, String, Vec<C64>)> = Vec::new();
    for (lane, ci, call) in case_calls {
        let a = &call.args;
        let (mut p, mut m) = (a[..3].to_vec(), a[3..6].to_vec());
        for r in 0..3 {
            let mut x = p.clone();
            x.extend(m.clone());
            x.push(a[6]);
            items.push((*lane, *ci, format!("rot{r}"), x));
            let mut y = vec![p[2], p[1], p[0]];
            y.extend([m[0], m[2], m[1]]);
            y.push(a[6]);
            items.push((*lane, *ci, format!("rot{r}+refl"), y));
            p = vec![p[1], p[2], p[0]];
            m = vec![m[1], m[2], m[0]];
        }
    }
    let mut rows: Vec<serde_json::Map<String, Value>> = vec![serde_json::Map::new(); items.len()];
    let mut native = OneLoopNativeMasters::new()?;
    for (k, (_, _, _, x)) in items.iter().enumerate() {
        let mut out = [[C64::ZERO; 3]];
        let mut ok = [false];
        FastMasters::evaluate(&mut native, FastMaster::C0, x, &mut out, &mut ok)?;
        rows[k].insert("native_f64".into(), json!({"ok": ok[0], "value": cj(out[0][0])}));
    }
    {
        let mut m64 = SymbolicaMasters::new(0_f64);
        for (k, (_, _, _, x)) in items.iter().enumerate() {
            rows[k].insert("symbolica_f64".into(), c0_eval(&mut m64, &0_f64, x).unwrap_or_else(|e| json!(e.to_string())));
        }
        let dd = DoubleFloat::from(0.0);
        let mut mdd = SymbolicaMasters::new(dd);
        for (k, (_, _, _, x)) in items.iter().enumerate() {
            rows[k].insert("symbolica_doublefloat".into(), c0_eval(&mut mdd, &dd, x).unwrap_or_else(|e| json!(e.to_string())));
        }
    }
    let bits_env = std::env::var("PROBE_BITS").unwrap_or("64,128,192,256,320,400,512,768,1024,2048".into());
    for bits in bits_env.split(',').map(|b| b.parse::<u32>().unwrap()) {
        let pf = Float::new(bits);
        let mut mf = SymbolicaMasters::new(pf.clone());
        for (k, (_, _, _, x)) in items.iter().enumerate() {
            rows[k].insert(format!("float{bits}"), c0_eval(&mut mf, &pf, x).unwrap_or_else(|e| json!(e.to_string())));
        }
    }
    for ((lane, ci, label, x), row) in items.iter().zip(rows) {
        let q = qp_all.get(lane).and_then(|t| t.get(ci)).map(qp64);
        println!(
            "{}",
            json!({"kind":"probe_c0","lane":lane,"call":ci,"labeling":label,
                "args": x.iter().map(|&z| cj(z)).collect::<Vec<_>>(),
                "oneloop_qp": q.map(|q| cj(q[0])), "results": Value::Object(row)})
        );
    }
    Ok(())
}

fn real_run() -> R<()> {
    let args: Vec<String> = std::env::args().skip(1).collect();
    let opt = |name: &str| args.iter().position(|a| a == name).map(|i| args[i + 1].clone());
    let percall = args.iter().any(|a| a == "--percall");
    let bits_list: Vec<u32> = opt("--bits")
        .map(|s| s.split(',').map(|x| x.parse().unwrap()).collect())
        .unwrap_or(vec![256, 400]);
    let qp_dir = opt("--qp").ok_or("--qp DIR")?;
    let files: Vec<&String> = args
        .iter()
        .enumerate()
        .filter(|(i, a)| !a.starts_with("--") && !(*i > 0 && args[i - 1].starts_with("--") && !["--percall", "--skip-dd", "--probe-only"].contains(&args[i - 1].as_str())))
        .map(|(_, a)| a)
        .collect();
    let case = parse_case(&std::fs::read_to_string(files[0])?)?;
    let ninja = parse_ninja(
        &std::fs::read_to_string(files[1])?,
        Some(&std::fs::read_to_string(files[2])?),
    )?;
    let lanes: Vec<usize> = match opt("--lanes") {
        Some(s) => s.split(',').map(|x| x.parse().unwrap()).collect(),
        None => (0..case.points.len()).collect(),
    };
    let qp_all = load_qp(&qp_dir)?;
    let empty = QpTable::new();
    let (n, rank) = (case.n, case.rank);

    if let Some(spec) = opt("--probe") {
        let list: Vec<(usize, usize, MasterCall)> = spec
            .split(',')
            .map(|s| {
                let (l, c) = s.split_once(':').unwrap();
                let (l, c): (usize, usize) = (l.parse().unwrap(), c.parse().unwrap());
                (l, c, ninja[l].calls[c].clone())
            })
            .collect();
        probe(&list, &qp_all)?;
        if args.iter().any(|a| a == "--probe-only") {
            return Ok(());
        }
    }

    // ---- binary64 fast engine
    let mut fast = FastAmplitude::new(n, rank, FastOptions::default())?;
    let mut fast_native = Vec::new();
    let mut fast_ninjam = Vec::new();
    let mut native_calls: Vec<Vec<MasterCall>> = Vec::new();
    for idx in 0..case.points.len() {
        let mut rec = Rec { inner: OneLoopNativeMasters::new()?, calls: Vec::new() };
        let (v, st) = fast_eval(&mut fast, &case, idx, &mut rec)?;
        fast_native.push((v, st));
        native_calls.push(rec.calls);
        let mut nm = NinjaMasters::new(&ninja[idx..=idx]);
        let (v2, st2) = fast_eval(&mut fast, &case, idx, &mut nm)?;
        fast_ninjam.push((v2, st2, nm.unmatched.len()));
    }
    for &idx in &lanes {
        println!(
            "{}",
            json!({"kind":"binary64","lane":idx,
                "ninja": v5(&ninja[idx].values),
                "fast_native": v5(&fast_native[idx].0), "fast_native_status": format!("{:?}", fast_native[idx].1),
                "fast_ninja_masters": v5(&fast_ninjam[idx].0), "fast_ninja_masters_status": format!("{:?}", fast_ninjam[idx].1),
                "fast_ninja_masters_unmatched": fast_ninjam[idx].2})
        );
    }

    let mut exact_calls: Vec<Option<Vec<MasterCall>>> = vec![None; case.points.len()];
    for &bits in &bits_list {
        let t0 = Instant::now();
        let mut g = Generic::new(n, rank, Float::new(bits))?;
        let prep = t0.elapsed().as_secs_f64();
        let parse_f = move |s: &str| Float::parse(s, Some(bits)).unwrap();
        for &idx in &lanes {
            let qp = qp_all.get(&idx).unwrap_or(&empty);
            // ---- master check against OneLOop quadruple precision
            let mut bad: Vec<usize> = Vec::new();
            let mut exact = Vec::new();
            let mut flagged = Vec::new();
            for (ci, call) in ninja[idx].calls.iter().enumerate() {
                let q = qp.get(&ci).map(qp64);
                let hp = g.hp_call(call);
                let mut values = call.values.clone();
                let mut is_bad = false;
                let mut info = json!(null);
                match &hp {
                    Ok(h) => {
                        for (k, hk) in h.iter().enumerate() {
                            values[k] = [c64(&hk[0]), c64(&hk[1]), c64(&hk[2])];
                        }
                        let bits_min = h
                            .iter()
                            .map(|hk| hk[0].re.get_precision().min(hk[0].im.get_precision()))
                            .min()
                            .unwrap();
                        if let Some(q) = q {
                            let s = q[0].abs().max(q[1].abs()).max(1e-300);
                            let rel = (0..3).map(|j| (values[0][j] - q[j]).abs() / s).fold(0., f64::max);
                            if rel > 1e-12 {
                                is_bad = true;
                            }
                            info = json!({"hp_vs_qp": rel, "hp_min_bits_eps0": bits_min});
                        }
                    }
                    Err(e) => {
                        is_bad = true;
                        info = json!({"hp_error": e.to_string()});
                    }
                }
                if is_bad {
                    bad.push(ci);
                    if let Some(q) = q {
                        values[0] = q;
                    }
                    flagged.push(json!({"call":ci,"family":format!("{:?}",call.family),"info":info,
                        "qp": q.map(|q| cj(q[0])), "ninja": cj(call.values[0][0])}));
                }
                exact.push(MasterCall { family: call.family, args: call.args.clone(), values });
            }
            println!("{}", json!({"kind":"master_check","bits":bits,"lane":idx,"bad_calls":bad,"flagged":flagged}));
            if exact_calls[idx].is_none() {
                exact_calls[idx] = Some(exact);
            }
            let bad_set = bad.clone();
            let hpfix = move |_: MasterIntegral, c: Option<usize>| {
                if c.is_some_and(|c| bad_set.contains(&c)) { Src::Qp } else { Src::Hp }
            };
            type P = Box<dyn Fn(MasterIntegral, Option<usize>) -> Src>;
            let mut runs: Vec<(String, usize, P)> = vec![
                ("hp_raw".into(), 0, Box::new(|_, _| Src::Hp)),
                ("hpfix".into(), 0, Box::new(hpfix.clone())),
                ("hpfix_nosmat".into(), 1, Box::new(hpfix.clone())),
                ("qp_scalar_all".into(), 0, Box::new(|_, _| Src::Qp)),
            ];
            if bits == bits_list[0] {
                runs.push(("ninja".into(), 0, Box::new(|_, _| Src::Ninja)));
                runs.push(("native".into(), 0, Box::new(|_, _| Src::Native)));
                for fam in ["D0", "C0", "B", "A"] {
                    for (lbl, src) in [("ninja", Src::Ninja), ("native", Src::Native)] {
                        let h = hpfix.clone();
                        runs.push((
                            format!("hpfix_except_{fam}_{lbl}"),
                            0,
                            Box::new(move |i: MasterIntegral, c| if family_of(i) == fam { src } else { h(i, c) }),
                        ));
                    }
                }
            }
            if let Ok(keep) = std::env::var("RUNS") {
                runs.retain(|(l, _, _)| keep.split(',').any(|k| k == l));
            }
            for (label, sm, policy) in &runs {
                let t = Instant::now();
                match g.eval(&case, idx, &ninja[idx].calls, &native_calls[idx], policy.as_ref(), 0, *sm, qp, &parse_f) {
                    Ok((v, log)) => {
                        let v64: Vec<C64> = v.iter().map(c64).collect();
                        let used_qp = log.iter().filter(|e| e.3 == Src::Qp).count();
                        println!(
                            "{}",
                            json!({"kind":"generic","carrier":"Float","bits":bits,"masters":label,"smat":*sm==0,"lane":idx,
                                "values":v5(&v64), "hp": hp_strings(&v), "qp_requests": used_qp,
                                "prep_s": prep, "eval_s": t.elapsed().as_secs_f64()})
                        );
                    }
                    Err(e) => println!("{}", json!({"kind":"generic","carrier":"Float","bits":bits,"masters":label,"lane":idx,"error":e.to_string()})),
                }
            }
            if percall && bits == bits_list[0] {
                for ci in 0..ninja[idx].calls.len() {
                    for (src_label, src) in [("ninja", Src::Ninja), ("native", Src::Native)] {
                        if std::env::var("PERCALL_SRC").is_ok_and(|s| s != src_label) { continue; }
                        let h = hpfix.clone();
                        let policy = move |i: MasterIntegral, c: Option<usize>| if c == Some(ci) { src } else { h(i, c) };
                        let (v, log) = g.eval(&case, idx, &ninja[idx].calls, &native_calls[idx], &policy, 0, 0, qp, &parse_f)?;
                        let used = log.iter().filter(|e| e.2 == Some(ci)).count();
                        let v64: Vec<C64> = v.iter().map(c64).collect();
                        println!("{}", json!({"kind":"percall","bits":bits,"lane":idx,"call":ci,"family":format!("{:?}", ninja[idx].calls[ci].family),"src":src_label,"requests_hit":used,"values":v5(&v64)}));
                    }
                }
            }
        }
    }

    // ---- binary64 fast engine with exact masters (rounded to binary64)
    for &idx in &lanes {
        if let Some(calls) = &exact_calls[idx] {
            let lane = ninja_reference::NinjaLane {
                id: ninja[idx].id.clone(),
                status: ninja[idx].status,
                values: ninja[idx].values,
                calls: calls.clone(),
            };
            let mut m = NinjaMasters::new(std::slice::from_ref(&lane));
            let (v, st) = fast_eval(&mut fast, &case, idx, &mut m)?;
            println!("{}", json!({"kind":"binary64_exact_masters","lane":idx,"fast_exact_masters":v5(&v),"status":format!("{st:?}"),"unmatched":m.unmatched.len()}));
        }
    }

    // ---- DoubleFloat generic (oneloopmaster native DoubleFloat masters)
    if !args.iter().any(|a| a == "--skip-dd") {
        let t0 = Instant::now();
        let mut g = Generic::new(n, rank, DoubleFloat::from(0.0))?;
        let prep = t0.elapsed().as_secs_f64();
        let parse_dd = |s: &str| DoubleFloat::from(s.parse::<f64>().unwrap());
        for &idx in &lanes {
            let t = Instant::now();
            let (v, _) = g.eval(&case, idx, &ninja[idx].calls, &native_calls[idx], &|_, _| Src::Hp, 0, 0, &empty, &parse_dd)?;
            let v64: Vec<C64> = v.iter().map(c64).collect();
            println!(
                "{}",
                json!({"kind":"generic","carrier":"DoubleFloat","masters":"hp","lane":idx,"values":v5(&v64),
                    "hp": hp_strings(&v), "prep_s": prep, "eval_s": t.elapsed().as_secs_f64()})
            );
        }
    }
    Ok(())
}
