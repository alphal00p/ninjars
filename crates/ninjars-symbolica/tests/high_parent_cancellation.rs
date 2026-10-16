//! Higher-cut residues must cancel from lower Laurent coefficient extraction.
//! These controls construct exact numerators independently of BoxReducer and
//! PentagonReducer, and never call those reducers.

use ninjars::{
    BubbleReducer, Complex, CutFailure, CutOutcome, FourMomentum, Numerator, TadpoleReducer,
    Topology, TriangleReducer,
};
use ninjars_symbolica::{
    PolynomialNumerator, SymbolicaBubbleKernels, SymbolicaCuts, SymbolicaTadpoleKernels,
    SymbolicaTriangleKernels,
};
use symbolica::{
    atom::{Atom, AtomCore},
    parse,
};

const RELATIVE: [[i64; 4]; 6] = [
    [0, 0, 0, 0],
    [2, 1, 0, 0],
    [3, 0, 1, 0],
    [4, 0, 0, 1],
    [5, 2, 3, 4],
    [-3, 2, -1, 5],
];
const ORIGIN: [i64; 4] = [1, -2, 1, 3];
const MASSES: [i64; 6] = [2, 3, 5, 7, 11, 13];
fn c(v: f64) -> Complex<f64> {
    Complex::new(v, 0.)
}
fn vars() -> [Atom; 5] {
    [
        parse!("q0"),
        parse!("q1"),
        parse!("q2"),
        parse!("q3"),
        parse!("mu2"),
    ]
}
fn routing() -> Vec<[i64; 4]> {
    RELATIVE
        .iter()
        .map(|p| std::array::from_fn(|i| p[i] + ORIGIN[i]))
        .collect()
}
fn symbolic_vector(p: [i64; 4]) -> FourMomentum<Atom> {
    FourMomentum::from(p.map(Atom::num))
}
fn loop_momentum() -> FourMomentum<Atom> {
    let q = vars();
    FourMomentum::from_args(q[0].clone(), q[1].clone(), q[2].clone(), q[3].clone())
}
fn denominator(p: [i64; 4], mass: i64) -> Atom {
    let shifted = loop_momentum() + symbolic_vector(p);
    shifted.dot(&shifted) - mass - &vars()[4]
}

/// Generate an exact transverse direction from the actual source routing.
/// Symbolica solves the linear constraints; no numerical cut geometry or
/// extraction formula contributes to this independently prepared numerator.
fn transverse(routing: &[[i64; 4]], partition: &[usize]) -> FourMomentum<Atom> {
    let variables = [
        parse!("source_v0"),
        parse!("source_v1"),
        parse!("source_v2"),
        parse!("source_v3"),
    ];
    let v = FourMomentum::from(variables.clone());
    let origin = symbolic_vector(routing[partition[0]]);
    let mut equations: Vec<_> = partition[1..]
        .iter()
        .map(|&line| v.dot(&(symbolic_vector(routing[line]) - origin.clone())))
        .collect();
    equations.push(&variables[0] - 1);
    let (matrix, rhs) = Atom::system_to_matrix::<u16, _, _>(&equations, &variables).unwrap();
    let components: Vec<_> = matrix
        .solve(&rhs)
        .unwrap()
        .into_vec()
        .into_iter()
        .map(|v| v.to_expression())
        .collect();
    let v = FourMomentum::from_args(
        components[0].clone(),
        components[1].clone(),
        components[2].clone(),
        components[3].clone(),
    );
    assert_ne!(v.dot(&v).expand(), Atom::Zero);
    for &line in &partition[1..] {
        assert_eq!(
            v.dot(&(symbolic_vector(routing[line]) - origin.clone()))
                .expand(),
            Atom::Zero
        );
    }
    v
}

fn box_numerator(n: usize, rank: usize, partition: &[usize]) -> Atom {
    let routing = routing();
    let v = transverse(&routing, partition);
    let u = (loop_momentum() + symbolic_vector(routing[partition[0]])).dot(&v);
    let mu = vars()[4].clone();
    let terms = [
        Atom::num(1),
        &u * 2,
        &mu * 3,
        &u * &mu * 4,
        mu.pow(2) * 5,
        &u * mu.pow(2) * 6,
    ];
    // Complementary quadratic denominators add two units of numerator rank.
    let allowed = rank - 2 * (n - 4);
    let residue = terms[..=allowed]
        .iter()
        .fold(Atom::Zero, |sum, term| sum + term);
    (0..n)
        .filter(|line| !partition.contains(line))
        .fold(residue, |value, line| {
            value * denominator(routing[line], MASSES[line])
        })
        .expand()
}
fn pentagon_numerator(n: usize, partition: &[usize]) -> Atom {
    let routing = routing();
    (0..n)
        .filter(|line| !partition.contains(line))
        .fold(parse!("7*mu2"), |value, line| {
            value * denominator(routing[line], MASSES[line])
        })
        .expand()
}

fn near_zero(value: Complex<f64>, context: &str) {
    assert!(
        value.re.abs() < 1e-9 && value.im.abs() < 1e-9,
        "{context}: residual coefficient {value}"
    );
}

fn check_lower_stages(n: usize, rank: usize, expression: Atom, label: &str, specialized: bool) {
    check_lower_stages_mode(n, rank, expression, label, specialized, false);
}

fn check_lower_stages_mode(
    n: usize,
    rank: usize,
    expression: Atom,
    label: &str,
    specialized: bool,
    native_rank: bool,
) {
    #[cfg(not(feature = "native-kernels"))]
    assert!(!native_rank);
    let topology = Topology::new(n, rank).unwrap();
    let lanes = 2;
    let routing = routing();
    let shifts: Vec<_> = (0..lanes)
        .flat_map(|_| {
            routing[..n]
                .iter()
                .map(|p| FourMomentum::from(p.map(|v| c(v as f64))))
        })
        .collect();
    let masses: Vec<_> = (0..lanes)
        .flat_map(|_| MASSES[..n].iter().map(|&m| c(m as f64)))
        .collect();
    let mut numerator =
        PolynomialNumerator::new(parse!("a") * expression, vars(), vec![parse!("a")], 0.).unwrap();
    assert!(
        numerator.rank() <= rank,
        "{label}: rank {} > {rank}",
        numerator.rank()
    );
    numerator
        .set_parameter_batch(&[c(1.), c(-2.)], lanes)
        .unwrap();
    let mut geometry = SymbolicaCuts::new(0.).unwrap();
    let mut triangle =
        TriangleReducer::new(SymbolicaTriangleKernels::new(n, rank, 0.).unwrap(), c(0.)).unwrap();
    let mut triangles =
        vec![CutOutcome::Unstable(CutFailure::NonFiniteResult); triangle.cut_count() * lanes];
    triangle
        .evaluate_batch(
            &shifts,
            &masses,
            lanes,
            &mut numerator,
            &mut geometry,
            &mut triangles,
        )
        .unwrap();
    for outcome in &triangles {
        let CutOutcome::Stable(residue) = outcome else {
            panic!("{label} n={n},rank={rank}: {outcome:?}")
        };
        for (index, &value) in residue.coefficients.iter().enumerate() {
            near_zero(
                value,
                &format!(
                    "{label} n={n},rank={rank},triangle={},lane={},c{index}",
                    residue.cut_index, residue.id.lane
                ),
            );
        }
    }
    let bubble_kernels = SymbolicaBubbleKernels::new(n, rank, 0.).unwrap();
    let bubble_kernels = if specialized {
        bubble_kernels.with_rank_specialized_corrections().unwrap()
    } else {
        bubble_kernels
    };
    #[cfg(feature = "native-kernels")]
    let bubble_kernels = if native_rank {
        bubble_kernels
            .with_native_rank_specialized_corrections()
            .unwrap()
    } else {
        bubble_kernels
    };
    let mut bubble = BubbleReducer::new(bubble_kernels, c(0.)).unwrap();
    let mut bubbles =
        vec![CutOutcome::Unstable(CutFailure::NonFiniteResult); bubble.cut_count() * lanes];
    bubble
        .evaluate_batch(
            &shifts,
            &masses,
            lanes,
            &topology,
            &triangles,
            &mut numerator,
            &mut geometry,
            &mut bubbles,
        )
        .unwrap();
    for outcome in &bubbles {
        let CutOutcome::Stable(residue) = outcome else {
            panic!("{label} n={n},rank={rank}: {outcome:?}")
        };
        for (index, &value) in residue.coefficients.iter().enumerate() {
            near_zero(
                value,
                &format!(
                    "{label} n={n},rank={rank},bubble={},lane={},c{index}",
                    residue.cut_index, residue.id.lane
                ),
            );
        }
    }
    let tadpole_kernels = SymbolicaTadpoleKernels::new(n, rank, 0.).unwrap();
    let tadpole_kernels = if specialized {
        tadpole_kernels.with_rank_specialized_corrections().unwrap()
    } else {
        tadpole_kernels
    };
    #[cfg(feature = "native-kernels")]
    let tadpole_kernels = if native_rank {
        tadpole_kernels
            .with_native_rank_specialized_corrections()
            .unwrap()
    } else {
        tadpole_kernels
    };
    let mut tadpole = TadpoleReducer::new(tadpole_kernels, c(0.)).unwrap();
    let mut tadpoles =
        vec![CutOutcome::Unstable(CutFailure::NonFiniteResult); tadpole.cut_count() * lanes];
    tadpole
        .evaluate_batch(
            &shifts,
            &masses,
            lanes,
            &topology,
            &triangles,
            &topology,
            &bubbles,
            &mut numerator,
            &mut geometry,
            &mut tadpoles,
        )
        .unwrap();
    for outcome in &tadpoles {
        let CutOutcome::Stable(residue) = outcome else {
            panic!("{label} n={n},rank={rank}: {outcome:?}")
        };
        for (index, (&value, &computed)) in residue
            .coefficients
            .iter()
            .zip(&residue.computed)
            .enumerate()
        {
            if computed {
                near_zero(
                    value,
                    &format!(
                        "{label} n={n},rank={rank},tadpole={},lane={},c{index}",
                        residue.cut_index, residue.id.lane
                    ),
                );
            }
        }
    }
}

#[test]
fn exact_box_residues_cancel_from_all_lower_stages_without_parent_subtraction() {
    for (n, partition) in [
        (4, vec![0, 1, 2, 3]),
        (5, vec![1, 2, 3, 4]),
        (6, vec![1, 2, 4, 5]),
    ] {
        for rank in [n, n + 1] {
            check_lower_stages(n, rank, box_numerator(n, rank, &partition), "box", false);
        }
    }
}

#[test]
fn exact_pentagon_residues_cancel_from_all_lower_stages_without_parent_subtraction() {
    for (n, partition) in [(5, vec![0, 1, 2, 3, 4]), (6, vec![1, 2, 3, 4, 5])] {
        for rank in [n, n + 1] {
            check_lower_stages(
                n,
                rank,
                pentagon_numerator(n, &partition),
                "pentagon",
                false,
            );
        }
    }
}

#[test]
fn rank_specialization_preserves_independent_exact_n5_r4_parent_cancellation() {
    for specialized in [false, true] {
        check_lower_stages(
            5,
            4,
            box_numerator(5, 4, &[1, 2, 3, 4]),
            "n5/r4 exact box",
            specialized,
        );
        check_lower_stages(
            5,
            4,
            pentagon_numerator(5, &[0, 1, 2, 3, 4]),
            "n5/r4 exact pentagon",
            specialized,
        );
    }
}

#[cfg(feature = "native-kernels")]
#[test]
fn native_rank_specialization_preserves_independent_exact_n5_r4_parent_cancellation() {
    check_lower_stages_mode(
        5,
        4,
        box_numerator(5, 4, &[1, 2, 3, 4]),
        "native n5/r4 exact box",
        true,
        true,
    );
    check_lower_stages_mode(
        5,
        4,
        pentagon_numerator(5, &[0, 1, 2, 3, 4]),
        "native n5/r4 exact pentagon",
        true,
        true,
    );
}
