//! Fixed two-case inventory of existing one-line LaurentDivision plans.
//! All algebra comes from the owning plan and Symbolica, not a new recurrence.

use super::*;

#[derive(Default)]
struct Budget(usize);

impl Budget {
    fn include(&mut self, label: &str, expressions: &[Atom]) -> ProbeResult<()> {
        self.0 += expressions
            .iter()
            .map(|e| e.to_string().len())
            .sum::<usize>();
        if self.0 > MAX_COMPOSED_BYTES {
            return Err(format!(
                "{label}: shared expression budget exceeded: {} bytes",
                self.0
            )
            .into());
        }
        Ok(())
    }
}

fn tuples(kind: &ExpansionKind) -> Vec<[i32; 3]> {
    match kind {
        ExpansionKind::T2 { powers } => powers
            .iter()
            .map(|&(t, x, mu)| [t, i32::from(x), i32::from(mu)])
            .collect(),
        ExpansionKind::T3 { powers } => powers
            .iter()
            .map(|&(t, mu)| [t, 0, i32::from(mu)])
            .collect(),
        ExpansionKind::Mu { .. } => panic!("only existing T2/T3 division layouts are supported"),
    }
}

/// Metadata relabeling only. Assert both bijection and the owning plan's full
/// order, including x/mu powers, before any coefficient-row reuse.
fn ordered_map(from: &ExpansionKind, to: &ExpansionKind, shift: i32) -> Vec<usize> {
    assert!(matches!(
        (from, to),
        (ExpansionKind::T2 { .. }, ExpansionKind::T2 { .. })
            | (ExpansionKind::T3 { .. }, ExpansionKind::T3 { .. })
    ));
    let source = tuples(from);
    let target = tuples(to);
    assert_eq!(source.len(), target.len());
    for values in [&source, &target] {
        for (index, value) in values.iter().enumerate() {
            assert!(!values[..index].contains(value), "duplicate power tuple");
        }
    }
    let map: Vec<_> = source
        .iter()
        .map(|p| {
            let translated = [p[0].checked_add(shift).unwrap(), p[1], p[2]];
            target
                .iter()
                .position(|q| q == &translated)
                .expect("missing translated tuple")
        })
        .collect();
    assert_eq!(
        map,
        (0..source.len()).collect::<Vec<_>>(),
        "tuple order changed"
    );
    map
}

struct Layout {
    original: LaurentDivision,
    one_line: Option<LaurentDivision>,
    degree: i32,
    input_map: Vec<usize>,
    next_map: Vec<usize>,
    final_map: Vec<usize>,
}

fn layout(n: usize, rank: usize, cut: usize, budget: &mut Budget) -> ProbeResult<Layout> {
    let original = LaurentDivision::new(n, rank, cut)?;
    budget.include("original plan", &original.exact.expressions)?;
    let u = i32::try_from(original.uncut_count())?;
    let delta = i32::try_from(rank)?.checked_sub(i32::try_from(n)?).unwrap();
    let degree = i32::try_from(cut)?.checked_add(delta).unwrap();
    assert_eq!(degree, i32::try_from(rank)?.checked_sub(u).unwrap());
    if degree < 0 {
        assert!(original.numerator_expansion().is_empty());
        assert!(original.quotient_expansion().is_empty());
        return Ok(Layout {
            original,
            one_line: None,
            degree,
            input_map: Vec::new(),
            next_map: Vec::new(),
            final_map: Vec::new(),
        });
    }
    if u == 0 {
        let input_map = ordered_map(
            original.numerator_expansion(),
            original.quotient_expansion(),
            0,
        );
        return Ok(Layout {
            original,
            one_line: None,
            degree,
            input_map,
            next_map: Vec::new(),
            final_map: Vec::new(),
        });
    }
    let one_line = LaurentDivision::new(
        cut.checked_add(1).unwrap(),
        usize::try_from(degree.checked_add(1).unwrap())?,
        cut,
    )?;
    budget.include("one-line plan", &one_line.exact.expressions)?;
    assert_eq!(one_line.uncut_count(), 1);
    assert_eq!(original.higher_rank(), one_line.higher_rank());
    assert_eq!(
        original.denominator_expansion(),
        one_line.denominator_expansion()
    );
    let input_map = ordered_map(
        original.numerator_expansion(),
        one_line.numerator_expansion(),
        1 - u,
    );
    let next_map = ordered_map(
        one_line.quotient_expansion(),
        one_line.numerator_expansion(),
        1,
    );
    let final_map = ordered_map(
        one_line.quotient_expansion(),
        original.quotient_expansion(),
        0,
    );
    assert_eq!(
        one_line.input_columns(),
        one_line.numerator_expansion().len() + one_line.denominator_expansion().len()
    );
    Ok(Layout {
        original,
        one_line: Some(one_line),
        degree,
        input_map,
        next_map,
        final_map,
    })
}

fn layout_record(layout: &Layout) -> Value {
    json!({"degree":layout.degree,"uncut":layout.original.uncut_count(),
        "higher_rank":layout.original.higher_rank(),
        "numerator_powers":tuples(layout.original.numerator_expansion()),
        "denominator_powers":tuples(layout.original.denominator_expansion()),
        "quotient_powers":tuples(layout.original.quotient_expansion()),
        "initial_input_map":layout.input_map,"next_input_map":layout.next_map,
        "final_output_map":layout.final_map,
        "one_line_constructed":layout.one_line.is_some(),
        "one_line_dimensions":layout.one_line.as_ref().map(|p| json!({
            "propagators":p.cut_size()+1,"rank":layout.degree+1,"cut":p.cut_size()})),
        "one_line_numerator_powers":layout.one_line.as_ref().map(|p| tuples(p.numerator_expansion())),
        "one_line_quotient_powers":layout.one_line.as_ref().map(|p| tuples(p.quotient_expansion()))})
}

fn checkpoint(
    output: &PathBuf,
    report: &mut Value,
    budget: &Budget,
    status: &str,
) -> ProbeResult<()> {
    report["status"] = json!(status);
    report["cumulative_expression_bytes"] = json!(budget.0);
    write_report(output, report)
}

fn inventory_case(
    n: usize,
    rank: usize,
    index: usize,
    report: &mut Value,
    output: &PathBuf,
    budget: &mut Budget,
) -> ProbeResult<()> {
    checkpoint(
        output,
        report,
        budget,
        &format!("incomplete_prepare_n{n}_r{rank}"),
    )?;
    let layout = layout(n, rank, 2, budget)?;
    let original = &layout.original;
    let one = layout
        .one_line
        .as_ref()
        .expect("preselected active uncut case");
    let width = original.quotient_expansion().len();
    let lines = original.uncut_count();
    assert_eq!(width, if n == 6 { 13 } else { 3 });
    report["cases"][index]["layout"] = layout_record(&layout);
    report["cases"][index]["original_input_symbols"] = json!(
        original
            .exact
            .parameters
            .iter()
            .map(ToString::to_string)
            .collect::<Vec<_>>()
    );
    checkpoint(
        output,
        report,
        budget,
        &format!("incomplete_inventory_n{n}_r{rank}"),
    )?;
    let baseline = record("expanded_product", "original_division", &original.exact)?;
    report["cases"][index]["expanded_product"] = baseline.clone();
    checkpoint(
        output,
        report,
        budget,
        &format!("incomplete_one_line_inventory_n{n}_r{rank}"),
    )?;
    let local = record("one_line", "reusable_local_evaluator", &one.exact)?;
    report["cases"][index]["unique_one_line_evaluator"] = local.clone();
    let mut call_records = Vec::new();
    let mut stages = Vec::new();
    let numerator_columns = original.numerator_expansion().len();
    assert_eq!(numerator_columns, width);
    let denominator_columns = original.denominator_expansion().len();
    let mut values = original.exact.parameters[..numerator_columns].to_vec();
    for step in 0..lines {
        checkpoint(
            output,
            report,
            budget,
            &format!("incomplete_n{n}_r{rank}_stage{}", step + 1),
        )?;
        // The checked maps are identity in the owning plan's ordered columns;
        // coefficients are relabeled, never expanded by a custom recurrence.
        let map = if step == 0 {
            &layout.input_map
        } else {
            &layout.next_map
        };
        let mut arguments = vec![Atom::Zero; width];
        for (source, &target) in map.iter().enumerate() {
            arguments[target] = values[source].clone();
        }
        let start = numerator_columns + step * denominator_columns;
        let denominator_range = start..start + denominator_columns;
        arguments.extend_from_slice(&original.exact.parameters[denominator_range.clone()]);
        assert_eq!(arguments.len(), one.input_columns());
        budget.include("bound one-line arguments", &arguments)?;
        values = bind(&one.exact, &arguments);
        budget.include("composed one-line output", &values)?;
        let shift = i32::try_from(lines - step - 1)?;
        let physical_powers: Vec<_> = tuples(one.quotient_expansion())
            .into_iter()
            .map(|[t, x, mu]| [t.checked_add(shift).unwrap(), x, mu])
            .collect();
        stages.push(json!({"stage":step+1,"denominator_input_range":[denominator_range.start,denominator_range.end],
            "physical_output_powers":physical_powers,"output_columns":values.len(),
            "composed_output_expression_bytes":values.iter().map(|e| e.to_string().len()).sum::<usize>(),
            "local_inventory":"unique_one_line_evaluator; composed output is identity-check data, not candidate evaluator"}));
        let mut call = local.clone();
        call["name"] = json!(format!("one_line_call_{}", step + 1));
        call["category"] = json!("actual_repeated_local_call");
        call_records.push(call);
        report["cases"][index]["stages"] = json!(stages);
        report["cases"][index]["actual_call_inventories"] = json!(call_records);
        checkpoint(
            output,
            report,
            budget,
            &format!("completed_n{n}_r{rank}_stage{}", step + 1),
        )?;
    }
    assert_eq!(values.len(), width);
    report["cases"][index]["exact_columns_checked"] = json!(0);
    for (source, &target) in layout.final_map.iter().enumerate() {
        checkpoint(
            output,
            report,
            budget,
            &format!("incomplete_n{n}_r{rank}_identity_{target}"),
        )?;
        let difference = &values[source] - &original.exact.expressions[target];
        budget.include("exact difference", std::slice::from_ref(&difference))?;
        let expanded = difference.expand();
        budget.include("expanded exact difference", std::slice::from_ref(&expanded))?;
        assert_eq!(
            expanded,
            Atom::Zero,
            "n{n}/r{rank} quotient column {target}"
        );
        report["cases"][index]["exact_columns_checked"] = json!(source + 1);
    }
    let repeated = totals(&call_records);
    report["cases"][index]["actual_call_inventories"] = json!(call_records);
    report["cases"][index]["repeated_call_totals"] = repeated.clone();
    report["cases"][index]["instruction_difference_expanded_minus_repeated"] = json!(
        baseline["instructions"].as_i64().unwrap() - repeated["instructions"].as_i64().unwrap()
    );
    report["cases"][index]["storage_and_body_accounting"] = json!({
        "actual_local_calls":lines,"unique_local_evaluator_bodies":1,
        "unique_local_source_bytes":local["source_bytes"],"unique_constants_source_bytes":local["constants_bytes"],
        "unrolled_repeated_callsite_source_bytes":repeated["unrolled_callsite_source_bytes"],
        "original_input_columns":original.input_columns(),"final_output_columns":width,
        "two_reusable_coefficient_buffers_columns":2*width,
        "one_reusable_argument_row_columns":one.input_columns(),
        "local_temporary_slots":local["temporary_slots"],
        "scope":"Buffer quantities are separate logical stores for one row, not measured peak memory; copying, dispatch and packing remain unmeasured"
    });
    report["cases"][index]["status"] = json!("complete_exact_identities");
    checkpoint(output, report, budget, &format!("completed_n{n}_r{rank}"))
}

#[test]
#[ignore = "bounded two-case factored Laurent inventory; no numerical or performance acceptance"]
fn inventory_factored_laurent_two_bubbles() -> ProbeResult<()> {
    let output = std::env::var_os("NINJARS_FACTORED_LAURENT_OUTPUT")
        .map(PathBuf::from)
        .unwrap_or_else(|| {
            PathBuf::from(env!("CARGO_MANIFEST_DIR"))
                .join("../../target/factored-laurent-two-bubbles.json")
        });
    let mut budget = Budget::default();
    let mut report = json!({"schema":"ninjars-factored-laurent-v1",
        "status":"incomplete_preparing","production_dispatch":false,"numerical_execution":false,
        "performance_acceptance":false,"shared_cumulative_expression_budget":MAX_COMPOSED_BYTES,
        "required_external_timeout_seconds_after_lock":180,
        "cases":[{"propagators":6,"rank":7,"cut":2,"status":"pending"},
                 {"propagators":5,"rank":4,"cut":2,"status":"pending"}],
        "candidate_count_boundary":"Actual repeated local evaluator calls; recomposed expressions only prove exact identities",
        "limitations":["No callback changes, runtime recurrence, numerical tests or timing.",
            "Exact identities do not validate roundoff, overflow, conditioning, full Dual behavior or failure semantics.",
            "Input and primal-pivot checks remain external; intermediate-finiteness rejection is not introduced.",
            "Source/body/buffer accounting does not measure allocation, dispatch or packing costs."]});
    checkpoint(&output, &mut report, &budget, "incomplete_layout_guards")?;
    let mut guards = Vec::new();
    for (name, n, rank, cut) in [
        ("uncut_zero", 2, 3, 2),
        ("inactive_degree", 5, 0, 2),
        ("degree_zero", 5, 3, 2),
        ("t3_triangle_mu", 4, 5, 3),
        ("t3_tadpole_mu", 3, 4, 1),
    ] {
        checkpoint(
            &output,
            &mut report,
            &budget,
            &format!("incomplete_layout_guard_{name}"),
        )?;
        let checked = layout(n, rank, cut, &mut budget)?;
        match name {
            "uncut_zero" => {
                assert_eq!(checked.original.uncut_count(), 0);
                assert!(checked.one_line.is_none());
            }
            "inactive_degree" => {
                assert!(checked.degree < 0);
                assert!(checked.one_line.is_none());
            }
            "degree_zero" => {
                assert_eq!(checked.degree, 0);
                assert_eq!(checked.original.quotient_expansion().len(), 1);
            }
            _ => {
                assert!(matches!(
                    checked.original.quotient_expansion(),
                    ExpansionKind::T3 { .. }
                ));
                assert!(
                    tuples(checked.original.quotient_expansion())
                        .iter()
                        .any(|p| p[2] > 0)
                );
            }
        }
        guards.push(json!({"name":name,"propagators":n,"rank":rank,"cut":cut,
            "layout_only":true,"layout":layout_record(&checked)}));
        report["layout_guards"] = json!(guards);
        checkpoint(
            &output,
            &mut report,
            &budget,
            &format!("completed_layout_guard_{name}"),
        )?;
    }
    for (index, (n, rank)) in [(6, 7), (5, 4)].into_iter().enumerate() {
        inventory_case(n, rank, index, &mut report, &output, &mut budget)?;
    }
    let identities: u64 = report["cases"]
        .as_array()
        .unwrap()
        .iter()
        .map(|case| case["exact_columns_checked"].as_u64().unwrap())
        .sum();
    assert_eq!(identities, 16);
    report["total_exact_quotient_identities"] = json!(identities);
    checkpoint(&output, &mut report, &budget, "complete_exact_identities")?;
    println!("{}", output.display());
    Ok(())
}
