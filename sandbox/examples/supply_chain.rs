//! Multi-objective supplier selection / order allocation.
//!
//! Loads supplier data from data/suppliers.csv via polars.
//! Decision variables: raw allocation fractions x_i ∈ [0, 1] per supplier.
//! Fractions are normalized to sum to 1 inside the objective.
//!
//!   f1 = weighted average unit cost         (minimize)
//!   f2 = weighted average lead time (days)  (minimize)
//!
//! The reliability_pct column adjusts effective cost:
//!   adjusted_cost_i = unit_cost_i / (reliability_pct_i / 100)
//! A 78% reliable supplier is cheaper on paper but costs more in rework/waste.
//!
//! Pareto front: portfolios that trade off delivery speed vs total cost.
//! Run: cargo run --example supply_chain

use polars::prelude::*;
use rustypus::core::{EvalFn, Problem};
use rustypus::gatypes::{Real, SolutionDataTypes};
use rustypus::genetic_algorithms_v2::{ExecutionMode, NSGAII};
use std::sync::{Arc, OnceLock};

struct SupplierData {
    names: Vec<String>,
    adjusted_cost: Vec<f64>,
    lead_time: Vec<f64>,
}

static DATA: OnceLock<SupplierData> = OnceLock::new();

fn supply_chain_objective(x: &Vec<f64>) -> Vec<f64> {
    let data = DATA.get().unwrap();
    let total: f64 = x.iter().sum();
    let w: Vec<f64> = if total > 1e-12 {
        x.iter().map(|xi| xi / total).collect()
    } else {
        vec![1.0 / x.len() as f64; x.len()]
    };

    let cost: f64 = w.iter().zip(&data.adjusted_cost).map(|(wi, c)| wi * c).sum();
    let lead: f64 = w.iter().zip(&data.lead_time).map(|(wi, l)| wi * l).sum();

    vec![cost, lead]
}

fn load_data(path: &str) -> Result<SupplierData, PolarsError> {
    let df = LazyCsvReader::new(path)
        .with_has_header(true)
        .finish()?
        .collect()?;

    let names: Vec<String> = df
        .column("name")?
        .str()?
        .into_no_null_iter()
        .map(|s| s.to_string())
        .collect();

    let unit_cost: Vec<f64> = df.column("unit_cost")?.f64()?.into_no_null_iter().collect();
    let lead_time: Vec<f64> = df
        .column("lead_time_days")?
        .cast(&polars::datatypes::DataType::Float64)?
        .f64()?
        .into_no_null_iter()
        .collect();
    let reliability: Vec<f64> = df.column("reliability_pct")?.f64()?.into_no_null_iter().collect();

    let adjusted_cost: Vec<f64> = unit_cost
        .iter()
        .zip(&reliability)
        .map(|(c, r)| c / (r / 100.0))
        .collect();

    Ok(SupplierData { names, adjusted_cost, lead_time })
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let data = load_data("data/suppliers.csv")?;
    let n = data.names.len();

    println!("Suppliers ({n}):");
    for (i, name) in data.names.iter().enumerate() {
        println!(
            "  {name:<14} adjusted_cost={:.2}  lead={:.0}d",
            data.adjusted_cost[i], data.lead_time[i]
        );
    }

    DATA.set(data).ok();

    let types: Vec<SolutionDataTypes> = (0..n)
        .map(|_| SolutionDataTypes::Real(Real::new(Some(0.0), Some(1.0))))
        .collect();

    let problem = Arc::new(Problem {
        solution_length: n,
        number_of_objectives: 2,
        objective_constraint: None,
        objective_constraint_operands: None,
        direction: Some(vec![-1, -1]),
        solution_data_types: types,
        variable_constraints: None,
        eval_fn: EvalFn::Single(supply_chain_objective),
    });

    let mut ga = NSGAII::new(Arc::clone(&problem), 100, ExecutionMode::MultiThreaded);
    ga.run(10_000);

    println!("\n--- Pareto-Optimal Supplier Allocations ---");
    println!("{:<12} {:<12} {}", "Cost/unit", "Lead (days)", "Allocation");

    let mut archive = ga.get_archive().to_vec();
    archive.sort_by(|a, b| {
        a.objective_fitness_values[0]
            .partial_cmp(&b.objective_fitness_values[0])
            .unwrap()
    });

    // Deduplicate by rounding objectives to 3 decimal places
    let mut seen: std::collections::HashSet<(u64, u64)> = std::collections::HashSet::new();
    let unique: Vec<_> = archive
        .iter()
        .filter(|s| {
            let k = (
                (s.objective_fitness_values[0] * 1000.0).round() as u64,
                (s.objective_fitness_values[1] * 1000.0).round() as u64,
            );
            seen.insert(k)
        })
        .collect();

    let names = &DATA.get().unwrap().names;
    println!("({} unique allocations on Pareto front)\n", unique.len());
    for sol in &unique {
        let total: f64 = sol.solution.iter().sum();
        let w: Vec<f64> = sol.solution.iter().map(|xi| xi / total).collect();
        let cost = sol.objective_fitness_values[0];
        let lead = sol.objective_fitness_values[1];

        // Only print suppliers with > 5% allocation
        let alloc: Vec<String> = w
            .iter()
            .zip(names)
            .filter(|(wi, _)| **wi > 0.05)
            .map(|(wi, name)| format!("{name}: {:.0}%", wi * 100.0))
            .collect();

        println!("{:<12.3} {:<12.2} {}", cost, lead, alloc.join("  "));
    }

    Ok(())
}
