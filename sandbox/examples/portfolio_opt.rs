//! Multi-objective portfolio optimization.
//!
//! Loads monthly returns from data/monthly_returns.csv via polars.
//! Decision variables: raw allocation weights x_i ∈ [0, 1] per asset.
//! Weights are normalized to sum to 1 inside the objective (no equality constraint needed).
//!
//!   f1 = portfolio variance   w^T Σ w       (minimize risk)
//!   f2 = -expected return     -w^T μ        (minimize → maximize return)
//!
//! Pareto front traces the efficient frontier.
//! Run: cargo run --example portfolio_opt

use polars::prelude::*;
use puggles::core::{EvalFn, Problem};
use puggles::gatypes::{Real, SolutionDataTypes};
use puggles::genetic_algorithms_v2::{ExecutionMode, NSGAII};
use std::sync::{Arc, OnceLock};

struct PortfolioData {
    tickers: Vec<String>,
    means: Vec<f64>,
    cov: Vec<Vec<f64>>,
}

static DATA: OnceLock<PortfolioData> = OnceLock::new();

fn portfolio_objective(x: &Vec<f64>) -> Vec<f64> {
    let data = DATA.get().unwrap();
    let n = data.means.len();

    let total: f64 = x.iter().sum();
    let weights: Vec<f64> = if total > 1e-12 {
        x.iter().map(|w| w / total).collect()
    } else {
        vec![1.0 / n as f64; n]
    };

    let expected_return: f64 = weights.iter().zip(&data.means).map(|(w, r)| w * r).sum();

    let variance: f64 = (0..n)
        .map(|i| (0..n).map(|j| weights[i] * weights[j] * data.cov[i][j]).sum::<f64>())
        .sum();

    vec![variance, -expected_return]
}

fn load_data(path: &str) -> Result<PortfolioData, PolarsError> {
    let df = LazyCsvReader::new(path)
        .with_has_header(true)
        .finish()?
        .collect()?;

    let tickers: Vec<String> = df
        .get_column_names()
        .iter()
        .filter(|n| **n != "month")
        .map(|s| s.to_string())
        .collect();

    let n = tickers.len();
    let mut series: Vec<Vec<f64>> = Vec::with_capacity(n);

    for ticker in &tickers {
        let vals: Vec<f64> = df
            .column(ticker)?
            .f64()?
            .into_no_null_iter()
            .collect();
        series.push(vals);
    }

    let nrows = series[0].len() as f64;
    let means: Vec<f64> = series.iter()
        .map(|s| s.iter().sum::<f64>() / nrows)
        .collect();

    let mut cov = vec![vec![0.0; n]; n];
    for i in 0..n {
        for j in 0..n {
            let mi = means[i];
            let mj = means[j];
            cov[i][j] = series[i].iter().zip(&series[j])
                .map(|(xi, xj)| (xi - mi) * (xj - mj))
                .sum::<f64>()
                / (nrows - 1.0);
        }
    }

    Ok(PortfolioData { tickers, means, cov })
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let data = load_data("data/monthly_returns.csv")?;
    let n = data.means.len();

    println!("Assets: {:?}", data.tickers);
    println!(
        "Monthly mean returns: {}",
        data.means
            .iter()
            .map(|r| format!("{:.2}%", r * 100.0))
            .collect::<Vec<_>>()
            .join(", ")
    );

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
        eval_fn: EvalFn::Single(portfolio_objective),
    });

    let mut ga = NSGAII::new(Arc::clone(&problem), 200, ExecutionMode::MultiThreaded);
    ga.run(20_000);

    println!("\n--- Efficient Frontier (sorted by risk) ---");
    println!("{:<14} {:<14} {}", "Risk (σ²)", "Return/mo (%)", "Weights");

    let mut archive = ga.get_archive().to_vec();
    archive.sort_by(|a, b| {
        a.objective_fitness_values[0]
            .partial_cmp(&b.objective_fitness_values[0])
            .unwrap()
    });

    // Deduplicate by rounding objectives to 4 decimal places
    let mut seen: std::collections::HashSet<(u64, u64)> = std::collections::HashSet::new();
    let unique: Vec<_> = archive
        .iter()
        .filter(|s| {
            let k = (
                (s.objective_fitness_values[0] * 1e4).round() as u64,
                (s.objective_fitness_values[1] * 1e4).round() as u64,
            );
            seen.insert(k)
        })
        .collect();

    let tickers = &DATA.get().unwrap().tickers;
    println!("({} unique portfolios on efficient frontier)\n", unique.len());
    for sol in &unique {
        let total: f64 = sol.solution.iter().sum();
        let weights: Vec<f64> = sol.solution.iter().map(|w| w / total).collect();
        let risk = sol.objective_fitness_values[0];
        let ret = -sol.objective_fitness_values[1] * 100.0;
        let w_str: Vec<String> = weights
            .iter()
            .zip(tickers)
            .map(|(w, t)| format!("{}: {:.0}%", t, w * 100.0))
            .collect();
        println!("{:<14.6} {:<14.4} {}", risk, ret, w_str.join("  "));
    }

    Ok(())
}
