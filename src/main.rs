#![allow(dead_code)]
#![allow(unused_variables)]

use std::fs;
use std::thread::sleep;
use std::time::Duration;

use crate::solver::baseline_solver::BaselineSolver;

mod parser;
mod solver;

fn main() {
    let contents = fs::read_to_string("benchmarks/icfp_benchmarks/icfp-problems/5_1000.sl").unwrap();
    let result = parser::parse(&contents);
    let solver = BaselineSolver{};
    let binding = result.unwrap();
    let grammar = solver::Solver::extract_grammar(&solver, &binding, "f");
    let program = solver::Solver::synthesize(&solver, &binding, "f");
    println!("{:?}", program);
}
