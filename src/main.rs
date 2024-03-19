#![allow(dead_code)]
#![allow(unused_variables)]

use std::fs;
use crate::solver::baseline_solver::BaselineSolver;

mod parser;
mod solver;

fn main() {
    let contents = fs::read_to_string("test_bool_1.sl").unwrap();
    let parsed = parser::parse(&contents).unwrap();
    let solver = BaselineSolver{};
    let program = solver::Solver::synthesize(&solver, &parsed, "AIG");
    println!("{:?}", program);
}
