#![allow(dead_code)]
#![allow(unused_variables)]

use std::fs;
use std::time::Instant;
use crate::solver::baseline_solver::BaselineSolver;

mod parser;
mod solver;


fn main() {
    #![allow(warnings)]
    let test_path = "test_bool_1.sl";
    let test_name = "AIG";
    let contents = fs::read_to_string(&test_path).unwrap();
    let parsed = parser::parse(&contents).unwrap();
    let solver = BaselineSolver{};
    let start = Instant::now();
    let program = solver::Solver::synthesize(&solver, &parsed, &test_name);
    let duration = start.elapsed();
    println!("-------------Program Synthesized Successfully-------------");
    println!("Test name: {:?}", test_path);
    println!("Func name: {:?}", test_name);
    println!("Program: \n\t{:?}", program);
    println!("Elapsed time: {:?}", duration);
    println!("----------------------------------------------------------");
}
