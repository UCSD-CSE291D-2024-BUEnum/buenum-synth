#![allow(dead_code)]
#![allow(unused_variables)]

use crate::solver::baseline_solver::BaselineSolver;
use std::fs;
use std::time::Instant;

mod parser;
mod solver;

fn main() {
    #![allow(warnings)]
    let test_path = "benchmarks/hackers_del/hd-01-d0-prog.sl";
    let test_name = "f";
    let contents = fs::read_to_string(&test_path).unwrap();
    let parsed = parser::parse(&contents).unwrap();
    let solver = BaselineSolver {};
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
