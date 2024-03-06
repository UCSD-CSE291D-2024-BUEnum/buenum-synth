use crate::parser::ast::*;

use crate::parser::{Rule, SyGuSParser};
use pest::error::Error;
use pest::iterators::Pair;
use std::collections::HashMap;

pub trait Visitor<T> {
     fn visit_main(&mut self, pair: Pair<Rule>) -> Result<T, Error<Rule>>;
}

pub struct SyGuSVisitor;

impl Visitor<SyGuSProg> for SyGuSVisitor {
    fn visit_main(&mut self, pair: Pair<Rule>) -> Result<SyGuSProg, Error<Rule>> {
        let mut set_logic = SetLogic::Unknown;
        let mut define_fun = HashMap::new();
        let mut declare_var = HashMap::new();
        let mut synthe_func = HashMap::new();
        let mut set_option = HashMap::new();

        for pair in pair.into_inner() {
            match pair.as_rule() {
                Rule::SyGuS => {

                }
                // Rule::set_logic => {
                //     set_logic = match pair.as_str() {
                //         "LIA" => SetLogic::LIA,
                //         "BV" => SetLogic::BV,
                //         _ => SetLogic::Unknown,
                //     };
                // }
                // Rule::define_fun => {
                //     let (name, body) = self.visit_define_fun(pair)?;
                //     define_fun.insert(name, body);
                // }
                // Rule::declare_var => {
                //     let (symbol, sort) = self.visit_declare_var(pair)?;
                //     declare_var.insert(symbol, sort);
                // }
                // Rule::synthe_fun => {
                //     let (name, grammar) = self.visit_synthe_fun(pair)?;
                //     synthe_func.insert(name, grammar);
                // }
                // Rule::set_option => {
                //     let (opt_name, opt_value) = self.visit_set_option(pair)?;
                //     set_option.insert(opt_name, opt_value);
                // }
                _ => unreachable!(),
            }
        }

        Ok(SyGuSProg {
            set_logic,
            define_fun,
            declare_var,
            synthe_func,
            set_option,
        })
    }
}

#[cfg(test)]
mod tests {
    use std::fs;

    use super::*;
    use crate::parser::{parse, SyGuSParser};

    #[test]
    fn test_visit_main() {
        let filename =
            "/home/jerry/Projects/buenum-synth/benchmarks/bitvector-benchmarks/parity-AIG-d0.sl"
                .to_string();
        let input = fs::read_to_string(&filename).unwrap();
        let res = parse(&input);
        let res = match res {
            Ok(res) => res,
            Err(e) => {
                panic!("Error parsing file: {}\nError: {:#?}", filename, e);
            }
        };
        // pretty print
        println!("{:#?}", res);
    }
}
