use crate::parser::ast::*;

use crate::parser::{Rule, SyGuSParser};
use pest::error::Error;
use pest::iterators::Pair;
use std::collections::HashMap;

pub trait Visitor {
    type Env;
    fn visit_main(&mut self, pair: Pair<Rule>) -> Result<&Self::Env, Error<Rule>>;
    // SyGuG
    fn visit_sygus(&mut self, pair: Pair<Rule>) -> Result<&Self::Env, Error<Rule>>;
    fn visit_cmd(&mut self, pair: Pair<Rule>) -> Result<&Self::Env, Error<Rule>>;
    fn visit_smt_cmd(&mut self, pair: Pair<Rule>) -> Result<&Self::Env, Error<Rule>>;
    // Cmd
    fn visit_check_synth(&mut self, pair: Pair<Rule>) -> Result<&Self::Env, Error<Rule>>;
    fn visit_constraint(&mut self, pair: Pair<Rule>) -> Result<&Self::Env, Error<Rule>>;
    fn visit_declare_var(&mut self, pair: Pair<Rule>) -> Result<&Self::Env, Error<Rule>>;
    fn visit_synthe_fun(&mut self, pair: Pair<Rule>) -> Result<&Self::Env, Error<Rule>>;
    // SmtCmd
    fn visit_define_fun(&mut self, pair: Pair<Rule>) -> Result<&Self::Env, Error<Rule>>;
    fn visit_set_logic(&mut self, pair: Pair<Rule>) -> Result<&Self::Env, Error<Rule>>;
    fn visit_set_option(&mut self, pair: Pair<Rule>) -> Result<&Self::Env, Error<Rule>>;
    // GrammarDef
    fn visit_grammar_def(&mut self, pair: Pair<Rule>) -> Result<&Self::Env, Error<Rule>>;
    fn visit_grouped_rule_list(&mut self, pair: Pair<Rule>) -> Result<&Self::Env, Error<Rule>>;
    fn visit_gterm(&mut self, pair: Pair<Rule>) -> Result<&Self::Env, Error<Rule>>;
    // Term
    fn visit_term(&mut self, pair: Pair<Rule>) -> Result<&Self::Env, Error<Rule>>;
    fn visit_bf_term(&mut self, pair: Pair<Rule>) -> Result<&Self::Env, Error<Rule>>;
    fn visit_sorted_var(&mut self, pair: Pair<Rule>) -> Result<&Self::Env, Error<Rule>>;
    fn visit_var_binding(&mut self, pair: Pair<Rule>) -> Result<&Self::Env, Error<Rule>>;
    // Term productions
    fn visit_term_ident(&mut self, pair: Pair<Rule>) -> Result<&Self::Env, Error<Rule>>;
    fn visit_term_literal(&mut self, pair: Pair<Rule>) -> Result<&Self::Env, Error<Rule>>;
    fn visit_term_ident_list(&mut self, pair: Pair<Rule>) -> Result<&Self::Env, Error<Rule>>;
    // fn visit_term_attri(&mut self, pair: Pair<Rule>) -> Result<&Self::Env, Error<Rule>>;
    // fn visit_term_exist(&mut self, pair: Pair<Rule>) -> Result<&Self::Env, Error<Rule>>;
    // fn visit_term_forall(&mut self, pair: Pair<Rule>) -> Result<&Self::Env, Error<Rule>>;
    // fn visit_term_let(&mut self, pair: Pair<Rule>) -> Result<&Self::Env, Error<Rule>>;
    // BfTerm productions
    fn visit_bfterm_ident(&mut self, pair: Pair<Rule>) -> Result<&Self::Env, Error<Rule>>;
    fn visit_bfterm_literal(&mut self, pair: Pair<Rule>) -> Result<&Self::Env, Error<Rule>>;
    fn visit_bfterm_ident_list(&mut self, pair: Pair<Rule>) -> Result<&Self::Env, Error<Rule>>;
    // fn visit_bfterm_attri(&mut self, pair: Pair<Rule>) -> Result<&Self::Env, Error<Rule>>;
    // Sort
    fn visit_sort(&mut self, pair: Pair<Rule>) -> Result<&Self::Env, Error<Rule>>;
    // fn visit_var_binding(&mut self, pair: Pair<Rule>) -> Result<&Self::Env, Error<Rule>>;
}

pub struct SyGuSVisitor {
    pub sygus_prog: SyGuSProg,
}

impl SyGuSVisitor {
    pub fn new() -> SyGuSVisitor {
        SyGuSVisitor {
            sygus_prog: SyGuSProg::new(),
        }
    }
}

impl Visitor for SyGuSVisitor {
    type Env = SyGuSProg;
    fn visit_main(&mut self, pair: Pair<Rule>) -> Result<&SyGuSProg, Error<Rule>> {
        for pair in pair.into_inner() {
            match pair.as_rule() {
                Rule::SyGuS => {
                    self.visit_sygus(pair)?;
                }
                _ => unreachable!(),
            }
        }
        Ok(&self.sygus_prog)
    }
    // SyGuG
    fn visit_sygus(&mut self, pair: Pair<Rule>) -> Result<&SyGuSProg, Error<Rule>> {
        for pair in pair.into_inner() {
            match pair.as_rule() {
                Rule::Cmd => {
                    self.visit_cmd(pair)?;
                }
                _ => unreachable!(),
            }
        }
        Ok(&self.sygus_prog)
    }
    fn visit_cmd(&mut self, pair: Pair<Rule>) -> Result<&SyGuSProg, Error<Rule>> {
        for pair in pair.into_inner() {
            match pair.as_rule() {
                Rule::SmtCmd => {
                    self.visit_smt_cmd(pair)?;
                }
                _ => {
                    // for the other children of Cmd
                    // check_synth, constraint, declare_var, synthe_fun
                    // use the tag to determine which function to call
                    match pair.as_node_tag() {
                        Some("check_synth") => self.visit_check_synth(pair)?,
                        Some("constraint") => self.visit_constraint(pair)?,
                        Some("declare_var") => self.visit_declare_var(pair)?,
                        Some("synthe_fun") => self.visit_synthe_fun(pair)?,
                        _ => unreachable!(),
                    };
                }
            }
        }
        Ok(&self.sygus_prog)
    }
    fn visit_smt_cmd(&mut self, pair: Pair<Rule>) -> Result<&SyGuSProg, Error<Rule>> {
        for pair in pair.into_inner() {
            match pair.as_node_tag() {
                Some("define_fun") => self.visit_define_fun(pair)?,
                Some("set_logic") => self.visit_set_logic(pair)?,
                Some("set_option") => self.visit_set_option(pair)?,
                _ => unreachable!(),
            };
        }
        Ok(&self.sygus_prog)
    }
    // Cmd
    fn visit_check_synth(&mut self, pair: Pair<Rule>) -> Result<&SyGuSProg, Error<Rule>> {
        Ok(&self.sygus_prog)
    }
    fn visit_constraint(&mut self, pair: Pair<Rule>) -> Result<&SyGuSProg, Error<Rule>> {
        Ok(&self.sygus_prog)
    }
    fn visit_declare_var(&mut self, pair: Pair<Rule>) -> Result<&SyGuSProg, Error<Rule>> {
        Ok(&self.sygus_prog)
    }
    fn visit_synthe_fun(&mut self, pair: Pair<Rule>) -> Result<&SyGuSProg, Error<Rule>> {
        Ok(&self.sygus_prog)
    }
    // SmtCmd
    fn visit_define_fun(&mut self, pair: Pair<Rule>) -> Result<&SyGuSProg, Error<Rule>> {
        Ok(&self.sygus_prog)
    }
    fn visit_set_logic(&mut self, pair: Pair<Rule>) -> Result<&SyGuSProg, Error<Rule>> {
        Ok(&self.sygus_prog)
    }
    fn visit_set_option(&mut self, pair: Pair<Rule>) -> Result<&SyGuSProg, Error<Rule>> {
        Ok(&self.sygus_prog)
    }
    // GrammarDef
    fn visit_grammar_def(&mut self, pair: Pair<Rule>) -> Result<&SyGuSProg, Error<Rule>> {
        Ok(&self.sygus_prog)
    }
    fn visit_grouped_rule_list(&mut self, pair: Pair<Rule>) -> Result<&SyGuSProg, Error<Rule>> {
        Ok(&self.sygus_prog)
    }
    fn visit_gterm(&mut self, pair: Pair<Rule>) -> Result<&SyGuSProg, Error<Rule>> {
        Ok(&self.sygus_prog)
    }
    // Term
    fn visit_term(&mut self, pair: Pair<Rule>) -> Result<&SyGuSProg, Error<Rule>> {
        Ok(&self.sygus_prog)
    }
    fn visit_bf_term(&mut self, pair: Pair<Rule>) -> Result<&SyGuSProg, Error<Rule>> {
        Ok(&self.sygus_prog)
    }
    fn visit_sorted_var(&mut self, pair: Pair<Rule>) -> Result<&SyGuSProg, Error<Rule>> {
        Ok(&self.sygus_prog)
    }
    fn visit_var_binding(&mut self, pair: Pair<Rule>) -> Result<&SyGuSProg, Error<Rule>> {
        Ok(&self.sygus_prog)
    }
    // Term productions
    fn visit_term_ident(&mut self, pair: Pair<Rule>) -> Result<&SyGuSProg, Error<Rule>> {
        Ok(&self.sygus_prog)
    }
    fn visit_term_literal(&mut self, pair: Pair<Rule>) -> Result<&SyGuSProg, Error<Rule>> {
        Ok(&self.sygus_prog)
    }
    fn visit_term_ident_list(&mut self, pair: Pair<Rule>) -> Result<&SyGuSProg, Error<Rule>> {
        Ok(&self.sygus_prog)
    }
    // BfTerm productions
    fn visit_bfterm_ident(&mut self, pair: Pair<Rule>) -> Result<&SyGuSProg, Error<Rule>> {
        Ok(&self.sygus_prog)
    }
    fn visit_bfterm_literal(&mut self, pair: Pair<Rule>) -> Result<&SyGuSProg, Error<Rule>> {
        Ok(&self.sygus_prog)
    }
    fn visit_bfterm_ident_list(&mut self, pair: Pair<Rule>) -> Result<&SyGuSProg, Error<Rule>> {
        Ok(&self.sygus_prog)
    }
    // Sort
    fn visit_sort(&mut self, pair: Pair<Rule>) -> Result<&SyGuSProg, Error<Rule>> {
        Ok(&self.sygus_prog)
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
