use crate::parser::ast::*;

use crate::parser::{Rule, SyGuSParser};
use pest::error::Error;
use pest::iterators::Pair;
use std::collections::HashMap;

use super::ast;

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
    fn visit_synth_fun(&mut self, pair: Pair<Rule>) -> Result<&Self::Env, Error<Rule>>;
    // SmtCmd
    fn visit_define_fun(&mut self, pair: Pair<Rule>) -> Result<&Self::Env, Error<Rule>>;
    fn visit_set_logic(&mut self, pair: Pair<Rule>) -> Result<&Self::Env, Error<Rule>>;
    fn visit_set_option(&mut self, pair: Pair<Rule>) -> Result<&Self::Env, Error<Rule>>;
    // GrammarDef
    fn visit_grammar_def(&mut self, pair: Pair<Rule>) -> Result<GrammarDef, Error<Rule>>;
    fn visit_grouped_rule_list(&mut self, pair: Pair<Rule>) -> Result<Production, Error<Rule>>;
    fn visit_gterm(&mut self, pair: Pair<Rule>) -> Result<GTerm, Error<Rule>>;
    // Term
    fn visit_term(&mut self, pair: Pair<Rule>) -> Result<Expr, Error<Rule>>;
    fn visit_bf_term(&mut self, pair: Pair<Rule>) -> Result<Expr, Error<Rule>>;
    fn visit_sorted_var(&mut self, pair: Pair<Rule>) -> Result<(Symbol, Sort), Error<Rule>>;
    // fn visit_var_binding(&mut self, pair: Pair<Rule>) -> Result<Expr, Error<Rule>>;
    // Term productions
    fn visit_term_ident(&mut self, pair: Pair<Rule>) -> Result<Expr, Error<Rule>>;
    fn visit_term_literal(&mut self, pair: Pair<Rule>) -> Result<Expr, Error<Rule>>;
    fn visit_term_ident_list(&mut self, pair: Pair<Rule>) -> Result<Vec<Expr>, Error<Rule>>;
    // fn visit_term_attri(&mut self, pair: Pair<Rule>) -> Result<&Self::Env, Error<Rule>>;
    // fn visit_term_exist(&mut self, pair: Pair<Rule>) -> Result<&Self::Env, Error<Rule>>;
    // fn visit_term_forall(&mut self, pair: Pair<Rule>) -> Result<&Self::Env, Error<Rule>>;
    // fn visit_term_let(&mut self, pair: Pair<Rule>) -> Result<&Self::Env, Error<Rule>>;
    // BfTerm productions
    fn visit_bfterm_ident(&mut self, pair: Pair<Rule>) -> Result<Expr, Error<Rule>>;
    fn visit_bfterm_literal(&mut self, pair: Pair<Rule>) -> Result<Expr, Error<Rule>>;
    fn visit_bfterm_ident_list(&mut self, pair: Pair<Rule>) -> Result<Vec<Expr>, Error<Rule>>;
    // fn visit_bfterm_attri(&mut self, pair: Pair<Rule>) -> Result<&Self::Env, Error<Rule>>;
    // Sort
    fn visit_sort(&mut self, pair: Pair<Rule>) -> Result<Sort, Error<Rule>>;
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
                        Some("synthe_fun") => self.visit_synth_fun(pair)?,
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
        // TODO: when running into this, we should call the solver with current collected SyGuSProg information
        // Implement in the last step
        Ok(&self.sygus_prog)
    }
    fn visit_constraint(&mut self, pair: Pair<Rule>) -> Result<&SyGuSProg, Error<Rule>> {
        for pair in pair.into_inner() {
            match pair.as_rule() {
                Rule::Term => {
                    let constr_expr = self.visit_term(pair)?;
                    self.sygus_prog.constraints.push(constr_expr);
                }
                _ => unreachable!(),
            }
        }
        Ok(&self.sygus_prog)
    }
    fn visit_declare_var(&mut self, pair: Pair<Rule>) -> Result<&SyGuSProg, Error<Rule>> {
        let mut var_name;
        let mut var_sort;
        for pair in pair.into_inner() {
            match pair.as_rule() {
                Rule::Symbol => {
                    var_name = pair.as_str().to_string();
                }
                Rule::Sort => {
                    var_sort = self.visit_sort(pair)?;
                }
                _ => unreachable!(),
            }
        }
        self.sygus_prog.declare_var.insert(var_name, var_sort);
        Ok(&self.sygus_prog)
    }
    fn visit_synth_fun(&mut self, pair: Pair<Rule>) -> Result<&SyGuSProg, Error<Rule>> {
        let mut sorted_vars = Vec::new();
        let mut func_name;
        let mut ret_sort;
        let mut body;
        let mut grammar_def = GrammarDef::new(); // optional
        for pair in pair.into_inner() {
            match pair.as_rule() {
                Rule::Symbol => {
                    func_name = pair.as_str().to_string();
                }
                Rule::SortedVar => {
                    let (var_name, var_sort) = self.visit_sorted_var(pair)?;
                    sorted_vars.push((var_name, var_sort));
                }
                Rule::Sort => {
                    ret_sort = self.visit_sort(pair)?;
                }
                Rule::Term => {
                    body = self.visit_term(pair)?;
                }
                Rule::GrammarDef => {
                    grammar_def = self.visit_grammar_def(pair)?;
                }
                _ => unreachable!(),
            }
        }

        let func_body = FuncBody {
            name: func_name,
            params: sorted_vars,
            return_type: ret_sort,
            body: body,
        };
        self.sygus_prog.synthe_func.insert(func_name, grammar_def);
        Ok(&self.sygus_prog)
    }
    // SmtCmd
    fn visit_define_fun(&mut self, pair: Pair<Rule>) -> Result<&SyGuSProg, Error<Rule>> {
        let mut func_name;
        let mut sorted_vars = Vec::new();
        let mut ret_sort;
        let mut body;
        for pair in pair.into_inner() {
            match pair.as_rule() {
                Rule::Symbol => {
                    func_name = pair.as_str().to_string();
                }
                Rule::SortedVar => {
                    let (var_name, var_sort) = self.visit_sorted_var(pair)?;
                    sorted_vars.push((var_name, var_sort));
                }
                Rule::Sort => {
                    ret_sort = self.visit_sort(pair)?;
                }
                Rule::Term => {
                    body = self.visit_term(pair)?;
                }
                _ => unreachable!(),
            }
        }
        let func_body = FuncBody {
            name: func_name,
            params: sorted_vars,
            return_type: ret_sort,
            body: body,
        };
        self.sygus_prog.define_fun.insert(func_name, func_body);
        Ok(&self.sygus_prog)
    }
    fn visit_set_logic(&mut self, pair: Pair<Rule>) -> Result<&SyGuSProg, Error<Rule>> {
        // only one child
        let logic = pair.into_inner().next().unwrap().as_str();
        match logic {
            "LIA" => self.sygus_prog.set_logic = SetLogic::LIA,
            "BV" => self.sygus_prog.set_logic = SetLogic::BV,
            _ => self.sygus_prog.set_logic = SetLogic::Unknown,
        }
        Ok(&self.sygus_prog)
    }
    fn visit_set_option(&mut self, pair: Pair<Rule>) -> Result<&SyGuSProg, Error<Rule>> {
        let mut opt_name;
        let mut opt_value;
        for pair in pair.into_inner() {
            match pair.as_rule() {
                Rule::Keyword => {
                    opt_name = pair.as_str().to_string();
                }
                Rule::Literal => {
                    opt_value = pair.as_str().to_string();
                }
                _ => unreachable!(),
            }
        }
        self.sygus_prog.set_option.insert(opt_name, opt_value);
        Ok(&self.sygus_prog)
    }
    // GrammarDef
    fn visit_grammar_def(&mut self, pair: Pair<Rule>) -> Result<GrammarDef, Error<Rule>> {
        let mut grammar_def = GrammarDef::new();
        for pair in pair.into_inner() {
            match pair.as_rule() {
                Rule::GroupedRuleList => {
                    let production = self.visit_grouped_rule_list(pair)?;
                    grammar_def.non_terminals.push(production);
                }
                _ => unreachable!(),
            }
        }
        Ok(grammar_def)
    }
    fn visit_grouped_rule_list(&mut self, pair: Pair<Rule>) -> Result<Production, Error<Rule>> {
        let mut production = Production::new();
        for pair in pair.into_inner() {
            match pair.as_rule() {
                Rule::Symbol => {
                    production.lhs = pair.as_str().to_string();
                }
                Rule::Sort => {
                    production.lhs_sort = self.visit_sort(pair)?;
                }
                Rule::GTerm => {
                    let gterm = self.visit_gterm(pair)?;
                    production.rhs.push(gterm);
                }
                _ => unreachable!(),
            }
        }
        Ok(production)
    }
    fn visit_gterm(&mut self, pair: Pair<Rule>) -> Result<GTerm, Error<Rule>> {
        let mut gterm = GTerm::None;
        // only one child
        let pair = pair.into_inner().next().unwrap();
        match pair.as_node_tag() {
            Some("gterm_constant") => {
                let sort = self.visit_sort(pair)?;
                gterm = GTerm::Constant(sort);
            }
            Some("gterm_variable") => {
                let sort = self.visit_sort(pair)?;
                gterm = GTerm::Variable(sort);
            }
            _ => todo!("BfTerm in gterm not implemented")
        }
        Ok(gterm)
    }
    // Term
    fn visit_term(&mut self, pair: Pair<Rule>) -> Result<Expr, Error<Rule>> {}
    fn visit_bf_term(&mut self, pair: Pair<Rule>) -> Result<Expr, Error<Rule>> {}
    fn visit_sorted_var(&mut self, pair: Pair<Rule>) -> Result<(Symbol, Sort), Error<Rule>> {}
    // Term productions
    fn visit_term_ident(&mut self, pair: Pair<Rule>) -> Result<Expr, Error<Rule>> {}
    fn visit_term_literal(&mut self, pair: Pair<Rule>) -> Result<Expr, Error<Rule>> {}
    fn visit_term_ident_list(&mut self, pair: Pair<Rule>) -> Result<Vec<Expr>, Error<Rule>> {}
    // BfTerm productions
    fn visit_bfterm_ident(&mut self, pair: Pair<Rule>) -> Result<Expr, Error<Rule>> {}
    fn visit_bfterm_literal(&mut self, pair: Pair<Rule>) -> Result<Expr, Error<Rule>> {}
    fn visit_bfterm_ident_list(&mut self, pair: Pair<Rule>) -> Result<Vec<Expr>, Error<Rule>> {}
    // Sort
    fn visit_sort(&mut self, pair: Pair<Rule>) -> Result<Sort, Error<Rule>> {
        let mut sort = Sort::None;
        for pair in pair.into_inner() {
            match pair.as_rule() {
                Rule::Sort => {
                    match pair.as_str() {
                        "Bool" => sort = Sort::Bool,
                        "Int" => sort = Sort::Int,
                        // TODO: BitVec should have a bit width, need to parse it from sibling
                        "BitVec" => {
                            let bit_width = pair.as_str().parse::<i32>().unwrap();
                            sort = Sort::BitVec(bit_width);
                        }
                        _ => unreachable!(),
                    }
                }
                _ => unreachable!(),
            }
        }
        Ok(sort)
    }
}

#[cfg(test)]
mod tests {
    use std::fs;

    use crate::parser::parse;

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
