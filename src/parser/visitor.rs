use crate::parser::ast::*;

use crate::parser::{Rule, SyGuSParser};
use core::panic;
use pest::error::Error;
use pest::iterators::Pair;
use std::collections::HashMap;
use std::env;

use super::ast;

pub trait Visitor {
    type Prog;
    type Env;
    fn visit_main(&mut self, pairs: Pair<Rule>) -> Result<&Self::Prog, Error<Rule>>;
    // SyGuG
    fn visit_sygus(&mut self, pairs: Pair<Rule>) -> Result<&Self::Prog, Error<Rule>>;
    fn visit_cmd(&mut self, pairs: Pair<Rule>) -> Result<&Self::Prog, Error<Rule>>;
    fn visit_smt_cmd(&mut self, pairs: Pair<Rule>) -> Result<&Self::Prog, Error<Rule>>;
    // Cmd
    fn visit_check_synth(&mut self, pairs: Pair<Rule>) -> Result<&Self::Prog, Error<Rule>>;
    fn visit_constraint(
        &mut self,
        env: &Self::Env,
        pairs: Pair<Rule>,
    ) -> Result<&Self::Prog, Error<Rule>>;
    fn visit_declare_var(&mut self, pairs: Pair<Rule>) -> Result<&Self::Prog, Error<Rule>>;
    fn visit_synth_fun(&mut self, pairs: Pair<Rule>) -> Result<&Self::Prog, Error<Rule>>;
    // SmtCmd
    fn visit_define_fun(&mut self, pairs: Pair<Rule>) -> Result<&Self::Prog, Error<Rule>>;
    fn visit_set_logic(&mut self, pairs: Pair<Rule>) -> Result<&Self::Prog, Error<Rule>>;
    fn visit_set_option(&mut self, pairs: Pair<Rule>) -> Result<&Self::Prog, Error<Rule>>;
    // GrammarDef
    fn visit_grammar_def(
        &mut self,
        env: &Self::Env,
        pairs: Pair<Rule>,
    ) -> Result<GrammarDef, Error<Rule>>;
    fn visit_grouped_rule_list(
        &mut self,
        env: &Self::Env,
        pairs: Pair<Rule>,
    ) -> Result<Production, Error<Rule>>;
    fn visit_gterm(&mut self, env: &Self::Env, pairs: Pair<Rule>) -> Result<GTerm, Error<Rule>>;
    // Term
    fn visit_term(&mut self, env: &Self::Env, pairs: Pair<Rule>) -> Result<Expr, Error<Rule>>;
    fn visit_bf_term(&mut self, env: &Self::Env, pairs: Pair<Rule>) -> Result<GExpr, Error<Rule>>;
    fn visit_sorted_var(&mut self, pairs: Pair<Rule>) -> Result<(Symbol, Sort), Error<Rule>>;
    // fn visit_var_binding(&mut self, pairs: Pair<Rule>) -> Result<Expr, Error<Rule>>;
    // Term productions
    fn visit_term_ident(&mut self, pairs: Pair<Rule>) -> Result<Expr, Error<Rule>>;
    fn visit_term_literal(&mut self, pairs: Pair<Rule>) -> Result<Expr, Error<Rule>>;
    fn visit_term_ident_list(
        &mut self,
        env: &Self::Env,
        pairs: Pair<Rule>,
    ) -> Result<Vec<Expr>, Error<Rule>>;
    // fn visit_term_attri(&mut self, pairs: Pair<Rule>) -> Result<&Self::Prog, Error<Rule>>;
    // fn visit_term_exist(&mut self, pairs: Pair<Rule>) -> Result<&Self::Prog, Error<Rule>>;
    // fn visit_term_forall(&mut self, pairs: Pair<Rule>) -> Result<&Self::Prog, Error<Rule>>;
    // fn visit_term_let(&mut self, pairs: Pair<Rule>) -> Result<&Self::Prog, Error<Rule>>;
    // BfTerm productions
    fn visit_bfterm_ident(&mut self, pairs: Pair<Rule>) -> Result<GExpr, Error<Rule>>;
    fn visit_bfterm_literal(&mut self, pairs: Pair<Rule>) -> Result<GExpr, Error<Rule>>;
    fn visit_bfterm_ident_list(
        &mut self,
        env: &Self::Env,
        pairs: Pair<Rule>,
    ) -> Result<Vec<GExpr>, Error<Rule>>;
    // fn visit_bfterm_attri(&mut self, pairs: Pair<Rule>) -> Result<&Self::Prog, Error<Rule>>;
    // Sort
    fn visit_sort(&mut self, pairs: Pair<Rule>) -> Result<Sort, Error<Rule>>;
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
    type Prog = SyGuSProg;
    type Env = Vec<(String, Sort)>;
    fn visit_main(&mut self, pairs: Pair<Rule>) -> Result<&SyGuSProg, Error<Rule>> {
        for pair in pairs.clone().into_inner() {
            match pair.as_rule() {
                Rule::SyGuS => {
                    self.visit_sygus(pair)?;
                }
                _ => continue,
            }
        }
        Ok(&self.sygus_prog)
    }
    // SyGuG
    fn visit_sygus(&mut self, pairs: Pair<Rule>) -> Result<&SyGuSProg, Error<Rule>> {
        for pair in pairs.clone().into_inner() {
            match pair.as_rule() {
                Rule::Cmd => {
                    self.visit_cmd(pair)?;
                }
                _ => unreachable!("SyGuS should only have Cmd as children"),
            }
        }
        Ok(&self.sygus_prog)
    }
    fn visit_cmd(&mut self, pairs: Pair<Rule>) -> Result<&SyGuSProg, Error<Rule>> {
        for pair in pairs.clone().into_inner() {
            let pair_str = format!("{:#?}", pair.clone());
            match pair.as_rule() {
                Rule::CheckSynthCmd => {
                    self.visit_check_synth(pair)?;
                }
                Rule::ConstraintCmd => {
                    let env = Vec::from_iter(
                        self.sygus_prog
                            .declare_var
                            .iter()
                            .map(|(k, v)| (k.clone(), v.clone())),
                    );
                    self.visit_constraint(&env, pair)?;
                }
                Rule::DeclareVarCmd => {
                    self.visit_declare_var(pair)?;
                }
                Rule::SynthFunCmd => {
                    self.visit_synth_fun(pair)?;
                }
                Rule::SmtCmd => {
                    self.visit_smt_cmd(pair)?;
                }
                _ => unreachable!("Cmd should only have CheckSynthCmd, ConstraintCmd, DeclareVarCmd, SynthFunCmd, or SmtCmd as children"),
            }
        }
        Ok(&self.sygus_prog)
    }
    fn visit_smt_cmd(&mut self, pairs: Pair<Rule>) -> Result<&SyGuSProg, Error<Rule>> {
        // panic!("{:#?}", pairs);
        let pairs_str = format!("{:#?}", pairs.clone());
        for pair in pairs.clone().into_inner() {
            let pair_str = format!("{:#?}", pair.clone());
            match pair.as_rule() {
                Rule::DefineFunCmd => {
                    self.visit_define_fun(pair)?;
                }
                Rule::SetLogicCmd => {
                    self.visit_set_logic(pair)?;
                }
                Rule::SetOptionCmd => {
                    self.visit_set_option(pair)?;
                }
                _ => continue,
            }
        }
        Ok(&self.sygus_prog)
    }
    // Cmd
    fn visit_check_synth(&mut self, pairs: Pair<Rule>) -> Result<&SyGuSProg, Error<Rule>> {
        // TODO: when running into this, we should call the solver with current collected SyGuSProg information
        // Implement in the last step
        Ok(&self.sygus_prog)
    }
    fn visit_constraint(
        &mut self,
        env: &Self::Env,
        pairs: Pair<Rule>,
    ) -> Result<&SyGuSProg, Error<Rule>> {
        for pair in pairs.clone().into_inner() {
            match pair.as_rule() {
                Rule::Term => {
                    let constr_expr = self.visit_term(env, pair)?;
                    self.sygus_prog.constraints.push(constr_expr);
                }
                _ => continue,
            }
        }
        Ok(&self.sygus_prog)
    }
    fn visit_declare_var(&mut self, pairs: Pair<Rule>) -> Result<&SyGuSProg, Error<Rule>> {
        let mut var_name = String::new();
        let mut var_sort = Sort::None;
        for pair in pairs.clone().into_inner() {
            match pair.as_rule() {
                Rule::Symbol => {
                    var_name = pair.as_str().to_string();
                }
                Rule::Sort => {
                    var_sort = self.visit_sort(pair)?;
                }
                _ => continue,
            }
        }
        if (!var_name.is_empty()) && (var_sort != Sort::None) {
            self.sygus_prog.declare_var.insert(var_name, var_sort);
        }
        Ok(&self.sygus_prog)
    }
    fn visit_synth_fun(&mut self, pairs: Pair<Rule>) -> Result<&SyGuSProg, Error<Rule>> {
        let mut sorted_vars = Vec::new();
        let mut func_name = String::new();
        let mut ret_sort = Sort::None;
        let mut body = Expr::Var("".to_string());
        let mut grammar_def = GrammarDef::new(); // optional
        for pair in pairs.clone().into_inner() {
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
                    body = self.visit_term(&sorted_vars, pair)?;
                }
                Rule::GrammarDef => {
                    grammar_def = self.visit_grammar_def(&sorted_vars, pair)?;
                }
                _ => continue,
            }
        }

        let synth_func = SynthFun {
            name: func_name.clone(),
            params: sorted_vars,
            return_type: ret_sort,
        };
        self.sygus_prog
            .synthe_func
            .insert(func_name, (synth_func, grammar_def));
        Ok(&self.sygus_prog)
    }
    // SmtCmd
    fn visit_define_fun(&mut self, pairs: Pair<Rule>) -> Result<&SyGuSProg, Error<Rule>> {
        let mut func_name = String::new();
        let mut sorted_vars = Vec::new();
        let mut ret_sort = Sort::None;
        let mut body = Expr::Var("".to_string());
        for pair in pairs.clone().into_inner() {
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
                    let env = sorted_vars.clone();
                    body = self.visit_term(&env, pair)?;
                }
                _ => continue,
            }
        }
        let func_body = FuncBody {
            name: func_name.clone(),
            params: sorted_vars,
            return_type: ret_sort,
            body: body.clone(),
        };
        self.sygus_prog.define_fun.insert(func_name, func_body);
        Ok(&self.sygus_prog)
    }
    fn visit_set_logic(&mut self, pairs: Pair<Rule>) -> Result<&SyGuSProg, Error<Rule>> {
        // only one child
        let logic = pairs.as_str();
        match logic {
            "LIA" => self.sygus_prog.set_logic = SetLogic::LIA,
            "BV" => self.sygus_prog.set_logic = SetLogic::BV,
            _ => self.sygus_prog.set_logic = SetLogic::Unknown,
        }
        Ok(&self.sygus_prog)
    }
    fn visit_set_option(&mut self, pairs: Pair<Rule>) -> Result<&SyGuSProg, Error<Rule>> {
        let mut opt_name = String::new();
        let mut opt_value = String::new();
        for pair in pairs.clone().into_inner() {
            match pair.as_rule() {
                Rule::Keyword => {
                    opt_name = pair.as_str().to_string();
                }
                Rule::Literal => {
                    opt_value = pair.as_str().to_string();
                }
                _ => continue,
            }
        }
        self.sygus_prog.set_option.insert(opt_name, opt_value);
        Ok(&self.sygus_prog)
    }
    // GrammarDef
    fn visit_grammar_def(
        &mut self,
        env: &Self::Env,
        pairs: Pair<Rule>,
    ) -> Result<GrammarDef, Error<Rule>> {
        let mut grammar_def = GrammarDef::new();
        for pair in pairs.clone().into_inner() {
            match pair.as_rule() {
                Rule::GroupedRuleList => {
                    let production = self.visit_grouped_rule_list(env, pair)?;
                    grammar_def.non_terminals.push(production);
                }
                _ => continue,
            }
        }
        Ok(grammar_def)
    }
    fn visit_grouped_rule_list(
        &mut self,
        env: &Self::Env,
        pairs: Pair<Rule>,
    ) -> Result<Production, Error<Rule>> {
        let mut production = Production::new();
        for pair in pairs.clone().into_inner() {
            match pair.as_rule() {
                Rule::Symbol => {
                    production.lhs = pair.as_str().to_string();
                }
                Rule::Sort => {
                    production.lhs_sort = self.visit_sort(pair)?;
                }
                Rule::GTerm => {
                    let gterm = self.visit_gterm(env, pair)?;
                    production.rhs.push(gterm);
                }
                _ => continue,
            }
        }
        Ok(production)
    }
    fn visit_gterm(&mut self, env: &Self::Env, pairs: Pair<Rule>) -> Result<GTerm, Error<Rule>> {
        let mut gterm = GTerm::None;
        // only one child
        let pair = pairs.clone().into_inner().next().unwrap();
        match pair.as_rule() {
            Rule::ConstGTerm => {
                let sort = self.visit_sort(pair)?;
                gterm = GTerm::Constant(sort);
            }
            Rule::VarGTerm => {
                let sort = self.visit_sort(pair)?;
                gterm = GTerm::Variable(sort);
            }
            Rule::BfTerm => {
                let bf_term = self.visit_bf_term(env, pair)?;
                gterm = GTerm::BfTerm(bf_term);
            }
            _ => unreachable!("gterm should only have ConstGTerm, VarGTerm, or BfTerm as children"),
        }
        Ok(gterm)
    }
    // Term
    fn visit_term(&mut self, env: &Self::Env, pairs: Pair<Rule>) -> Result<Expr, Error<Rule>> {
        let mut env = env.clone();
        let length = pairs.clone().into_inner().count();
        if length == 1 {
            let pair = pairs.clone().into_inner().next().unwrap();
            match pair.as_rule() {
                Rule::Identifier => self.visit_term_ident(pair),
                Rule::Literal => self.visit_term_literal(pair),
                _ => unreachable!(
                    "Term should only have Identifier or Literal as children when length is 1"
                ),
            }
        } else {
            let mut expr = Expr::Var("".to_string());
            let mut id = String::new();
            for pair in pairs.clone().into_inner() {
                match pair.as_rule() {
                    Rule::Identifier => {
                        id = pair.as_str().to_string();
                        // try to find id in env
                        if let Some((_, sort)) = env.iter().find(|(k, _)| k == &id) {
                            // if found, create a variable expression
                            expr = Expr::Var(id.clone());
                            env.push((id, sort.clone()));
                        } else {
                            panic!("Variable {} not found in environment", id);
                        }
                    }
                    Rule::Term => {
                        expr = self.visit_term(&env, pair)?;
                        // TODO: need to store the id, sort, and expr
                    }
                    _ => unreachable!(
                        "The third branch should only have Identifier or Term as children"
                    ),
                }
            }
            Ok(expr)
        }
    }
    fn visit_bf_term(&mut self, env: &Self::Env, pairs: Pair<Rule>) -> Result<GExpr, Error<Rule>> {
        let mut env = env.clone();
        let length = pairs.clone().into_inner().count();
        if length == 1 {
            let pair = pairs.clone().into_inner().next().unwrap();
            match pair.as_rule() {
                Rule::Identifier => self.visit_bfterm_ident(pair),
                Rule::Literal => self.visit_bfterm_literal(pair),
                _ => unreachable!(
                    "BfTerm should only have Identifier or Literal as children when length is 1"
                ),
            }
        } else {
            let mut expr = GExpr::Var("".to_string());
            let mut id = String::new();
            for pair in pairs.clone().into_inner() {
                match pair.as_rule() {
                    Rule::Identifier => {
                        id = pair.as_str().to_string();
                        // try to find id in env
                        if let Some((_, sort)) = env.iter().find(|(k, _)| k == &id) {
                            // if found, create a variable expression
                            expr = GExpr::Var(id.clone());
                            env.push((id, sort.clone()));
                        } else {
                            panic!("Variable {} not found in environment", id);
                        }
                    }
                    Rule::BfTerm => {
                        expr = self.visit_bf_term(&env, pair)?;
                        // TODO: need to store the id, sort, and expr
                    }
                    _ => unreachable!(
                        "The third branch should only have Identifier or Term as children"
                    ),
                }
            }
            Ok(expr)
        }
    }
    fn visit_sorted_var(&mut self, pairs: Pair<Rule>) -> Result<(Symbol, Sort), Error<Rule>> {
        let mut var_name = String::new();
        let mut var_sort = Sort::None;
        for pair in pairs.clone().into_inner() {
            match pair.as_rule() {
                Rule::Symbol => {
                    var_name = pair.as_str().to_string();
                }
                Rule::Sort => {
                    var_sort = self.visit_sort(pair)?;
                }
                _ => continue,
            }
        }
        Ok((var_name, var_sort))
    }
    // Term productions
    fn visit_term_ident(&mut self, pairs: Pair<Rule>) -> Result<Expr, Error<Rule>> {
        let id = pairs.as_str().to_string();
        Ok(Expr::Var(id))
    }
    fn visit_term_literal(&mut self, pairs: Pair<Rule>) -> Result<Expr, Error<Rule>> {
        let id = pairs.as_str().to_string();
        match pairs.clone().into_inner().next().unwrap().as_rule() {
            Rule::Numeral => {
                let val = id.parse::<i64>().unwrap();
                Ok(Expr::ConstInt(val))
            }
            Rule::BoolConst => {
                let val = id.parse::<bool>().unwrap();
                Ok(Expr::ConstBool(val))
            }
            Rule::HexConst => {
                let val = u64::from_str_radix(&id[2..], 16).unwrap();
                Ok(Expr::ConstBitVec(val))
            }
            Rule::StringConst => Ok(Expr::Var(id)),
            _ => unreachable!(),
        }
    }
    fn visit_term_ident_list(
        &mut self,
        env: &Self::Env,
        pairs: Pair<Rule>,
    ) -> Result<Vec<Expr>, Error<Rule>> {
        let mut exprs = Vec::new();
        for pair in pairs.clone().into_inner() {
            match pair.as_rule() {
                Rule::Identifier => {
                    let id = pair.as_str().to_string();
                    exprs.push(Expr::Var(id));
                }
                Rule::Literal => {
                    let id = pair.as_str().to_string();
                    match pair.into_inner().next().unwrap().as_rule() {
                        Rule::Numeral => {
                            let val = id.parse::<i64>().unwrap();
                            exprs.push(Expr::ConstInt(val));
                        }
                        Rule::BoolConst => {
                            let val = id.parse::<bool>().unwrap();
                            exprs.push(Expr::ConstBool(val));
                        }
                        Rule::HexConst => {
                            let val = u64::from_str_radix(&id[2..], 16).unwrap();
                            exprs.push(Expr::ConstBitVec(val));
                        }
                        Rule::StringConst => {
                            exprs.push(Expr::Var(id));
                        }
                        _ => continue,
                    }
                }
                Rule::Term => {
                    let expr = self.visit_term(env, pair)?;
                    exprs.push(expr);
                }
                _ => continue,
            }
        }
        Ok(exprs)
    }
    // BfTerm productions
    fn visit_bfterm_ident(&mut self, pairs: Pair<Rule>) -> Result<GExpr, Error<Rule>> {
        let id = pairs.as_str().to_string();
        Ok(GExpr::Var(id))
    }

    fn visit_bfterm_literal(&mut self, pairs: Pair<Rule>) -> Result<GExpr, Error<Rule>> {
        let id = pairs.as_str().to_string();
        match pairs.clone().into_inner().next().unwrap().as_rule() {
            Rule::Numeral => {
                let val = id.parse::<i64>().unwrap();
                Ok(GExpr::ConstInt(val))
            }
            Rule::BoolConst => {
                let val = id.parse::<bool>().unwrap();
                Ok(GExpr::ConstBool(val))
            }
            Rule::HexConst => {
                let val = u64::from_str_radix(&id[2..], 16).unwrap();
                Ok(GExpr::ConstBitVec(val))
            }
            Rule::StringConst => Ok(GExpr::Var(id)),
            _ => unreachable!(),
        }
    }

    fn visit_bfterm_ident_list(
        &mut self,
        env: &Self::Env,
        pairs: Pair<Rule>,
    ) -> Result<Vec<GExpr>, Error<Rule>> {
        let mut exprs = Vec::new();
        for pair in pairs.clone().into_inner() {
            match pair.as_rule() {
                Rule::Identifier => {
                    let id = pair.as_str().to_string();
                    exprs.push(GExpr::Var(id));
                }
                Rule::Literal => {
                    let id = pair.as_str().to_string();
                    match pair.into_inner().next().unwrap().as_rule() {
                        Rule::Numeral => {
                            let val = id.parse::<i64>().unwrap();
                            exprs.push(GExpr::ConstInt(val));
                        }
                        Rule::BoolConst => {
                            let val = id.parse::<bool>().unwrap();
                            exprs.push(GExpr::ConstBool(val));
                        }
                        Rule::HexConst => {
                            let val = u64::from_str_radix(&id[2..], 16).unwrap();
                            exprs.push(GExpr::ConstBitVec(val));
                        }
                        Rule::StringConst => {
                            exprs.push(GExpr::Var(id));
                        }
                        _ => continue,
                    }
                }
                Rule::BfTerm => {
                    let expr = self.visit_bfterm_ident(pair)?;
                    exprs.push(expr);
                }
                _ => continue,
            }
        }
        Ok(exprs)
    }
    // Sort
    fn visit_sort(&mut self, pairs: Pair<Rule>) -> Result<Sort, Error<Rule>> {
        let mut sort = Sort::None;
        for pair in pairs.clone().into_inner() {
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
                        _ => continue,
                    }
                }
                _ => continue,
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
        let project_root = env!("CARGO_MANIFEST_DIR");
        let filename = format!(
            "{}/benchmarks/bitvector-benchmarks/parity-AIG-d0.sl",
            project_root
        );
        let input = fs::read_to_string(&filename).unwrap();
        let res = parse(&input);
        let res = match res {
            Ok(res) => res,
            Err(e) => {
                panic!("Error parsing file: {}\nError: {:#?}", filename, e);
            }
        };
        // pretty print
        // println!("{:#?}", res);
        panic!("{:#?}", res);
    }
}
