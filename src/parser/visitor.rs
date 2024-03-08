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
    fn visit_term_ident(&mut self, env: &Self::Env, pairs: Pair<Rule>)
        -> Result<Expr, Error<Rule>>;
    fn visit_term_literal(
        &mut self,
        env: &Self::Env,
        pairs: Pair<Rule>,
    ) -> Result<Expr, Error<Rule>>;
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
    fn visit_bfterm_ident(
        &mut self,
        env: &Self::Env,
        pairs: Pair<Rule>,
    ) -> Result<GExpr, Error<Rule>>;
    fn visit_bfterm_literal(
        &mut self,
        env: &Self::Env,
        pairs: Pair<Rule>,
    ) -> Result<GExpr, Error<Rule>>;
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
macro_rules! parse_expr {
    ($id:ident, $expr_type:ty, $self:ident, $env:expr, $pairs:expr, $visit_method:ident) => {
        match $id.as_str() {
            // Unary Operators
            "not" => <$expr_type>::Not(Box::new(
                $self.$visit_method($env, $pairs.clone().into_inner().next().unwrap())?,
            )),
            // Binary Operators
            "=" | "and" | "or" | "xor" | "iff" | "bvand" | "bvor" | "bvxor" | "bvadd" | "bvmul"
            | "bvsub" | "bvudiv" | "bvurem" | "bvshl" | "bvlshr" | "bvult" => {
                let mut exprs = Vec::new();
                for pair in $pairs.clone().into_inner() {
                    let expr = $self.$visit_method($env, pair)?;
                    exprs.push(expr);
                }
                match $id.as_str() {
                    "=" => {
                        <$expr_type>::Equal(Box::new(exprs[0].clone()), Box::new(exprs[1].clone()))
                    }
                    "and" => {
                        <$expr_type>::And(Box::new(exprs[0].clone()), Box::new(exprs[1].clone()))
                    }
                    "or" => {
                        <$expr_type>::Or(Box::new(exprs[0].clone()), Box::new(exprs[1].clone()))
                    }
                    "xor" => {
                        <$expr_type>::Xor(Box::new(exprs[0].clone()), Box::new(exprs[1].clone()))
                    }
                    "iff" => {
                        <$expr_type>::Iff(Box::new(exprs[0].clone()), Box::new(exprs[1].clone()))
                    }
                    "bvand" => {
                        <$expr_type>::BvAnd(Box::new(exprs[0].clone()), Box::new(exprs[1].clone()))
                    }
                    "bvor" => {
                        <$expr_type>::BvOr(Box::new(exprs[0].clone()), Box::new(exprs[1].clone()))
                    }
                    "bvxor" => {
                        <$expr_type>::BvXor(Box::new(exprs[0].clone()), Box::new(exprs[1].clone()))
                    }
                    "bvadd" => {
                        <$expr_type>::BvAdd(Box::new(exprs[0].clone()), Box::new(exprs[1].clone()))
                    }
                    "bvmul" => {
                        <$expr_type>::BvMul(Box::new(exprs[0].clone()), Box::new(exprs[1].clone()))
                    }
                    "bvsub" => {
                        <$expr_type>::BvSub(Box::new(exprs[0].clone()), Box::new(exprs[1].clone()))
                    }
                    "bvudiv" => {
                        <$expr_type>::BvUdiv(Box::new(exprs[0].clone()), Box::new(exprs[1].clone()))
                    }
                    "bvurem" => {
                        <$expr_type>::BvUrem(Box::new(exprs[0].clone()), Box::new(exprs[1].clone()))
                    }
                    "bvshl" => {
                        <$expr_type>::BvShl(Box::new(exprs[0].clone()), Box::new(exprs[1].clone()))
                    }
                    "bvlshr" => {
                        <$expr_type>::BvLshr(Box::new(exprs[0].clone()), Box::new(exprs[1].clone()))
                    }
                    "bvult" => {
                        <$expr_type>::BvUlt(Box::new(exprs[0].clone()), Box::new(exprs[1].clone()))
                    }
                    _ => unreachable!(),
                }
            }
            "bvnot" => <$expr_type>::BvNot(Box::new(
                $self.$visit_method($env, $pairs.clone().into_inner().next().unwrap())?,
            )),
            "bvneg" => <$expr_type>::BvNeg(Box::new(
                $self.$visit_method($env, $pairs.clone().into_inner().next().unwrap())?,
            )),
            _ => panic!(
                "Unknown operator: {}\nCurrent environment: {:#?}",
                $id, $env
            ),
        }
    };
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
                    let mut env = Vec::from_iter(
                        self.sygus_prog
                            .declare_var
                            .iter()
                            .map(|(k, v)| (k.clone(), v.clone())),
                    );
                    // extend the funtions name from both define_fun and synth_fun
                    env.extend(
                        self.sygus_prog
                            .define_fun
                            .iter()
                            .map(|(k, v)| (k.clone(), v.return_type.clone())),
                    );
                    env.extend(
                        self.sygus_prog
                            .synthe_func
                            .iter()
                            .map(|(k, (v, _))| (k.clone(), v.return_type.clone())),
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
        let mut body = Expr::Var("".to_string(), Sort::None);
        let pairs_str = format!("{:#?}", pairs.clone());
        for pair in pairs.clone().into_inner() {
            let pair_str = format!("{:#?}", pair.clone());
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
                _ => unreachable!(
                    "define_fun should only have Symbol, SortedVar, Sort, or Term as children"
                ),
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
        let pair = pairs.clone().into_inner().next().unwrap();
        let pair_str = pair.as_str();
        match pair.as_str() {
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
        let env_str = format!("{:#?}", env);
        let length = pairs.clone().into_inner().count();
        let pairs_str = format!("{:#?}", pairs.clone());
        if length == 0 {
            // just the pairs itself
            let pair = pairs.clone();
            let pair_str = format!("{:#?}", pair.clone());
            match pair.as_rule() {
                Rule::Identifier => self.visit_term_ident(&env, pair),
                Rule::Literal => self.visit_term_literal(&env, pair),
                _ => unreachable!(
                    "Term should only have Identifier or Literal as children when length is 0"
                ),
            }
        } else if length <= 1 {
            let pair = pairs.clone().into_inner().next().unwrap();
            match pair.as_rule() {
                Rule::Identifier => self.visit_term_ident(&env, pair),
                Rule::Literal => self.visit_term_literal(&env ,pair),
                other => unreachable!(
                    "Term should only have Identifier or Literal as children when length is 1, but got {:?}", other
                ),
            }
        } else {
            let mut expr = Expr::Var("".to_string(), Sort::None);
            let mut id = String::new();
            let length = pairs.clone().into_inner().count();
            let pairs_str = format!("{:#?}", pairs.clone());
            for pair in pairs.clone().into_inner() {
                let pair_str = format!("{:#?}", pair.clone());
                match pair.as_rule() {
                    Rule::Identifier => {
                        id = pair.as_str().to_string();
                        // try to find id in env
                        if let Some((_, sort)) = env.iter().find(|(k, _)| k == &id) {
                            // if found, create a variable expression
                            expr = Expr::Var(id.clone(), sort.clone());
                            env.push((id, sort.clone()));
                        } else {
                            // expr = parse_expr!(id, Expr, self, &env, pairs.clone(), visit_term);
                            // expand the parse_expr! macro below
                            match id.as_str() {
                                // Unary Operators
                                "not" => {
                                    expr = Expr::Not(Box::new(self.visit_term(
                                        &env,
                                        pairs.clone().into_inner().next().unwrap(),
                                    )?));
                                }
                                // Binary Operators
                                "=" | "and" | "or" | "xor" | "iff" | "bvand" | "bvor" | "bvxor"
                                | "bvadd" | "bvmul" | "bvsub" | "bvudiv" | "bvurem" | "bvshl"
                                | "bvlshr" | "bvult" => {
                                    let mut exprs = Vec::new();
                                    for pair in pairs.clone().into_inner() {
                                        let expr = self.visit_term(&env, pair)?;
                                        exprs.push(expr);
                                    }
                                    match id.as_str() {
                                        "=" => {
                                            expr = Expr::Equal(
                                                Box::new(exprs[0].clone()),
                                                Box::new(exprs[1].clone()),
                                            )
                                        }
                                        "and" => {
                                            expr = Expr::And(
                                                Box::new(exprs[0].clone()),
                                                Box::new(exprs[1].clone()),
                                            )
                                        }
                                        "or" => {
                                            expr = Expr::Or(
                                                Box::new(exprs[0].clone()),
                                                Box::new(exprs[1].clone()),
                                            )
                                        }
                                        "xor" => {
                                            expr = Expr::Xor(
                                                Box::new(exprs[0].clone()),
                                                Box::new(exprs[1].clone()),
                                            )
                                        }
                                        "iff" => {
                                            expr = Expr::Iff(
                                                Box::new(exprs[0].clone()),
                                                Box::new(exprs[1].clone()),
                                            )
                                        }
                                        "bvand" => {
                                            expr = Expr::BvAnd(
                                                Box::new(exprs[0].clone()),
                                                Box::new(exprs[1].clone()),
                                            )
                                        }
                                        "bvor" => {
                                            expr = Expr::BvOr(
                                                Box::new(exprs[0].clone()),
                                                Box::new(exprs[1].clone()),
                                            )
                                        }
                                        "bvxor" => {
                                            expr = Expr::BvXor(
                                                Box::new(exprs[0].clone()),
                                                Box::new(exprs[1].clone()),
                                            )
                                        }
                                        "bvadd" => {
                                            expr = Expr::BvAdd(
                                                Box::new(exprs[0].clone()),
                                                Box::new(exprs[1].clone()),
                                            )
                                        }
                                        "bvmul" => {
                                            expr = Expr::BvMul(
                                                Box::new(exprs[0].clone()),
                                                Box::new(exprs[1].clone()),
                                            )
                                        }
                                        "bvsub" => {
                                            expr = Expr::BvSub(
                                                Box::new(exprs[0].clone()),
                                                Box::new(exprs[1].clone()),
                                            )
                                        }
                                        "bvudiv" => {
                                            expr = Expr::BvUdiv(
                                                Box::new(exprs[0].clone()),
                                                Box::new(exprs[1].clone()),
                                            )
                                        }
                                        "bvurem" => {
                                            expr = Expr::BvUrem(
                                                Box::new(exprs[0].clone()),
                                                Box::new(exprs[1].clone()),
                                            )
                                        }
                                        "bvshl" => {
                                            expr = Expr::BvShl(
                                                Box::new(exprs[0].clone()),
                                                Box::new(exprs[1].clone()),
                                            )
                                        }
                                        "bvlshr" => {
                                            expr = Expr::BvLshr(
                                                Box::new(exprs[0].clone()),
                                                Box::new(exprs[1].clone()),
                                            )
                                        }
                                        "bvult" => {
                                            expr = Expr::BvUlt(
                                                Box::new(exprs[0].clone()),
                                                Box::new(exprs[1].clone()),
                                            )
                                        }
                                        _ => unreachable!(),
                                    }
                                }
                                _ => panic!(
                                    "Unknown operator: {}\nCurrent environment: {:#?}",
                                    id, env
                                ),
                            }
                        }
                    }
                    Rule::Literal => {
                        id = pair.as_str().to_string();
                        match pair.into_inner().next().unwrap().as_rule() {
                            Rule::Numeral => {
                                let val = id.parse::<i64>().unwrap();
                                expr = Expr::ConstInt(val);
                            }
                            Rule::BoolConst => {
                                let val = id.parse::<bool>().unwrap();
                                expr = Expr::ConstBool(val);
                            }
                            Rule::HexConst => {
                                let val = u64::from_str_radix(&id[2..], 16).unwrap();
                                expr = Expr::ConstBitVec(val);
                            }
                            Rule::StringConst => expr = Expr::Var(id, Sort::String),
                            _ => {
                                // skip Decimal, BinConst
                            }
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
        if length == 0 {
            // just the pairs itself
            let pair = pairs.clone();
            match pair.as_rule() {
                Rule::Identifier => self.visit_bfterm_ident(&env, pair),
                Rule::Literal => self.visit_bfterm_literal(&env, pair),
                _ => unreachable!(
                    "BfTerm should only have Identifier or Literal as children when length is 0"
                ),
            }
        } else if length <= 1 {
            let pair = pairs.clone().into_inner().next().unwrap();
            match pair.as_rule() {
                Rule::Identifier => self.visit_bfterm_ident(&env, pair),
                Rule::Literal => self.visit_bfterm_literal(&env, pair),
                _ => unreachable!(
                    "BfTerm should only have Identifier or Literal as children when length is 1"
                ),
            }
        } else {
            let mut expr = GExpr::Var("".to_string(), Sort::None);
            let mut id = String::new();
            for pair in pairs.clone().into_inner() {
                match pair.as_rule() {
                    Rule::Identifier => {
                        id = pair.as_str().to_string();
                        // try to find id in env
                        if let Some((_, sort)) = env.iter().find(|(k, _)| k == &id) {
                            // if found, create a variable expression
                            expr = GExpr::Var(id.clone(), sort.clone());
                            env.push((id, sort.clone()));
                        } else {
                            expr = parse_expr!(id, GExpr, self, &env, pairs.clone(), visit_bf_term);
                        }
                    }
                    Rule::BfTerm => {
                        expr = self.visit_bf_term(&env, pair)?;
                        // TODO: need to store the id, sort, and expr
                    }
                    _ => unreachable!(
                        "The third branch should only have Identifier or BfTerm as children"
                    ),
                }
            }
            Ok(expr)
        }
    }
    fn visit_sorted_var(&mut self, pairs: Pair<Rule>) -> Result<(Symbol, Sort), Error<Rule>> {
        let mut var_name = String::new();
        let mut var_sort = Sort::None;
        let pairs_str = format!("{:#?}", pairs.clone());
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
    fn visit_term_ident(
        &mut self,
        env: &Self::Env,
        pairs: Pair<Rule>,
    ) -> Result<Expr, Error<Rule>> {
        let id = pairs.as_str().to_string();
        let sort = env.iter().find(|(k, _)| k == &id).unwrap().1.clone(); // TODO: crash on unwrap the operator
        Ok(Expr::Var(id, sort))
    }
    fn visit_term_literal(
        &mut self,
        env: &Self::Env,
        pairs: Pair<Rule>,
    ) -> Result<Expr, Error<Rule>> {
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
            Rule::StringConst => Ok(Expr::Var(id, Sort::None)), // TODO
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
                    exprs.push(Expr::Var(id, Sort::None));
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
                            exprs.push(Expr::Var(id, Sort::None));
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
    fn visit_bfterm_ident(
        &mut self,
        env: &Self::Env,
        pairs: Pair<Rule>,
    ) -> Result<GExpr, Error<Rule>> {
        let id = pairs.as_str().to_string();
        let sort = env.iter().find(|(k, _)| k == &id).unwrap().1.clone();
        Ok(GExpr::Var(id, sort))
    }

    fn visit_bfterm_literal(
        &mut self,
        env: &Self::Env,
        pairs: Pair<Rule>,
    ) -> Result<GExpr, Error<Rule>> {
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
            Rule::StringConst => Ok(GExpr::Var(id, Sort::String)),
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
                    exprs.push(GExpr::Var(id, Sort::None));
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
                            exprs.push(GExpr::Var(id, Sort::None));
                        }
                        _ => continue,
                    }
                }
                Rule::BfTerm => {
                    let expr = self.visit_bfterm_ident(&env, pair)?;
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
        let pairs_str = format!("{:#?}", pairs.clone());
        for pair in pairs.clone().into_inner() {
            match pair.as_rule() {
                Rule::Identifier => match pair.as_str() {
                    "Bool" => sort = Sort::Bool,
                    "Int" => sort = Sort::Int,
                    "BitVec" => {
                        let bit_width = pair.as_str().parse::<i32>().unwrap();
                        sort = Sort::BitVec(bit_width);
                    }
                    _ => unreachable!("Unknown sort"),
                },
                Rule::Sort => {
                    sort = self.visit_sort(pair)?;
                }
                _ => unreachable!("Sort should only have Identifier or Sort as children"),
            }
        }
        Ok(sort)
    }
}

#[cfg(test)]
mod tests {
    use std::fs;

    use crate::parser::parse;

    const PROJECT_ROOT: &str = env!("CARGO_MANIFEST_DIR");
    macro_rules! check_parse_tree {
        ($relative_path:expr) => {
            let filename = format!("{}/{}", PROJECT_ROOT, $relative_path);
            let input = fs::read_to_string(&filename).unwrap();
            let res = parse(&input);
            let res = match res {
                Ok(res) => res,
                Err(e) => {
                    panic!("Error parsing file: {}\nError: {:#?}", filename, e);
                }
            };
            panic!("{:#?}", res);
        };
    }
    #[test]
    fn test_visit_main_eg1() {
        check_parse_tree!(format!("benchmarks/bitvector-benchmarks/parity-AIG-d0.sl"));
    }
    #[test]
    fn test_visit_main_eg2() {
        check_parse_tree!(format!("benchmarks/bitvector-benchmarks/parity-NAND-d1.sl"));
    }
    #[test]
    fn test_visit_main_eg3() {
        check_parse_tree!(format!("benchmarks/hackers_del/hd-20-d1-prog.sl"));
    }
}
