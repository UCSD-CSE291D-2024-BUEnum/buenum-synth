use std::cell::RefCell;
use std::collections::HashMap;
use egg::{*, rewrite as rw};
use z3::ast::{Ast, Dynamic};
use z3::{Context, FuncDecl};

use crate::parser::{ast, ast::*};

use super::{GrammarTrait, ProgTrait};

define_language! {
    pub enum Language {
        ConstBool(bool),
        ConstInt(i64),
        ConstBitVec(u64),
        ConstString(String),
        // Var(ast::Symbol, ast::Sort), // TODO: unsupported multiple arguments with no pattern matching
        // FuncApply([Id; 2]),
        "not" = Not([Id; 1]),
        "and" = And([Id; 2]),
        "or" = Or([Id; 2]),
        "xor" = Xor([Id; 2]),
        "iff" = Iff([Id; 2]), // if and only if
        "=" = Equal([Id; 2]),
        "bvand" = BvAnd([Id; 2]),
        "bvor" = BvOr([Id; 2]),
        "bvxor" = BvXor([Id; 2]),
        "bvnot" = BvNot([Id; 1]),
        "bvadd" = BvAdd([Id; 2]),
        "bvmul" = BvMul([Id; 2]),
        "bvsub" = BvSub([Id; 2]),
        "bvudiv" = BvUdiv([Id; 2]), // Unsigned division
        "bvurem" = BvUrem([Id; 2]), // Unsigned remainder
        "bvshl" = BvShl([Id; 2]),  // Logical shift left
        "bvlshr" = BvLshr([Id; 2]), // Logical shift right
        "bvneg" = BvNeg([Id; 1]), // Negation
        "bvult" = BvUlt([Id; 2]),  // Unsigned less than
        // BvConst(u64, u32),
    }
}
impl Language {
    fn from_expr(enode: &Language) -> Expr {
        match enode {
            Language::ConstBool(b) => Expr::ConstBool(*b),
            Language::ConstInt(i) => Expr::ConstInt(*i),
            Language::ConstBitVec(u) => Expr::ConstBitVec(*u),
            Language::ConstString(s) => Expr::ConstString(s.clone()),
            // Language::Var(symbol, sort) => Expr::Var(symbol.clone(), sort.clone()),
            // Language::FuncApply(ids) => Expr::FuncApply(ids.clone()), // TODO: id to expr
            Language::Not(id) => Expr::Not(*id),
            Language::And(ids) => Expr::And(ids.clone()),
            Language::Or(ids) => Expr::Or(ids.clone()),
            Language::Xor(ids) => Expr::Xor(ids.clone()),
            Language::Iff(ids) => Expr::Iff(ids.clone()),
            Language::Equal(ids) => Expr::Equal(ids.clone()),
            Language::BvAnd(ids) => Expr::BvAnd(ids.clone()),
            Language::BvOr(ids) => Expr::BvOr(ids.clone()),
            Language::BvXor(ids) => Expr::BvXor(ids.clone()),
            Language::BvNot(id) => Expr::BvNot(*id),
            Language::BvAdd(ids) => Expr::BvAdd(ids.clone()),
            Language::BvMul(ids) => Expr::BvMul(ids.clone()),
            Language::BvSub(ids) => Expr::BvSub(ids.clone()),
            Language::BvUdiv(ids) => Expr::BvUdiv(ids.clone()),
            Language::BvUrem(ids) => Expr::BvUrem(ids.clone()),
            Language::BvShl(ids) => Expr::BvShl(ids.clone()),
            Language::BvLshr(ids) => Expr::BvLshr(ids.clone()),
            Language::BvNeg(id) => Expr::BvNeg(*id),
            Language::BvUlt(ids) => Expr::BvUlt(ids.clone()),
            // Language::BvConst(u, n) => Expr::BvConst(*u, *n),
        }
    }
}
pub trait SolverTrait {
    type Prog: ProgTrait;
    type Expr;
    type Grammar: GrammarTrait;
    type Constraint;
    type CounterExample;
    const MAX_SIZE: usize = 4;


    fn extract_grammar<'a>(&'a self, p: &'a Self::Prog, func_name: &str) -> &Self::Grammar;
    fn extract_constraint(&self, p: &Self::Prog) -> Self::Constraint;
    fn verify(&self, p: &Self::Prog, func_name: &str, expr: &Self::Expr) -> Option<Self::CounterExample>;
    fn expr_to_smt<'ctx>(
        &self,
        expr: &Self::Expr,
        vars: &'ctx HashMap<String, Dynamic<'ctx>>,
        funcs: &'ctx HashMap<String, FuncDecl<'ctx>>,
        ctx: &'ctx Context,
    ) -> Box<Dynamic<'ctx>>;
    fn sort_to_z3_sort<'ctx>(&self, sort: &Sort, ctx: &'ctx Context) -> z3::Sort<'ctx>;
    fn synthesize(&self, p: &Self::Prog, func_name: &str) -> Option<Self::Expr>;
}

struct EggSolver;

#[derive(Default)]
struct ObsEquiv {
    pts: Vec<HashMap<String, Expr>>,
}
impl Analysis<Language> for ObsEquiv {
    type Data = Vec<Expr>;

    fn merge(&mut self, to: &mut Self::Data, from: Self::Data) -> DidMerge {
        if to.len() != from.len() {
            *to = from;
            DidMerge(true, false)
        } else {
            let mut changed = false;
            for (t, f) in to.iter_mut().zip(from.iter()) {
                if t != f {
                    *t = f.clone();
                    changed = true;
                }
            }
            DidMerge(changed, false)
        }
    }

    fn make(egraph: &EGraph<Language, Self>, enode: &Language) -> Self::Data {
        let x = |i: &Id| &egraph[*i].data;
        match enode {
            Language::ConstBool(b) => vec![Expr::ConstBool(*b)],
            Language::ConstInt(i) => vec![Expr::ConstInt(*i)],
            Language::ConstBitVec(u) => vec![Expr::ConstBitVec(*u)],
            Language::ConstString(s) => vec![Expr::ConstString(s.clone())],
            Language::Var([sym, sort]) => vec![Expr::Var(sym.to_string(), sort.clone())],
            Language::GFuncApply([name, args]) => {
                let args_expr: Vec<GExpr> = args.iter().map(|&id| x(&id)[0].clone()).collect();
                vec![GExpr::GFuncApply(name.to_string(), args_expr)]
            }
            Language::FuncApply([name, args]) => {
                let args_expr: Vec<GExpr> = args.iter().map(|&id| x(&id)[0].clone()).collect();
                vec![GExpr::FuncApply(name.to_string(), args_expr)]
            }
            Language::Not([arg]) => vec![Expr::Not(Box::new(x(arg)[0].clone()))],
            Language::And([left, right]) => {
                vec![Expr::And(Box::new(x(left)[0].clone()), Box::new(x(right)[0].clone()))]
            }
            Language::Or([left, right]) => {
                vec![Expr::Or(Box::new(x(left)[0].clone()), Box::new(x(right)[0].clone()))]
            }
            Language::Xor([left, right]) => {
                vec![Expr::Xor(Box::new(x(left)[0].clone()), Box::new(x(right)[0].clone()))]
            }
            Language::Iff([left, right]) => {
                vec![Expr::Iff(Box::new(x(left)[0].clone()), Box::new(x(right)[0].clone()))]
            }
            Language::Equal([left, right]) => {
                vec![Expr::Equal(Box::new(x(left)[0].clone()), Box::new(x(right)[0].clone()))]
            }
            Language::BvAnd([left, right]) => {
                vec![Expr::BvAnd(Box::new(x(left)[0].clone()), Box::new(x(right)[0].clone()))]
            }
            Language::BvOr([left, right]) => {
                vec![Expr::BvOr(Box::new(x(left)[0].clone()), Box::new(x(right)[0].clone()))]
            }
            Language::BvXor([left, right]) => {
                vec![Expr::BvXor(Box::new(x(left)[0].clone()), Box::new(x(right)[0].clone()))]
            }
            Language::BvNot([arg]) => vec![Expr::BvNot(Box::new(x(arg)[0].clone()))],
            Language::BvAdd([left, right]) => {
                vec![Expr::BvAdd(Box::new(x(left)[0].clone()), Box::new(x(right)[0].clone()))]
            }
            Language::BvMul([left, right]) => {
                vec![Expr::BvMul(Box::new(x(left)[0].clone()), Box::new(x(right)[0].clone()))]
            }
            Language::BvSub([left, right]) => {
                vec![Expr::BvSub(Box::new(x(left)[0].clone()), Box::new(x(right)[0].clone()))]
            }
            Language::BvUdiv([left, right]) => {
                vec![Expr::BvUdiv(Box::new(x(left)[0].clone()), Box::new(x(right)[0].clone()))]
            }
            Language::BvUrem([left, right]) => {
                vec![Expr::BvUrem(Box::new(x(left)[0].clone()), Box::new(x(right)[0].clone()))]
            }
            Language::BvShl([left, right]) => {
                vec![Expr::BvShl(Box::new(x(left)[0].clone()), Box::new(x(right)[0].clone()))]
            }
            Language::BvLshr([left, right]) => {
                vec![Expr::BvLshr(Box::new(x(left)[0].clone()), Box::new(x(right)[0].clone()))]
            }
            Language::BvNeg([arg]) => vec![Expr::BvNeg(Box::new(x(arg)[0].clone()))],
            Language::BvUlt([left, right]) => {
                vec![Expr::BvUlt(Box::new(x(left)[0].clone()), Box::new(x(right)[0].clone()))]
            }
            Language::BvConst(val, width) => vec![Expr::BvConst(*val, *width)],
        }
    }

    fn modify(egraph: &mut EGraph<Language, Self>, id: Id) {
        if let obs_equiv_data = egraph[id].data.clone() {
            if let Some(added_expr) = obs_equiv_data.first() {
                let added = egraph.add(added_expr.clone());
                egraph.union(id, added);
            }
        }
    }
}


impl SolverTrait for EggSolver {
    type Prog = SyGuSProg;
    type Expr = Expr;
    type Grammar = GrammarDef;
    type Constraint = Vec<Expr>;
    type CounterExample = HashMap<String, Expr>;

    fn extract_grammar<'a>(&'a self, p: &'a Self::Prog, func_name: &str) -> &Self::Grammar {
        let synth = &p.synth_func;
        match synth.get(func_name) {
            Some((_, grammar)) => grammar,
            None => panic!("Function not found"),
        }
    }

    fn extract_constraint(&self, p: &Self::Prog) -> Self::Constraint {
        p.constraints.clone()
    }

    fn verify(&self, p: &Self::Prog, func_name: &str, expr: &Self::Expr) -> Option<Self::CounterExample> {
        unimplemented!()
    }

    fn expr_to_smt<'ctx>(
        &self,
        expr: &Self::Expr,
        vars: &'ctx HashMap<String, Dynamic<'ctx>>,
        funcs: &'ctx HashMap<String, FuncDecl<'ctx>>,
        ctx: &'ctx Context,
    ) -> Box<Dynamic<'ctx>> {
        unimplemented!()
    }

    fn sort_to_z3_sort<'ctx>(&self, sort: &Sort, ctx: &'ctx Context) -> z3::Sort<'ctx> {
        unimplemented!()
    }

    fn synthesize(&self, p: &Self::Prog, func_name: &str) -> Option<Self::Expr> {
        let counterexamples: RefCell<Vec<Self::CounterExample>> = RefCell::new(Vec::new());
        let constraint = self.extract_constraint(p);
        let g = self.extract_grammar(p, func_name);

        let runner = RefCell::new(
            Runner::<Language, ObsEquiv, ()>::default()
                .with_iter_limit(10) // Set the iteration limit
                .with_time_limit(std::time::Duration::from_secs(60)), // Set the time limit
        );

        loop {
            let pts = counterexamples.borrow();
            let mut enumerator = EggEnumerator::new(p, &g, &pts, &runner);

            if let Some(expr) = enumerator.enumerate(&g, &pts) {
                match self.verify(p, func_name, &expr) {
                    None => return Some(expr),
                    Some(cex) => counterexamples.borrow_mut().push(cex),
                }
            } else {
                if counterexamples.borrow().is_empty() {
                    return None;
                }
            }
        }
    }
}

trait Enumerator {
    type Expr;
    type Grammar;
    type CounterExample;

    fn new(prog: &SyGuSProg, grammar: &Self::Grammar, pts: &[Self::CounterExample], runner: &RefCell<Runner<GExpr, ObsEquiv, ()>>) -> Self;
    fn grow(&mut self, non_terminal: &ProdName);
    fn terms(&mut self, productions: &Production, d: usize, all_expressions: &mut Vec<GExpr>);
    fn enumerate(&mut self, grammar: &Self::Grammar, pts: &[Self::CounterExample]) -> Option<Self::Expr>;
}

struct EggEnumerator<'a> {
    prog: &'a SyGuSProg,
    grammar: &'a GrammarDef,
    pts: &'a [HashMap<String, Expr>],
    runner: &'a RefCell<Runner<Language, ObsEquiv, ()>>,
    cache: HashMap<(ProdName, usize), Vec<GExpr>>,
    current_size: usize,
}

impl<'a> Enumerator for EggEnumerator<'a> {
    type Expr = Expr;
    type Grammar = GrammarDef;
    type CounterExample = HashMap<String, Expr>;

    fn new(
        prog: &'a SyGuSProg,
        grammar: &'a GrammarDef,
        pts: &'a [HashMap<String, Expr>],
        runner: &'a RefCell<Runner<Language, ObsEquiv, ()>>,
    ) -> Self {
        EggEnumerator {
            prog,
            grammar,
            pts,
            runner,
            cache: HashMap::new(),
            current_size: 0,
        }
    }
    fn grow(&mut self, non_terminal: &ProdName) {
        let size = self.current_size;
        if let Some(productions) = self.grammar.non_terminals().iter().find(|p| p.lhs == *non_terminal) {
            let mut generated_terms: Vec<GExpr> = Vec::new();
            self.terms(productions, size, &mut generated_terms);
            self.cache.entry((non_terminal.clone(), size)).or_insert(generated_terms);
        }
    }
    fn terms(&mut self, productions: &Production, d: usize, all_expressions: &mut Vec<GExpr>) {
        unimplemented!();
    }
    fn enumerate(&mut self, grammar: &GrammarDef, pts: &[HashMap<String, Expr>]) -> Option<Expr> {
        loop {
            for non_terminal in grammar.non_terminals().iter().map(|p| &p.lhs) {
                if let Some(expressions) = self.cache.get(&(non_terminal.clone(), self.current_size)) {
                    if let Some(expr) = expressions.first() {
                        return Some(expr.to_expr());
                    }
                } else {
                    self.grow(non_terminal);
                    if let Some(expressions) = self.cache.get(&(non_terminal.clone(), self.current_size)) {
                        if let Some(expr) = expressions.first() {
                            return Some(expr.to_expr());
                        }
                    }
                }
            }
            self.current_size += 1;
            if self.current_size > 10 {
                // Set a maximum size limit
                return None;
            }
        }
    }
}
