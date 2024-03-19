use std::cell::RefCell;
use std::collections::HashMap;
use egg::{*, rewrite as rw};
use z3::ast::{Ast, Dynamic};
use z3::{Context, FuncDecl};

use crate::parser::{ast, ast::*};

use super::{GrammarTrait, ProgTrait};
define_language! {
    pub enum EggExpr {
        "const_bool" = ConstBool(bool),
        "const_int" = ConstInt(i64),
        "const_bv" = ConstBitVec(u64),
        "const_string" = ConstString(String),
        "var" = Var((Symbol, Sort)),
        "not" = Not(Box<EggExpr>),
        "and" = And((Box<EggExpr>, Box<EggExpr>)),
        // ... other binary and unary operators ...
        "func_apply" = FuncApply((Symbol, Vec<Id>)),
    }
}
define_language! {
    pub enum SyGuSLang {
        ConstBool(bool),
        ConstInt(i64),
        ConstBitVec(u64),
        ConstString(String),
        Var(ast::Symbol), // TODO: unsupported multiple arguments with no pattern matching
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

        FuncApply(ast::Symbol, Vec<Id>),
    }
}

impl SyGuSLang {
    // fn to_expr(enode: &SyGuSLang) -> Expr {
    //     match enode {
    //         SyGuSLang::ConstBool(b) => Expr::ConstBool(*b),
    //         SyGuSLang::ConstInt(i) => Expr::ConstInt(*i),
    //         SyGuSLang::ConstBitVec(u) => Expr::ConstBitVec(*u),
    //         SyGuSLang::ConstString(s) => Expr::ConstString(s.clone()),
    //         SyGuSLang::Var(symbol) => Expr::Var(symbol.clone(), Sort::Bool),
    //         SyGuSLang::Not(id) => Expr::Not(*id),
    //         SyGuSLang::And(ids) => Expr::And(ids.clone()),
    //         SyGuSLang::Or(ids) => Expr::Or(ids.clone()),
    //         SyGuSLang::Xor(ids) => Expr::Xor(ids.clone()),
    //         SyGuSLang::Iff(ids) => Expr::Iff(ids.clone()),
    //         SyGuSLang::Equal(ids) => Expr::Equal(ids.clone()),
    //         SyGuSLang::BvAnd(ids) => Expr::BvAnd(ids.clone()),
    //         SyGuSLang::BvOr(ids) => Expr::BvOr(ids.clone()),
    //         SyGuSLang::BvXor(ids) => Expr::BvXor(ids.clone()),
    //         SyGuSLang::BvNot(id) => Expr::BvNot(*id),
    //         SyGuSLang::BvAdd(ids) => Expr::BvAdd(ids.clone()),
    //         SyGuSLang::BvMul(ids) => Expr::BvMul(ids.clone()),
    //         SyGuSLang::BvSub(ids) => Expr::BvSub(ids.clone()),
    //         SyGuSLang::BvUdiv(ids) => Expr::BvUdiv(ids.clone()),
    //         SyGuSLang::BvUrem(ids) => Expr::BvUrem(ids.clone()),
    //         SyGuSLang::BvShl(ids) => Expr::BvShl(ids.clone()),
    //         SyGuSLang::BvLshr(ids) => Expr::BvLshr(ids.clone()),
    //         SyGuSLang::BvNeg(id) => Expr::BvNeg(*id),
    //         SyGuSLang::BvUlt(ids) => Expr::BvUlt(ids.clone()),
    //         // SyGuSLang::BvConst(u, n) => Expr::BvConst(*u, *n),
    //         SyGuSLang::FuncApply(name, ids) => Expr::FuncApply(name.clone(), ids.clone()),
    //     }
    // }
    fn from_expr(expr: &ast::Expr) -> SyGuSLang {
        let expr = format!("{:?}", expr);
        let expr: SyGuSLang = expr.parse().unwrap();

        match expr {
            ast::Expr::ConstBool(b) => SyGuSLang::ConstBool(*b),
            ast::Expr::ConstInt(i) => SyGuSLang::ConstInt(*i),
            ast::Expr::ConstBitVec(u) => SyGuSLang::ConstBitVec(*u),
            ast::Expr::ConstString(s) => SyGuSLang::ConstString(s.clone()),
            ast::Expr::Var(symbol, _) => SyGuSLang::Var(symbol.clone()),
            ast::Expr::Not(expr) => {
                let expr = expr.clone();
                SyGuSLang::Not()
            }
            ast::Expr::And(ids) => SyGuSLang::And([ids[0], ids[1]]),
            ast::Expr::Or(ids) => SyGuSLang::Or([ids[0], ids[1]]),
            ast::Expr::Xor(ids) => SyGuSLang::Xor([ids[0], ids[1]]),
            ast::Expr::Iff(ids) => SyGuSLang::Iff([ids[0], ids[1]]),
            ast::Expr::Equal(ids) => SyGuSLang::Equal([ids[0], ids[1]]),
            ast::Expr::BvAnd(ids) => SyGuSLang::BvAnd([ids[0], ids[1]]),
            ast::Expr::BvOr(ids) => SyGuSLang::BvOr([ids[0], ids[1]]),
            ast::Expr::BvXor(ids) => SyGuSLang::BvXor([ids[0], ids[1]]),
            ast::Expr::BvNot(id) => SyGuSLang::BvNot([*id]),
            ast::Expr::BvAdd(ids) => SyGuSLang::BvAdd([ids[0], ids[1]]),
            ast::Expr::BvMul(ids) => SyGuSLang::BvMul([ids[0], ids[1]]),
            ast::Expr::BvSub(ids) => SyGuSLang::BvSub([ids[0], ids[1]]),
            ast::Expr::BvUdiv(ids) => SyGuSLang::BvUdiv([ids[0], ids[1]]),
            ast::Expr::BvUrem(ids) => SyGuSLang::BvUrem([ids[0], ids[1]]),
            ast::Expr::BvShl(ids) => SyGuSLang::BvShl([ids[0], ids[1]]),
            ast::Expr::BvLshr(ids) => SyGuSLang::BvLshr([ids[0], ids[1]]),
            ast::Expr::BvNeg(id) => SyGuSLang::BvNeg([*id]),
            ast::Expr::BvUlt(ids) => SyGuSLang::BvUlt([ids[0], ids[1]]),
            ast::Expr::FuncApply(name, ids) => SyGuSLang::FuncApply(name.clone(), ids.clone()),
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
impl Analysis<SyGuSLang> for ObsEquiv {
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

    fn make(egraph: &EGraph<SyGuSLang, Self>, enode: &SyGuSLang) -> Self::Data {
        let x = |i: &Id| &egraph[*i].data;
        match enode {
            SyGuSLang::ConstBool(b) => vec![Expr::ConstBool(*b)],
            SyGuSLang::ConstInt(i) => vec![Expr::ConstInt(*i)],
            SyGuSLang::ConstBitVec(u) => vec![Expr::ConstBitVec(*u)],
            SyGuSLang::ConstString(s) => vec![Expr::ConstString(s.clone())],
            SyGuSLang::Var([sym, sort]) => vec![Expr::Var(sym.to_string(), sort.clone())],
            SyGuSLang::GFuncApply([name, args]) => {
                let args_expr: Vec<GExpr> = args.iter().map(|&id| x(&id)[0].clone()).collect();
                vec![GExpr::GFuncApply(name.to_string(), args_expr)]
            }
            SyGuSLang::FuncApply([name, args]) => {
                let args_expr: Vec<GExpr> = args.iter().map(|&id| x(&id)[0].clone()).collect();
                vec![GExpr::FuncApply(name.to_string(), args_expr)]
            }
            SyGuSLang::Not([arg]) => vec![Expr::Not(Box::new(x(arg)[0].clone()))],
            SyGuSLang::And([left, right]) => {
                vec![Expr::And(Box::new(x(left)[0].clone()), Box::new(x(right)[0].clone()))]
            }
            SyGuSLang::Or([left, right]) => {
                vec![Expr::Or(Box::new(x(left)[0].clone()), Box::new(x(right)[0].clone()))]
            }
            SyGuSLang::Xor([left, right]) => {
                vec![Expr::Xor(Box::new(x(left)[0].clone()), Box::new(x(right)[0].clone()))]
            }
            SyGuSLang::Iff([left, right]) => {
                vec![Expr::Iff(Box::new(x(left)[0].clone()), Box::new(x(right)[0].clone()))]
            }
            SyGuSLang::Equal([left, right]) => {
                vec![Expr::Equal(Box::new(x(left)[0].clone()), Box::new(x(right)[0].clone()))]
            }
            SyGuSLang::BvAnd([left, right]) => {
                vec![Expr::BvAnd(Box::new(x(left)[0].clone()), Box::new(x(right)[0].clone()))]
            }
            SyGuSLang::BvOr([left, right]) => {
                vec![Expr::BvOr(Box::new(x(left)[0].clone()), Box::new(x(right)[0].clone()))]
            }
            SyGuSLang::BvXor([left, right]) => {
                vec![Expr::BvXor(Box::new(x(left)[0].clone()), Box::new(x(right)[0].clone()))]
            }
            SyGuSLang::BvNot([arg]) => vec![Expr::BvNot(Box::new(x(arg)[0].clone()))],
            SyGuSLang::BvAdd([left, right]) => {
                vec![Expr::BvAdd(Box::new(x(left)[0].clone()), Box::new(x(right)[0].clone()))]
            }
            SyGuSLang::BvMul([left, right]) => {
                vec![Expr::BvMul(Box::new(x(left)[0].clone()), Box::new(x(right)[0].clone()))]
            }
            SyGuSLang::BvSub([left, right]) => {
                vec![Expr::BvSub(Box::new(x(left)[0].clone()), Box::new(x(right)[0].clone()))]
            }
            SyGuSLang::BvUdiv([left, right]) => {
                vec![Expr::BvUdiv(Box::new(x(left)[0].clone()), Box::new(x(right)[0].clone()))]
            }
            SyGuSLang::BvUrem([left, right]) => {
                vec![Expr::BvUrem(Box::new(x(left)[0].clone()), Box::new(x(right)[0].clone()))]
            }
            SyGuSLang::BvShl([left, right]) => {
                vec![Expr::BvShl(Box::new(x(left)[0].clone()), Box::new(x(right)[0].clone()))]
            }
            SyGuSLang::BvLshr([left, right]) => {
                vec![Expr::BvLshr(Box::new(x(left)[0].clone()), Box::new(x(right)[0].clone()))]
            }
            SyGuSLang::BvNeg([arg]) => vec![Expr::BvNeg(Box::new(x(arg)[0].clone()))],
            SyGuSLang::BvUlt([left, right]) => {
                vec![Expr::BvUlt(Box::new(x(left)[0].clone()), Box::new(x(right)[0].clone()))]
            }
            SyGuSLang::BvConst(val, width) => vec![Expr::BvConst(*val, *width)],
        }
    }

    fn modify(egraph: &mut EGraph<SyGuSLang, Self>, id: Id) {
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
            Runner::<SyGuSLang, ObsEquiv, ()>::default()
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
    runner: &'a RefCell<Runner<SyGuSLang, ObsEquiv, ()>>,
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
        runner: &'a RefCell<Runner<SyGuSLang, ObsEquiv, ()>>,
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

#[cfg(test)]
mod tests {
    use super::*;
    use crate::parser::parse;
    use std::fs;

    #[test]
    fn test_synthesize() {
        env_logger::init();
        let content = fs::read_to_string("examples/and.sygus").unwrap();
        let prog = parse(&content).unwrap();
        let solver = EggSolver;
        let expr = solver.synthesize(&prog, "and");
        assert_eq!(expr, Some(Expr::And(Box::new(Expr::Var("x".to_string(), Sort::Bool)), Box::new(Expr::Var("y".to_string(), Sort::Bool)))));
    }
}