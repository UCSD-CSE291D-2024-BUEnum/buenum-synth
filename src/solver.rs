pub mod baseline_solver;
pub mod egg_solver;

use std::cell::RefCell;

use crate::parser::ast;
use crate::parser::ast::ProdName;

use std::collections::HashMap;

pub trait Solver {
    type Prog: ProgTrait;
    type Expr;
    type Grammar: GrammarTrait;
    type Constraint;
    type CounterExample;
    const MAX_SIZE: usize = 10;

    fn enumerate<'a>(
        &'a self,
        g: &'a Self::Grammar,
        c: &'a [Self::CounterExample]
    ) -> Box<dyn Iterator<Item = Self::Expr> + 'a>;

    fn extract_grammar<'a>(&'a self, p: &'a Self::Prog, func_name: &str) -> &Self::Grammar;

    fn extract_constraint(&self, p: &Self::Prog) -> Self::Constraint;

    fn verify(&self, p: &Self::Prog, func_name: &str, expr: &Self::Expr) -> Option<Self::CounterExample>;

    fn expr_to_smt<'ctx>(&self, expr: &Self::Expr, vars: &'ctx HashMap<String, z3::ast::Dynamic>, funcs: &'ctx HashMap<String, z3::RecFuncDecl>, ctx: &'ctx z3::Context) -> Box<z3::ast::Dynamic<'ctx>>;

    fn sort_to_z3_sort<'ctx>(&self, sort: &self::ast::Sort, ctx: &'ctx z3::Context) -> z3::Sort<'ctx>;

    fn synthesize(&self, p: &Self::Prog, func_name: &str) -> Option<Self::Expr> {
        let counterexamples: RefCell<Vec<Self::CounterExample>> = RefCell::new(Vec::new()); // we need to use RefCell because we cannot decide the reference lifetime statically
        let constraint = self.extract_constraint(p);
        let g = self.extract_grammar(p, func_name);

        loop {
            let pts = counterexamples.borrow();
            let mut candidates = self.enumerate(&g, &pts);

            while let Some(expr) = candidates.next() {
                match self.verify(p, func_name, &expr) {
                    None => return Some(expr),
                    Some(cex) => counterexamples.borrow_mut().push(cex)
                }
            }
            if counterexamples.borrow().is_empty() {
                return None;
            }
        }
    }
}
pub trait ProgTrait {
    fn get_func_body(&self, func_name: &str) -> Option<&ast::FuncBody>;
    fn get_synth_func(&self, func_name: &str) -> Option<(&ast::SynthFun, &ast::GrammarDef)>;
    fn get_constraint(&self) -> &[ast::Expr];
}

impl ProgTrait for ast::SyGuSProg {
    fn get_func_body(&self, func_name: &str) -> Option<&ast::FuncBody> {
        self.define_fun.get(func_name)
    }

    fn get_synth_func(&self, func_name: &str) -> Option<(&ast::SynthFun, &ast::GrammarDef)> {
        self.synth_func
            .get(func_name)
            .map(|(synth_fun, grammar)| (synth_fun, grammar))
    }

    fn get_constraint(&self) -> &[ast::Expr] {
        &self.constraints
    }
}

pub trait GrammarTrait {
    fn non_terminals(&self) -> &[ast::Production];

    fn lhs_names(&self) -> Vec<&ProdName>;
}

impl GrammarTrait for ast::GrammarDef {
    fn non_terminals(&self) -> &[ast::Production] {
        &self.non_terminals
    }

    fn lhs_names(&self) -> Vec<&ProdName> {
        let mut names = Vec::<&ProdName>::new();
        for prod in self.non_terminals.iter() {
            names.push(&prod.lhs);
        }
        names
    }
}
