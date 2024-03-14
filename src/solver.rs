pub mod baseline_solver;
pub mod egg_solver;

use std::cell::RefCell;
use std::collections::HashMap;

use crate::parser::ast;

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

    fn extract_grammar(&self, p: &Self::Prog, func_name: &str) -> Self::Grammar;

    fn extract_constraint(&self, p: &Self::Prog) -> Self::Constraint;

    fn verify(&self, p: &Self::Prog, func_name: &str, expr: &Self::Expr) -> Option<Self::CounterExample>;

    fn expr_to_smt<'ctx>(&self, expr: &Self::Expr, vars: &'ctx HashMap<String, z3::ast::Dynamic>, funcs: &'ctx HashMap<String, z3::RecFuncDecl>, ctx: &'ctx z3::Context) -> Box<z3::ast::Dynamic<'ctx>>;

    // fn expr_to_smt<'a>(&'a self, expr: &Self::Expr, vars: &Vec<String>, ctx: &'a z3::Context) -> Box<dyn z3::ast::Ast<'a> + 'a>;

    // fn z3_ast_to_z3_bool(&self, ast: Box<dyn z3::ast::Ast>) -> Box<z3::ast::Bool>;

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
}

impl GrammarTrait for ast::GrammarDef {
    fn non_terminals(&self) -> &[ast::Production] {
        &self.non_terminals
    }
}

pub struct Enumerator<'a, S: Solver> {
    solver: &'a S,
    grammar: &'a S::Grammar,
    constraints: &'a [S::Constraint],
    cache: HashMap<(ast::ProdName, usize), Vec<ast::GExpr>>,
    current_size: usize
}

impl<'a, S: Solver> Enumerator<'a, S> {
    pub fn new(solver: &'a S, grammar: &'a S::Grammar, constraints: &'a [S::Constraint]) -> Self {
        Enumerator {
            solver,
            grammar,
            constraints,
            cache: HashMap::new(),
            current_size: 0
        }
    }

    fn grow(&mut self, non_terminal: &ast::ProdName) {
        let size = self.current_size;
        if let Some(productions) = self.grammar.non_terminals().iter().find(|p| &p.lhs == non_terminal) {
            let mut expressions = Vec::new();
            for production in &productions.rhs {
                // TODO: implement the logic to generate expressions based on the production rules
            }
            self.cache.insert((non_terminal.clone(), size), expressions);
        }
    }
}
impl<'a, S: Solver> Iterator for Enumerator<'a, S> {
    type Item = ast::GExpr;

    fn next(&mut self) -> Option<Self::Item> {
        loop {
            for non_terminal in self.grammar.non_terminals().iter().map(|p| &p.lhs) {
                if let Some(expressions) = self.cache.get(&(non_terminal.clone(), self.current_size)) {
                    if let Some(expr) = expressions.get(0) {
                        return Some(expr.clone());
                    }
                } else {
                    self.grow(non_terminal);
                    if let Some(expressions) = self.cache.get(&(non_terminal.clone(), self.current_size)) {
                        if let Some(expr) = expressions.get(0) {
                            return Some(expr.clone());
                        }
                    }
                }
            }
            self.current_size += 1;
            if self.current_size > S::MAX_SIZE {
                return None;
            }
        }
    }
}
