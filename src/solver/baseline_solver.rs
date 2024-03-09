use std::collections::HashMap;

use crate::parser::ast::*;
use crate::solver::GrammarTrait;
use crate::solver::Solver;
pub struct BaselineSolver;

#[derive(Debug, Clone, PartialEq, Default)]
pub struct BaselineConstraint {
    // TODO: Define the constraint representation
    // At least, it should contain
}

#[derive(Debug, Clone, PartialEq, Default)]
pub struct BaselineCounterExample {
    // TODO: Define the counter-example representation
}

pub struct BaselineEnumerator<'a, S: Solver> {
    solver: &'a S,
    grammar: &'a S::Grammar,
    counterexamples: &'a [S::CounterExample],
    cache: HashMap<(ProdName, usize), Vec<GExpr>>,
    current_size: usize
}

impl<'a, S: Solver> BaselineEnumerator<'a, S> {
    pub fn new(solver: &'a S, grammar: &'a S::Grammar, counterexamples: &'a [S::CounterExample]) -> Self {
        BaselineEnumerator {
            solver,
            grammar,
            counterexamples,
            cache: HashMap::new(),
            current_size: 0
        }
    }

    fn grow(&mut self, non_terminal: &ProdName) {
        let size = self.current_size;
        if let Some(productions) = self.grammar.non_terminals().iter().find(|p| &p.lhs == non_terminal) {
            let mut expressions = Vec::new();
            for production in &productions.rhs {
                // TODO: implement the logic to generate expressions based on the production rules
            }
            self.cache.entry((non_terminal.clone(), size)).or_insert(expressions);
        }
    }
}
impl<'a, S: Solver> Iterator for BaselineEnumerator<'a, S> {
    type Item = Expr;

    fn next(&mut self) -> Option<Self::Item> {
        loop {
            for non_terminal in self.grammar.non_terminals().iter().map(|p| &p.lhs) {
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
            if self.current_size > S::MAX_SIZE {
                return None;
            }
        }
    }
}

impl Solver for BaselineSolver {
    type Prog = SyGuSProg;
    type Expr = Expr;
    type Grammar = GrammarDef;
    type Constraint = BaselineConstraint;
    type CounterExample = BaselineCounterExample;

    fn enumerate<'a>(
        &'a self,
        g: &'a Self::Grammar,
        c: &'a [Self::CounterExample]
    ) -> Box<dyn Iterator<Item = Self::Expr> + 'a> {
        let enumerator = BaselineEnumerator::new(self, g, c);
        Box::new(enumerator)
    }

    fn extract_grammar(&self, p: &Self::Prog, func_name: &str) -> Self::Grammar {
        // TODO: Extract the grammar for the specified function from the SyGuS program
        unimplemented!()
    }

    fn extract_constraint(&self, p: &Self::Prog) -> Self::Constraint {
        // TODO: Extract the constraint from the SyGuS program
        unimplemented!()
    }

    fn verify(&self, p: &Self::Prog, func_name: &str, expr: &Self::Expr) -> Result<(), Self::CounterExample> {
        // TODO: Verify the expression against the constraints in the SyGuS program
        unimplemented!()
    }
}
