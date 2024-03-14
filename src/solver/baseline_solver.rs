use crate::solver::ast::GTerm;
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
    current_size: usize,
}

// d: enumeration depth, k: number of non-terminals in rhs
pub fn permutation(d: i32, k: i32, ans: &mut Vec<Vec<i32>>, curr: &mut Vec<i32>){
    if k == 0 {
        if d == 1 {
            ans.push(curr.clone());
        }
        return;
    }
    for i in 0..d {
        curr.push(i);
        permutation(d - i, k - 1, ans, curr);
        curr.remove(curr.len()-1);
    }
}

impl<'a, S: Solver> BaselineEnumerator<'a, S> {
    pub fn new(solver: &'a S, grammar: &'a S::Grammar, counterexamples: &'a [S::CounterExample]) -> Self {
        BaselineEnumerator {
            solver,
            grammar,
            counterexamples,
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


    fn terms(&self, productions: &Production, d: usize, all_expressions: &mut Vec<GExpr>) {
        let lhs = &productions.lhs;
        let rhs = &productions.rhs;
        for production in rhs {
            if d == 0 {
                match production {
                    GTerm::BfTerm(expr) => {
                        match expr {
                            GExpr::ConstBool(_)
                            | GExpr::ConstInt(_)
                            | GExpr::ConstBitVec(_)
                            | GExpr::ConstString(_)
                            | GExpr::Var(_, _)
                            | GExpr::BvConst(_, _)
                            => all_expressions.push(expr.clone()),
                            _ => {}
                        }
                    }
                    _ => eprint!("Unsupported Production Type!"),
                }
            } else {
                for name in self.grammar.lhs_names() {
                    println!("{}", name);
                }
                loop {

                }
            }
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
                            //let v = expr.to_expr().eval(&Default::default());
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
        c: &'a [Self::CounterExample],
    ) -> Box<dyn Iterator<Item=Self::Expr> + 'a> {
        let enumerator = BaselineEnumerator::new(self, g, c);
        Box::new(enumerator)
    }

    fn extract_grammar<'a>(&'a self, p: &'a Self::Prog, func_name: &str) -> &'a Self::Grammar {
        let synth = &p.synth_func;
        match synth.get(func_name) {
            Some((_, grammar)) => grammar,
            None => panic!("Function not found"), // You might handle the None case differently
        }
    }


    fn extract_constraint(&self, p: &Self::Prog) -> &Self::Constraint {
        // TODO: Extract the constraint from the SyGuS program
        let constraints = &p.constraints;
        unimplemented!()
    }

    fn verify(&self, p: &Self::Prog, func_name: &str, expr: &Self::Expr) -> Result<(), Self::CounterExample> {
        // TODO: Verify the expression against the constraints in the SyGuS program
        unimplemented!()
    }
}


mod tests {
    use super::*;

    #[test]
    fn test_permutation_1() {
        let mut ans = Vec::new();
        let mut tmp = Vec::new();
        permutation(4,3, &mut ans, &mut tmp);
        assert_eq!(ans,
                   [[0, 0, 3], [0, 1, 2], [0, 2, 1], [0, 3, 0], [1, 0, 2], [1, 1, 1], [1, 2, 0], [2, 0, 1], [2, 1, 0], [3, 0, 0]]
        )
    }

    #[test]
    fn test_permutation_2() {
        let mut ans = Vec::new();
        let mut tmp = Vec::new();
        permutation(2,2, &mut ans, &mut tmp);
        assert_eq!(ans,
                   [[0, 1], [1, 0]]
        )
    }

    #[test]
    fn test_permutation_3() {
        let mut ans = Vec::new();
        let mut tmp = Vec::new();
        permutation(3,2, &mut ans, &mut tmp);
        assert_eq!(ans,
                   [[0, 2], [1, 1], [2, 0]]
        )
    }
}