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

struct Counter {
    value: usize,
}

impl Counter {
    fn new(n: usize) -> Counter {
        Counter { value: n }
    }

    fn decrement(&mut self, n: usize) {
        self.value -= n;
    }

    fn value(&self) -> usize {
        self.value
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
                    _ => eprintln!("Unsupported Production Type!"),
                }
            } else {
                match production {
                    GTerm::BfTerm(expr) => {
                        let expr_vec = self.permutation(expr, &mut Counter::new(d));
                        all_expressions.extend(expr_vec.clone());
                    }
                    _ => eprintln!("Unsupported Rules!")
                }
            }
        }
    }

    fn permutation(&self, expr: &GExpr, counter: &mut Counter) -> Vec<GExpr> {
        let mut ret= Vec::new();
        match expr {
            GExpr::Var(symbol, sort) => {
                if self.grammar.lhs_names().contains(&&symbol) {
                    for index in 0..counter.value {
                        counter.decrement(index);
                        let terms = self.cache.get(&(symbol.clone(), index)).unwrap();
                        ret.extend(terms.clone());
                        return ret;
                    }
                }
            },
            GExpr::Not(sub)
            | GExpr::BvNot(sub)
            | GExpr::BvNeg(sub) => {
                let terms = self.permutation(sub, counter);
                for t in terms.iter() {
                    match expr {
                        GExpr::Not(sub) => ret.push(GExpr::Not(Box::new(t.clone()))),
                        GExpr::BvNot(sub) => ret.push(GExpr::BvNot(Box::new(t.clone()))),
                        GExpr::BvNeg(sub) => ret.push(GExpr::BvNeg(Box::new(t.clone()))),
                        _ => unreachable!()
                    }
                }
            },
            GExpr::And(l_sub, r_sub)
            | GExpr::Or(l_sub, r_sub)
            | GExpr::Xor(l_sub, r_sub)
            | GExpr::Iff(l_sub, r_sub)
            | GExpr::Equal(l_sub, r_sub)
            | GExpr::BvAnd(l_sub, r_sub)
            | GExpr::BvOr(l_sub, r_sub)
            | GExpr::BvXor(l_sub, r_sub)
            | GExpr::BvAdd(l_sub, r_sub)
            | GExpr::BvMul(l_sub, r_sub)
            | GExpr::BvSub(l_sub, r_sub)
            | GExpr::BvUdiv(l_sub, r_sub)
            | GExpr::BvUrem(l_sub, r_sub)
            | GExpr::BvShl(l_sub, r_sub)
            | GExpr::BvLshr(l_sub, r_sub)
            | GExpr::BvUlt(l_sub, r_sub)
            => {
                let l_terms = self.permutation(l_sub, counter);
                let r_terms = self.permutation(r_sub, counter);
                for lt in l_terms.iter() {
                    for rt in r_terms.iter() {
                        match expr {
                            GExpr::And(_, _) => ret.push(GExpr::And(Box::new(lt.clone()), Box::new(rt.clone()))),
                            GExpr::Or(_, _) => ret.push(GExpr::Or(Box::new(lt.clone()), Box::new(rt.clone()))),
                            GExpr::Xor(_, _) => ret.push(GExpr::Xor(Box::new(lt.clone()), Box::new(rt.clone()))),
                            GExpr::Iff(_, _) => ret.push(GExpr::Iff(Box::new(lt.clone()), Box::new(rt.clone()))),
                            GExpr::Equal(_, _) => ret.push(GExpr::Equal(Box::new(lt.clone()), Box::new(rt.clone()))),
                            GExpr::BvAnd(_, _) => ret.push(GExpr::BvAnd(Box::new(lt.clone()), Box::new(rt.clone()))),
                            GExpr::BvOr(_, _) => ret.push(GExpr::BvOr(Box::new(lt.clone()), Box::new(rt.clone()))),
                            GExpr::BvXor(_, _) => ret.push(GExpr::BvXor(Box::new(lt.clone()), Box::new(rt.clone()))),
                            GExpr::BvAdd(_, _) => ret.push(GExpr::BvAdd(Box::new(lt.clone()), Box::new(rt.clone()))),
                            GExpr::BvMul(_, _) => ret.push(GExpr::BvMul(Box::new(lt.clone()), Box::new(rt.clone()))),
                            GExpr::BvSub(_, _) => ret.push(GExpr::BvSub(Box::new(lt.clone()), Box::new(rt.clone()))),
                            GExpr::BvUdiv(_, _) => ret.push(GExpr::BvUdiv(Box::new(lt.clone()), Box::new(rt.clone()))),
                            GExpr::BvUrem(_, _) => ret.push(GExpr::BvUrem(Box::new(lt.clone()), Box::new(rt.clone()))),
                            GExpr::BvShl(_, _) => ret.push(GExpr::BvShl(Box::new(lt.clone()), Box::new(rt.clone()))),
                            GExpr::BvLshr(_, _) => ret.push(GExpr::BvLshr(Box::new(lt.clone()), Box::new(rt.clone()))),
                            GExpr::BvUlt(_, _) => ret.push(GExpr::BvUlt(Box::new(lt.clone()), Box::new(rt.clone()))),
                            _ => unreachable!()
                        }
                    }
                }
            },
            GExpr::ConstBool(_)
            | GExpr::ConstInt(_)
            | GExpr::ConstBitVec(_)
            | GExpr::ConstString(_)
            | GExpr::Var(_, _)
            | GExpr::BvConst(_, _) => ret.push(expr.clone()),
            GExpr::Let(_, _) => eprintln!("Please implement Let"),
            GExpr::FuncApply(_, _) => eprintln!("Please implement FuncApply"),
            GExpr::GFuncApply(_, _) => eprintln!("Please implement GFuncApply"),
            _ => eprintln!("Unsupported: {:?}", expr)
        }
        ret
    }
}

impl<'a, S: Solver> Iterator for BaselineEnumerator<'a, S> {
    type Item = Expr;

    fn next(&mut self) -> Option<Self::Item> {
        loop {
            println!("{:?}", self.cache);
            for non_terminal in self.grammar.non_terminals().iter().map(|p| &p.lhs) {
                if let Some(expressions) = self.cache.get(&(non_terminal.clone(), self.current_size)) {
                    if let Some(expr) = expressions.first() {
                        //TODO: do something
                        println!("{:?}", expr.to_expr());
                        continue;
                        return Some(expr.to_expr());
                    }
                } else {
                    self.grow(non_terminal);
                    if let Some(expressions) = self.cache.get(&(non_terminal.clone(), self.current_size)) {
                        if let Some(expr) = expressions.first() {
                            println!("{:?}", expr.to_expr());
                            //TODO: do something
                            continue;
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
