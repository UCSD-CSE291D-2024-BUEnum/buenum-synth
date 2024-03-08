use crate::parser::ast::*;
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
        // TODO: Implement the enumeration logic based on the grammar and counter-examples
        unimplemented!()
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
