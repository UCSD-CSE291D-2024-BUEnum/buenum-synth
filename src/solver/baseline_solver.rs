use std::collections::HashMap;

use crate::parser::ast::*;
use crate::solver::Solver;

pub struct BaselineSolver;

#[derive(Debug, Clone, PartialEq, Default)]
pub struct BaselineConstraint {
    constraints: Vec<Expr>,
}

#[derive(Debug, Clone, PartialEq, Default)]
pub struct BaselineCounterExample {
    assignment: HashMap<String, Expr>,
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
    ) -> Box<dyn Iterator<Item = Self::Expr> + 'a> {
        // TODO: Implement the enumeration logic based on the grammar and counter-examples
        unimplemented!()
    }

    fn extract_grammar(&self, p: &Self::Prog, func_name: &str) -> Self::Grammar {
        // TODO: Extract the grammar for the specified function from the SyGuS program
        unimplemented!()
    }

    fn extract_constraint(&self, p: &Self::Prog) -> Self::Constraint {
        BaselineConstraint {
            constraints: p.constraints.clone(),
        }
    }

    fn verify(&self, p: &Self::Prog, func_name: &str, expr: &Self::Expr) -> Option<Self::CounterExample> {
        let constraints = self.extract_constraint(p).constraints;
        let declare_vars = p.declare_var.clone();
        let define_funs = p.define_fun.clone();
        // if any neg(cond) is sat, return the assignment as counter-example
        for cond in constraints {
            let cfg = z3::Config::new();
            let ctx = z3::Context::new(&cfg);
            let solver = z3::Solver::new(&ctx);

            let vars: Vec<String> = Vec::new();
            // TODO: add declare_vars into vars

            // TODO: add define_funs into solver

            // TODO: add func_name with expr into solver

            solver.assert(&self.expr_to_z3_bool(&cond, &vars, &ctx).not());

            match solver.check() {
                z3::SatResult::Unsat => {} // check next constraint
                z3::SatResult::Unknown => panic!("Unknown z3 solver result"),
                z3::SatResult::Sat => {
                    let model = solver.get_model().unwrap();
                    let assignment: HashMap<String, Expr> = HashMap::new();
                    // TODO: return value assignments in the model

                    return Some(Self::CounterExample { assignment });
                }
            }
        }
        return None;
    }

    fn expr_to_z3_bool<'ctx>(&self, expr: &Self::Expr, vars: &Vec<String>, ctx: &'ctx z3::Context) -> z3::ast::Bool<'ctx> {
        match expr {
            Expr::ConstBool(b) => z3::ast::Bool::from_bool(&ctx, *b),
            Expr::ConstInt(i) => panic!("Cannot convert integer to boolean"),
            Expr::ConstBitVec(bv) => panic!("Cannot convert bitvector to boolean"),
            Expr::ConstString(s) => panic!("Cannot convert string to boolean"),
            Expr::Var(_, _) => todo!(),
            Expr::FuncApply(_, _) => todo!(),
            Expr::Let(_, _) => todo!(),
            Expr::Not(b) => todo!(),
            Expr::And(b1, b2) => todo!(),
            Expr::Or(_, _) => todo!(),
            Expr::Xor(_, _) => todo!(),
            Expr::Iff(_, _) => todo!(),
            Expr::Equal(_, _) => todo!(),
            Expr::BvAnd(_, _) => todo!(),
            Expr::BvOr(_, _) => todo!(),
            Expr::BvXor(_, _) => todo!(),
            Expr::BvNot(_) => todo!(),
            Expr::BvAdd(_, _) => todo!(),
            Expr::BvMul(_, _) => todo!(),
            Expr::BvSub(_, _) => todo!(),
            Expr::BvUdiv(_, _) => todo!(),
            Expr::BvUrem(_, _) => todo!(),
            Expr::BvShl(_, _) => todo!(),
            Expr::BvLshr(_, _) => todo!(),
            Expr::BvNeg(_) => todo!(),
            Expr::BvUlt(_, _) => todo!(),
            Expr::BvConst(_, _) => todo!(),
            Expr::UnknownExpr => todo!(),
        }
    }
}
