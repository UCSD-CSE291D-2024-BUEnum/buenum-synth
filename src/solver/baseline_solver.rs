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

        let cfg = z3::Config::new();
        let ctx = z3::Context::new(&cfg);
        let solver = z3::Solver::new(&ctx);

        // declare variables which are used in the constraint
        let mut vars: HashMap<String, z3::ast::Dynamic> = HashMap::new();
        for v in declare_vars.keys() {
            match declare_vars[v] {
                Sort::Bool => {
                    vars.insert(
                        v.clone(),
                        z3::ast::Dynamic::from(z3::ast::Bool::new_const(&ctx, v.clone())),
                    );
                }
                Sort::Int => {
                    vars.insert(
                        v.clone(),
                        z3::ast::Dynamic::from(z3::ast::Int::new_const(&ctx, v.clone())),
                    );
                }
                Sort::BitVec(w) => {
                    vars.insert(
                        v.clone(),
                        z3::ast::Dynamic::from(z3::ast::BV::new_const(&ctx, v.clone(), w as u32)),
                    );
                }
                Sort::String => {
                    vars.insert(
                        v.clone(),
                        z3::ast::Dynamic::from(z3::ast::String::new_const(&ctx, v.clone())),
                    );
                }
                _ => panic!("Unsupported sort"),
            }
        }

        // TODO: add define_funs into solver

        // TODO: add func_name with expr into solver

        let mut clauses: Vec<z3::ast::Bool> = Vec::new();
        for c in constraints {
            clauses.push(self.expr_to_smt(&c, &vars, &ctx).as_bool().unwrap().not());
        }
        let clauses_references: Vec<&z3::ast::Bool<'_>> = clauses.iter().collect();
        solver.assert(&z3::ast::Bool::or(&ctx, clauses_references.as_slice()));

        match solver.check() {
            z3::SatResult::Unsat => None,
            z3::SatResult::Unknown => panic!("Unknown z3 solver result"),
            z3::SatResult::Sat => {
                let model = solver.get_model().unwrap();
                let assignment: HashMap<String, Expr> = HashMap::new();
                // TODO: return value assignments in the model

                return Some(Self::CounterExample { assignment });
            }
        }
    }

    fn expr_to_smt<'ctx>(
        &self,
        expr: &Self::Expr,
        vars: &'ctx HashMap<String, z3::ast::Dynamic>,
        ctx: &'ctx z3::Context,
    ) -> Box<z3::ast::Dynamic<'ctx>> {
        macro_rules! bv_unary_operation {
            ($expr:ident, $e:expr, $vars:expr, $ctx:expr, $method:ident) => {
                Box::from(z3::ast::Dynamic::from(
                    $expr.expr_to_smt($e, $vars, $ctx).as_bv().unwrap().$method(),
                ))
            };
        }

        macro_rules! bv_binary_operation {
            ($expr:ident, $e1:expr, $e2:expr, $vars:expr, $ctx:expr, $method:ident) => {
                Box::from(z3::ast::Dynamic::from(
                    $expr
                        .expr_to_smt($e1, $vars, $ctx)
                        .as_bv()
                        .unwrap()
                        .$method(&$expr.expr_to_smt($e2, $vars, $ctx).as_bv().unwrap()),
                ))
            };
        }

        match expr {
            Expr::ConstBool(b) => Box::from(z3::ast::Dynamic::from(z3::ast::Bool::from_bool(ctx, *b))),
            Expr::ConstInt(i) => Box::from(z3::ast::Dynamic::from(z3::ast::Int::from_i64(ctx, *i))),
            Expr::ConstBitVec(u) => Box::from(z3::ast::Dynamic::from(z3::ast::BV::from_u64(ctx, *u, 64))),
            Expr::ConstString(s) => todo!(),
            Expr::Var(symbol, _) => Box::from(vars[symbol].clone()),
            Expr::FuncApply(_, _) => todo!(),
            Expr::Let(_, _) => unimplemented!("Not supporting lamdba calculus in constraints."),
            Expr::Not(e) => Box::from(z3::ast::Dynamic::from(
                self.expr_to_smt(e, vars, ctx).as_bool().unwrap().not(),
            )),
            Expr::And(e1, e2) => Box::from(z3::ast::Dynamic::from(z3::ast::Bool::and(
                ctx,
                &[
                    &self.expr_to_smt(e1, vars, ctx).as_bool().unwrap(),
                    &self.expr_to_smt(e2, vars, ctx).as_bool().unwrap(),
                ],
            ))),
            Expr::Or(e1, e2) => Box::from(z3::ast::Dynamic::from(z3::ast::Bool::or(
                ctx,
                &[
                    &self.expr_to_smt(e1, vars, ctx).as_bool().unwrap(),
                    &self.expr_to_smt(e2, vars, ctx).as_bool().unwrap(),
                ],
            ))),
            Expr::Xor(e1, e2) => Box::from(z3::ast::Dynamic::from(
                self.expr_to_smt(e1, vars, ctx)
                    .as_bool()
                    .unwrap()
                    .xor(&self.expr_to_smt(e2, vars, ctx).as_bool().unwrap()),
            )),
            Expr::Iff(e1, e2) => Box::from(z3::ast::Dynamic::from(
                self.expr_to_smt(e1, vars, ctx)
                    .as_bool()
                    .unwrap()
                    .iff(&self.expr_to_smt(e2, vars, ctx).as_bool().unwrap()),
            )),
            Expr::Equal(_, _) => todo!(),
            Expr::BvAnd(e1, e2) => bv_binary_operation!(self, e1, e2, vars, ctx, bvand),
            Expr::BvOr(e1, e2) => bv_binary_operation!(self, e1, e2, vars, ctx, bvor),
            Expr::BvXor(e1, e2) => bv_binary_operation!(self, e1, e2, vars, ctx, bvxor),
            Expr::BvNot(e) => bv_unary_operation!(self, e, vars, ctx, bvnot),
            Expr::BvAdd(e1, e2) => bv_binary_operation!(self, e1, e2, vars, ctx, bvadd),
            Expr::BvMul(e1, e2) => bv_binary_operation!(self, e1, e2, vars, ctx, bvmul),
            Expr::BvSub(e1, e2) => bv_binary_operation!(self, e1, e2, vars, ctx, bvsub),
            Expr::BvUdiv(e1, e2) => bv_binary_operation!(self, e1, e2, vars, ctx, bvudiv),
            Expr::BvUrem(e1, e2) => bv_binary_operation!(self, e1, e2, vars, ctx, bvurem),
            Expr::BvShl(e1, e2) => bv_binary_operation!(self, e1, e2, vars, ctx, bvshl),
            Expr::BvLshr(e1, e2) => bv_binary_operation!(self, e1, e2, vars, ctx, bvlshr),
            Expr::BvNeg(e) => bv_unary_operation!(self, e, vars, ctx, bvneg),
            Expr::BvUlt(e1, e2) => bv_binary_operation!(self, e1, e2, vars, ctx, bvult),
            Expr::BvConst(v, width) => Box::from(z3::ast::Dynamic::from(z3::ast::BV::from_u64(
                ctx,
                *v as u64,
                *width as u32,
            ))),
            Expr::UnknownExpr => panic!("Unknown expression"),
        }
    }
}
