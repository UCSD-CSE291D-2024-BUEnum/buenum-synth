use std::collections::HashMap;

use z3::ast::Ast;

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

        // Add define_funs into solver
        let mut funcs: HashMap<String, z3::FuncDecl> = HashMap::new();
        for f_name in define_funs.keys() {
            let f_params = &define_funs[f_name].params;
            let mut domain: Vec<z3::Sort> = Vec::new();
            for param in f_params {
                domain.push(match param.1 {
                    Sort::Bool => z3::Sort::bool(&ctx),
                    Sort::Int => z3::Sort::int(&ctx),
                    Sort::BitVec(w) => z3::Sort::bitvector(&ctx, w as u32),
                    Sort::String => z3::Sort::string(&ctx),
                    _ => panic!("Unsupported sort"),
                })
            }
            let domain_references: Vec<&z3::Sort> = domain.iter().collect();

            let f_ret_sort = &define_funs[f_name].ret_sort;
            let range = match f_ret_sort {
                Sort::Bool => z3::Sort::bool(&ctx),
                Sort::Int => z3::Sort::int(&ctx),
                Sort::BitVec(w) => z3::Sort::bitvector(&ctx, *w as u32),
                Sort::String => z3::Sort::string(&ctx),
                _ => panic!("Unsupported sort"),
            };

            let f_body = &define_funs[f_name].body;
            // TODO: add function body
            let decl = z3::FuncDecl::new(&ctx, f_name.clone(), domain_references.as_slice(), &range);

            funcs.insert(f_name.clone(), decl);
        }

        // TODO: add func_name with expr into solver similar to define_funcs

        let mut clauses: Vec<z3::ast::Bool> = Vec::new();
        for c in constraints {
            match c {
                // assume there is at least one constraint and it's equation
                Expr::Equal(e1, e2) => match declare_vars[declare_vars.keys().next().unwrap()] {
                    Sort::Bool => {
                        clauses.push(
                            self.expr_to_smt(&e1, &vars, &funcs, &ctx)
                                .as_bool()
                                .unwrap()
                                ._eq(&self.expr_to_smt(&e2, &vars, &funcs, &ctx).as_bool().unwrap())
                                .not(),
                        );
                    }
                    Sort::Int => {
                        clauses.push(
                            self.expr_to_smt(&e1, &vars, &funcs, &ctx)
                                .as_int()
                                .unwrap()
                                ._eq(&self.expr_to_smt(&e2, &vars, &funcs, &ctx).as_int().unwrap())
                                .not(),
                        );
                    }
                    Sort::BitVec(_) => {
                        clauses.push(
                            self.expr_to_smt(&e1, &vars, &funcs, &ctx)
                                .as_bv()
                                .unwrap()
                                ._eq(&self.expr_to_smt(&e2, &vars, &funcs, &ctx).as_bv().unwrap())
                                .not(),
                        );
                    }
                    Sort::String => {
                        clauses.push(
                            self.expr_to_smt(&e1, &vars, &funcs, &ctx)
                                .as_string()
                                .unwrap()
                                ._eq(&self.expr_to_smt(&e2, &vars, &funcs, &ctx).as_string().unwrap())
                                .not(),
                        );
                    }
                    _ => panic!("Unsupported sort"),
                },
                _ => panic!("Unsupported constraint format"),
            }
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
        funcs: &'ctx HashMap<String, z3::FuncDecl>,
        ctx: &'ctx z3::Context,
    ) -> Box<z3::ast::Dynamic<'ctx>> {
        macro_rules! bv_unary_operation {
            ($expr:ident, $e:expr, $vars:expr, $funcs:expr, $ctx:expr, $method:ident) => {
                Box::from(z3::ast::Dynamic::from(
                    $expr
                        .expr_to_smt($e, $vars, $funcs, $ctx)
                        .as_bv()
                        .unwrap()
                        .$method(),
                ))
            };
        }

        macro_rules! bv_binary_operation {
            ($expr:ident, $e1:expr, $e2:expr, $vars:expr, $funcs:expr, $ctx:expr, $method:ident) => {
                Box::from(z3::ast::Dynamic::from(
                    $expr
                        .expr_to_smt($e1, $vars, $funcs, $ctx)
                        .as_bv()
                        .unwrap()
                        .$method(&$expr.expr_to_smt($e2, $vars, $funcs, $ctx).as_bv().unwrap()),
                ))
            };
        }

        match expr {
            Expr::ConstBool(b) => Box::from(z3::ast::Dynamic::from(z3::ast::Bool::from_bool(ctx, *b))),
            Expr::ConstInt(i) => Box::from(z3::ast::Dynamic::from(z3::ast::Int::from_i64(ctx, *i))),
            Expr::ConstBitVec(u) => Box::from(z3::ast::Dynamic::from(z3::ast::BV::from_u64(ctx, *u, 64))),
            Expr::ConstString(s) => Box::from(z3::ast::Dynamic::from(z3::ast::String::from_str(ctx, s).unwrap())),
            Expr::Var(symbol, _) => Box::from(vars[symbol].clone()),
            Expr::FuncApply(name, exprs) => {
                let mut args: Vec<z3::ast::Dynamic> = Vec::new();
                for e in exprs {
                    args.push(*self.expr_to_smt(e, vars, funcs, ctx));
                }
                let args_references: Vec<&dyn z3::ast::Ast> = args.iter().map(|arg| arg as &dyn z3::ast::Ast).collect();
                Box::from(z3::ast::Dynamic::from(funcs[name].apply(args_references.as_slice())))
            }
            Expr::Let(_, _) => unimplemented!("Not supporting lamdba calculus in constraints."),
            Expr::Not(e) => Box::from(z3::ast::Dynamic::from(
                self.expr_to_smt(e, vars, funcs, ctx).as_bool().unwrap().not(),
            )),
            Expr::And(e1, e2) => Box::from(z3::ast::Dynamic::from(z3::ast::Bool::and(
                ctx,
                &[
                    &self.expr_to_smt(e1, vars, funcs, ctx).as_bool().unwrap(),
                    &self.expr_to_smt(e2, vars, funcs, ctx).as_bool().unwrap(),
                ],
            ))),
            Expr::Or(e1, e2) => Box::from(z3::ast::Dynamic::from(z3::ast::Bool::or(
                ctx,
                &[
                    &self.expr_to_smt(e1, vars, funcs, ctx).as_bool().unwrap(),
                    &self.expr_to_smt(e2, vars, funcs, ctx).as_bool().unwrap(),
                ],
            ))),
            Expr::Xor(e1, e2) => Box::from(z3::ast::Dynamic::from(
                self.expr_to_smt(e1, vars, funcs, ctx)
                    .as_bool()
                    .unwrap()
                    .xor(&self.expr_to_smt(e2, vars, funcs, ctx).as_bool().unwrap()),
            )),
            Expr::Iff(e1, e2) => Box::from(z3::ast::Dynamic::from(
                self.expr_to_smt(e1, vars, funcs, ctx)
                    .as_bool()
                    .unwrap()
                    .iff(&self.expr_to_smt(e2, vars, funcs, ctx).as_bool().unwrap()),
            )),
            Expr::Equal(e1, e2) => unreachable!(), // handled before reaching here
            Expr::BvAnd(e1, e2) => bv_binary_operation!(self, e1, e2, vars, funcs, ctx, bvand),
            Expr::BvOr(e1, e2) => bv_binary_operation!(self, e1, e2, vars, funcs, ctx, bvor),
            Expr::BvXor(e1, e2) => bv_binary_operation!(self, e1, e2, vars, funcs, ctx, bvxor),
            Expr::BvNot(e) => bv_unary_operation!(self, e, vars, funcs, ctx, bvnot),
            Expr::BvAdd(e1, e2) => bv_binary_operation!(self, e1, e2, vars, funcs, ctx, bvadd),
            Expr::BvMul(e1, e2) => bv_binary_operation!(self, e1, e2, vars, funcs, ctx, bvmul),
            Expr::BvSub(e1, e2) => bv_binary_operation!(self, e1, e2, vars, funcs, ctx, bvsub),
            Expr::BvUdiv(e1, e2) => bv_binary_operation!(self, e1, e2, vars, funcs, ctx, bvudiv),
            Expr::BvUrem(e1, e2) => bv_binary_operation!(self, e1, e2, vars, funcs, ctx, bvurem),
            Expr::BvShl(e1, e2) => bv_binary_operation!(self, e1, e2, vars, funcs, ctx, bvshl),
            Expr::BvLshr(e1, e2) => bv_binary_operation!(self, e1, e2, vars, funcs, ctx, bvlshr),
            Expr::BvNeg(e) => bv_unary_operation!(self, e, vars, funcs, ctx, bvneg),
            Expr::BvUlt(e1, e2) => bv_binary_operation!(self, e1, e2, vars, funcs, ctx, bvult),
            Expr::BvConst(v, width) => Box::from(z3::ast::Dynamic::from(z3::ast::BV::from_u64(
                ctx,
                *v as u64,
                *width as u32,
            ))),
            Expr::UnknownExpr => panic!("Unknown expression"),
        }
    }
}
