use crate::solver::ast::GTerm;
use std::collections::HashMap;

use z3::ast::Ast;

use crate::parser::ast::*;
use crate::solver::GrammarTrait;
use crate::solver::Solver;
use crate::parser::eval::EvalEnv;

use super::ProgTrait;

pub struct BaselineSolver;

#[derive(Debug, Clone, PartialEq, Default)]
pub struct BaselineConstraint {
    constraints: Vec<Expr>,
}

#[derive(Debug, Clone, PartialEq, Default)]
pub struct BaselineCounterExample {
    assignment: HashMap<Symbol, Expr>,
}

pub struct BaselineEnumerator<'a, S: Solver> {
    solver: &'a S,
    grammar: &'a S::Grammar,
    counterexamples: &'a [S::CounterExample],
    cache: HashMap<(ProdName, usize), Vec<GExpr>>,
    current_size: usize,
    index: usize,
}

/// DFS to search all the possible depth combination of sub non-terminals
pub fn depth_of_subs(d: usize, k: usize, ans: &mut Vec<Vec<usize>>, curr: &mut Vec<usize>){
    if k == 0 {
        if d == 1 {
            ans.push(curr.clone());
        }
        return;
    }
    for i in 0..d {
        curr.push(i);
        depth_of_subs(d - i, k - 1, ans, curr);
        curr.remove(curr.len()-1);
    }
}

fn argument(term_vec: &Vec<Vec<GExpr>>, visited: &Vec<Vec<bool>>, i: usize, curr: &mut Vec<GExpr>, ret: &mut Vec<Vec<GExpr>>){
    if i == term_vec.len() {
        ret.push(curr.clone());
    } else {
        for j in 0..term_vec[i].len() {
            if !visited[i][j] {
                curr.push((&term_vec[i][j]).clone());
                argument(term_vec, visited, i+1, curr, ret);
                curr.remove(curr.len()-1);
            }
        }
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
            index: 0,
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
                        let expr_vec = self.permutation(lhs, expr, &GExpr::UnknownGExpr, d);
                        all_expressions.extend(expr_vec.clone());
                    }
                    _ => eprintln!("Unsupported Rules!")
                }
            }
        }
    }

    fn permutation(&self, prod_name: &ProdName, expr: &GExpr, father: &GExpr, d: usize) -> Vec<GExpr> {
        let mut ret: Vec<GExpr>= Vec::new();
        if d == 0 {
            ret.extend(self.cache.get(&(prod_name.clone(), 0)).unwrap().clone());
            return ret;
        }
        match expr {
            GExpr::Var(symbol, sort) => {
                if self.grammar.lhs_names().contains(&&symbol) {
                    if let Some(productions) = self.grammar.non_terminals().iter().find(|p| p.lhs == *symbol) {
                        let lhs = &productions.lhs;
                        match father {
                            GExpr::Var(name, ..) => {
                                if name == symbol {
                                    return ret;
                                }
                            },
                            _ => {
                                ret.extend(self.cache.get(&(lhs.clone(), d)).unwrap().clone())
                            }
                        }
                    }
                }
            },
            GExpr::FuncApply(funcName, subs)
            | GExpr::GFuncApply(funcName, subs)=> {
                let len = subs.len();
                let mut depth = Vec::new();
                depth_of_subs(d, len, &mut depth, &mut Vec::new());

                let mut final_res = Vec::new();
                for d_i in depth {
                    let mut terms_vec = Vec::new();
                    let mut visited = Vec::new();
                    for k in 0..len {
                        let mut permutation_result = Vec::new();
                        match &subs[k] {
                            GExpr::Var(symbol, _) => {
                                if self.grammar.lhs_names().contains(&symbol) {
                                    permutation_result.extend(self.cache.get(&(symbol.clone(), d_i[k])).unwrap().clone())
                                } else {
                                    permutation_result.push(subs[k].clone())
                                }
                            }
                            _ => eprintln!("There are multiple op in RHS!")
                        }

                        terms_vec.push(permutation_result.clone());
                        let curr_visited = vec![false; (&permutation_result).len()];
                        visited.push(curr_visited);
                    }

                    let mut result = Vec::new();
                    argument(&terms_vec, &visited, 0, &mut Vec::new(), &mut result);

                    final_res.extend(result);
                }


                for res in final_res {
                    match expr {
                        GExpr::FuncApply(_, _) => ret.push(GExpr::FuncApply(funcName.clone(), res.clone())),
                        GExpr::GFuncApply(_, _) => ret.push(GExpr::GFuncApply(funcName.clone(), res.clone())),
                        _ => unreachable!()
                    }
                }
            },
            GExpr::Not(sub)
            | GExpr::BvNot(sub)
            | GExpr::BvNeg(sub) => {
                let terms = self.permutation(prod_name, sub, expr, d-1);
                for t in terms.iter() {
                    //println!("{:?}", t);
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
                let mut depth = Vec::new();
                depth_of_subs(d, 2, &mut depth, &mut Vec::new());
                for d_i in depth {
                    let l_terms = self.permutation(prod_name, l_sub, expr, d_i[0]);
                    let r_terms = self.permutation(prod_name, r_sub, expr, d_i[1]);

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
                }
            },
            GExpr::ConstBool(_)
            | GExpr::ConstInt(_)
            | GExpr::ConstBitVec(_)
            | GExpr::ConstString(_)
            | GExpr::Var(_, _)
            | GExpr::BvConst(_, _) => {},
            GExpr::Let(_, _) => eprintln!("Please implement Let"),
            _ => eprintln!("Unsupported: {:?}", expr)
        }

        // println!("\nHere is the result of {:?}, when d = {}!", expr, d);
        // for e in &ret {
        //     println!("{:?}", e);
        // }
        ret
    }
}

impl<'a, S: Solver> Iterator for BaselineEnumerator<'a, S> {
    type Item = Expr;

    fn next(&mut self) -> Option<Self::Item> {
        loop {
            //println!("{:?}", self.cache);
            for non_terminal in self.grammar.non_terminals().iter().map(|p| &p.lhs) {
                if let Some(expressions) = self.cache.get(&(non_terminal.clone(), self.current_size)) {
                    let program = &expressions[self.index];
                    self.index += 1;
                    if self.index == expressions.len() {
                        self.current_size += 1;
                    }
                    return Option::from(program.to_expr());
                } else {
                    self.grow(non_terminal);
                    if let Some(expressions) = self.cache.get(&(non_terminal.clone(), self.current_size)) {
                        self.index = 0;
                        let program = &expressions[self.index];
                        self.index += 1;
                        return Option::from(program.to_expr());
                    }
                }
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

    fn extract_constraint(&self, p: &Self::Prog) -> Self::Constraint {
        BaselineConstraint {
            constraints: p.constraints.clone(),
        }
    }

    fn oe(&self, env: &EvalEnv, e1: &Self::Expr, e2: &Self::Expr) -> bool {
        e1.eval(env) == e2.eval(env)
    }

    fn verify(&self, p: &Self::Prog, func_name: &str, expr: &Self::Expr) -> Option<Self::CounterExample> {
        let constraints = self.extract_constraint(p).constraints;
        let declare_vars = p.declare_var.clone();
        let mut define_funs = p.define_fun.clone();

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

        // add func_name and expr into define_funs
        define_funs.insert(
            func_name.to_string(),
            self::FuncBody{
                name:func_name.to_string(),
                params: p.get_synth_func(func_name).unwrap().0.params.clone(),
                ret_sort: p.get_synth_func(func_name).unwrap().0.ret_sort.clone(),
                body: expr.clone(),
        });

        // Add define_funs into solver
        let mut funcs: HashMap<String, z3::RecFuncDecl> = HashMap::new();
        for f_name in define_funs.keys() {
            // function declaration
            let f_params = &define_funs[f_name].params;
            let mut domain: Vec<z3::Sort> = Vec::new();
            for param in f_params {
                domain.push(self.sort_to_z3_sort(&param.1, &ctx));
            }
            let domain_references: Vec<&z3::Sort> = domain.iter().collect();
            let f_ret_sort = &define_funs[f_name].ret_sort;
            let decl = z3::RecFuncDecl::new(&ctx, f_name.clone(), domain_references.as_slice(), &self.sort_to_z3_sort(f_ret_sort, &ctx));

            // add function definition
            let mut args: Vec<z3::ast::Dynamic> = Vec::new();
            for param in f_params {
                match param.1 {
                    Sort::Bool => {
                        args.push(z3::ast::Dynamic::from(z3::ast::Bool::new_const(&ctx, param.0.clone())));
                    }
                    Sort::Int => {
                        args.push(z3::ast::Dynamic::from(z3::ast::Int::new_const(&ctx, param.0.clone())));
                    }
                    Sort::BitVec(w) => {
                        args.push(z3::ast::Dynamic::from(z3::ast::BV::new_const(&ctx, param.0.clone(), w as u32)));
                    }
                    Sort::String => {
                        args.push(z3::ast::Dynamic::from(z3::ast::String::new_const(&ctx, param.0.clone())));
                    }
                    _ => panic!("Unsupported sort"),
                }
            }
            
            let args_references: Vec<&dyn z3::ast::Ast> = args.iter().map(|arg| arg as &dyn z3::ast::Ast).collect();
            let mut local_var: HashMap<String, z3::ast::Dynamic> = HashMap::new();
            for param in f_params {
                local_var.insert(param.0.clone(), match param.1 {
                    Sort::Bool => z3::ast::Dynamic::from(z3::ast::Bool::new_const(&ctx, param.0.clone())),
                    Sort::Int => z3::ast::Dynamic::from(z3::ast::Int::new_const(&ctx, param.0.clone())),
                    Sort::BitVec(w) => z3::ast::Dynamic::from(z3::ast::BV::new_const(&ctx, param.0.clone(), w as u32)),
                    Sort::String => z3::ast::Dynamic::from(z3::ast::String::new_const(&ctx, param.0.clone())),
                    _ => panic!("Unsupported sort"),
                });
            }
            let f_body = &define_funs[f_name].body;
            decl.add_def(&args_references, &*self.expr_to_smt(&f_body, &local_var, &funcs, &ctx));

            // bookkeeping functions
            funcs.insert(f_name.clone(), decl);
        }

        // add constraint clauses with disjunction of neg(constraint)
        // if any neg(constraint) is sat, a counter-example is found
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

        // solver.check();
        // println!("{:?}", solver.get_model().unwrap());

        match solver.check() {
            z3::SatResult::Unsat => None, // no counter-example found
            z3::SatResult::Unknown => panic!("Unknown z3 solver result"),
            z3::SatResult::Sat => { // return value assignment where at least one of the constraints is violated
                let model = solver.get_model().unwrap();
                let mut assignment: HashMap<String, Expr> = HashMap::new();
                for (name, sort) in declare_vars {
                    let interp = model.get_const_interp(&vars[&name.clone()]).unwrap();
                    assignment.insert(
                        name.clone(),
                        match sort {
                            Sort::Bool => Expr::ConstBool(interp.as_bool().unwrap().as_bool().unwrap()),
                            Sort::Int => Expr::ConstInt(interp.as_int().unwrap().as_i64().unwrap()),
                            Sort::BitVec(_) => Expr::ConstBitVec(interp.as_bv().unwrap().as_u64().unwrap()),
                            Sort::Compound(_, _) => unimplemented!("Not supporting compound solving"),
                            Sort::String => Expr::ConstString(interp.as_string().unwrap().as_string().unwrap()),
                            Sort::None => panic!("Unsupported sort"),
                        },
                    );
                }

                return Some(Self::CounterExample { assignment });
            }
        }
    }

    fn expr_to_smt<'ctx>(
        &self,
        expr: &Self::Expr,
        vars: &'ctx HashMap<String, z3::ast::Dynamic>,
        funcs: &'ctx HashMap<String, z3::RecFuncDecl>,
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

    fn sort_to_z3_sort<'ctx>(&self, sort: &Sort, ctx: &'ctx z3::Context) -> z3::Sort<'ctx> {
        match sort {
            Sort::Bool => z3::Sort::bool(ctx),
            Sort::Int => z3::Sort::int(ctx),
            Sort::BitVec(w) => z3::Sort::bitvector(ctx, *w as u32),
            Sort::String => z3::Sort::string(ctx),
            _ => panic!("Unsupported sort"),
        }
    }
}


mod tests {
    use std::fs;

    use crate::{parser::parse, solver::Solver};

    use super::{depth_of_subs, Expr, Sort};

    #[test]
    fn test_permutation_1() {
        let mut ans = Vec::new();
        let mut tmp = Vec::new();
        depth_of_subs(4,3, &mut ans, &mut tmp);
        assert_eq!(ans,
                   [[0, 0, 3], [0, 1, 2], [0, 2, 1], [0, 3, 0], [1, 0, 2], [1, 1, 1], [1, 2, 0], [2, 0, 1], [2, 1, 0], [3, 0, 0]]
        )
    }

    #[test]
    fn test_permutation_2() {
        let mut ans = Vec::new();
        let mut tmp = Vec::new();
        depth_of_subs(2,2, &mut ans, &mut tmp);
        assert_eq!(ans,
                   [[0, 1], [1, 0]]
        )
    }

    #[test]
    fn test_permutation_3() {
        let mut ans = Vec::new();
        let mut tmp = Vec::new();
        depth_of_subs(3,2, &mut ans, &mut tmp);
        assert_eq!(ans,
                   [[0, 2], [1, 1], [2, 0]]
        )
    }

    #[test]
    fn test_oe() {
        unimplemented!()
    }

    #[test]
    fn test_verify_1() {
        let filename = format!("{}/{}", env!("CARGO_MANIFEST_DIR"), format!("test_bool_1.sl"));
        let input = fs::read_to_string(&filename).unwrap();
        let prog = match parse(&input){
            Ok(res) => res,
            Err(e) => {
                panic!("Error parsing file: {}\nError: {:#?}", filename, e);
            }
        };
        println!("{:#?}", prog);
        let solver = super::BaselineSolver;
        let func_name = "AIG";

        // first attempt: a
        let expr = Expr::Var("a".to_string(), Sort::Bool);
        println!("Expr: {:?}", expr);
        let counter_example = solver.verify(&prog, func_name, &expr);
        match counter_example {
            Some(cex) => {
                println!("Counter Example: {:?}", cex);
            }
            None => {
                println!("No Counter Example Found!");
            }
        }

        // second attempt: not a
        let expr = Expr::Not(Box::from(Expr::Var("a".to_string(), Sort::Bool)));
        println!("Expr: {:?}", expr);
        let counter_example = solver.verify(&prog, func_name, &expr);
        match counter_example {
            Some(cex) => {
                println!("Counter Example: {:?}", cex);
            }
            None => {
                println!("No Counter Example Found!");
            }
        }

        // thrid attempt: and a b
        let expr = Expr::And(Box::from(Expr::Var("a".to_string(), Sort::Bool)), Box::from(Expr::Var("b".to_string(), Sort::Bool)));
        println!("Expr: {:?}", expr);
        let counter_example = solver.verify(&prog, func_name, &expr);
        match counter_example {
            Some(cex) => {
                println!("Counter Example: {:?}", cex);
            }
            None => {
                println!("No Counter Example Found!");
            }
        }

        // fourth attempt: and (not a) b
        let expr = Expr::And(
                            Box::from(Expr::Not(
                                Box::from(Expr::Var("a".to_string(), Sort::Bool)))), 
                            Box::from(Expr::Var("b".to_string(), Sort::Bool)));
        let counter_example = solver.verify(&prog, func_name, &expr);
        println!("Expr: {:?}", expr);
        match counter_example {
            Some(cex) => {
                println!("Counter Example: {:?}", cex);
            }
            None => {
                println!("No Counter Example Found!");
            }
        }

        // fifth attempt (correct): not (and (not a) (not b))
        let expr = Expr::Not(
                            Box::from(Expr::And(
                                Box::from(Expr::Not(
                                    Box::from(Expr::Var("a".to_string(), Sort::Bool)))), 
                                Box::from(Expr::Not(
                                    Box::from(Expr::Var("b".to_string(), Sort::Bool)))))));
        let counter_example = solver.verify(&prog, func_name, &expr);
        println!("Expr: {:?}", expr);
        match counter_example {
            Some(cex) => {
                println!("Counter Example: {:?}", cex);
            }
            None => {
                println!("No Counter Example Found!");
            }
        }

    }

    #[test]
    fn test_verify_2() {
        let filename = format!("{}/{}", env!("CARGO_MANIFEST_DIR"), format!("test_bool_2.sl"));
        let input = fs::read_to_string(&filename).unwrap();
        let prog = match parse(&input){
            Ok(res) => res,
            Err(e) => {
                panic!("Error parsing file: {}\nError: {:#?}", filename, e);
            }
        };
        println!("{:#?}", prog);
        let solver = super::BaselineSolver;
        let func_name = "AIG";

        // first attempt: a
        let expr = Expr::Var("a".to_string(), Sort::Bool);
        println!("Expr: {:?}", expr);
        let counter_example = solver.verify(&prog, func_name, &expr);
        match counter_example {
            Some(cex) => {
                println!("Counter Example: {:?}", cex);
            }
            None => {
                println!("No Counter Example Found!");
            }
        }

        // second attempt: not a
        let expr = Expr::Not(Box::from(Expr::Var("a".to_string(), Sort::Bool)));
        println!("Expr: {:?}", expr);
        let counter_example = solver.verify(&prog, func_name, &expr);
        match counter_example {
            Some(cex) => {
                println!("Counter Example: {:?}", cex);
            }
            None => {
                println!("No Counter Example Found!");
            }
        }

        // thrid attempt: and a b
        let expr = Expr::And(Box::from(Expr::Var("a".to_string(), Sort::Bool)), Box::from(Expr::Var("b".to_string(), Sort::Bool)));
        println!("Expr: {:?}", expr);
        let counter_example = solver.verify(&prog, func_name, &expr);
        match counter_example {
            Some(cex) => {
                println!("Counter Example: {:?}", cex);
            }
            None => {
                println!("No Counter Example Found!");
            }
        }

        // fourth attempt: and (and a b) c
        let expr = Expr::And(
                            Box::from(Expr::And(
                                Box::from(Expr::Var("a".to_string(), Sort::Bool)),
                                Box::from(Expr::Var("b".to_string(), Sort::Bool))),
                            ), 
                            Box::from(Expr::Var("c".to_string(), Sort::Bool)));
        let counter_example = solver.verify(&prog, func_name, &expr);
        println!("Expr: {:?}", expr);
        match counter_example {
            Some(cex) => {
                println!("Counter Example: {:?}", cex);
            }
            None => {
                println!("No Counter Example Found!");
            }
        }

        // fifth attempt (correct): not (and (and (not a) (not b)) (not c))
        let expr = Expr::Not(
                            Box::from(Expr::And(
                                Box::from(Expr::And(
                                    Box::from(Expr::Not(
                                        Box::from(Expr::Var("a".to_string(), Sort::Bool)))), 
                                    Box::from(Expr::Not(
                                        Box::from(Expr::Var("b".to_string(), Sort::Bool)))))),
                                Box::from(Expr::Not(
                                    Box::from(Expr::Var("c".to_string(), Sort::Bool)))))));
        let counter_example = solver.verify(&prog, func_name, &expr);
        println!("Expr: {:?}", expr);
        match counter_example {
            Some(cex) => {
                println!("Counter Example: {:?}", cex);
            }
            None => {
                println!("No Counter Example Found!");
            }
        }

    }
}