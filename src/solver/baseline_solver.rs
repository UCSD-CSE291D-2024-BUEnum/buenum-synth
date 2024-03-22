use crate::solver::ast::GTerm;
use std::collections::HashMap;

use z3::ast::Ast;

use crate::parser::ast::*;
use crate::parser::eval::EvalEnv;
use crate::parser::eval::Value;
use crate::solver::GrammarTrait;
use crate::solver::Solver;

use super::ProgTrait;

pub struct BaselineSolver;

#[derive(Debug, Clone, PartialEq, Default)]
pub struct BaselineConstraint {
    constraints: Vec<Expr>,
}

#[derive(Debug, Clone, PartialEq, Default)]
pub struct BaselineCounterExample {
    assignment: HashMap<Symbol, Value>,
    output: Value,
}

pub struct BaselineEnumerator<'a, S: Solver> {
    solver: &'a S,
    grammar: &'a S::Grammar,
    counterexamples: &'a Vec<BaselineCounterExample>,
    cache: HashMap<(ProdName, usize), Vec<GExpr>>,
    current_size: usize,    // height of current candidates (starting from 0)
    index: usize,           // index of current candidate accross all non-terminals
    candidates: Vec<GExpr>, // list of all candidates (accross non-terminals) with the current size
    oe_cache: HashMap<String, Vec<BaselineCounterExample>>, // To be hashable, String is the format!() of candicate expression
}

/// DFS to search all the possible depth combination of sub non-terminals
pub fn depth_of_subs(d: usize, k: usize, ans: &mut Vec<Vec<usize>>, curr: &mut Vec<usize>) {
    if k == 0 {
        if d == 1 {
            ans.push(curr.clone());
        }
        return;
    }
    for i in 0..d {
        curr.push(i);
        depth_of_subs(d - i, k - 1, ans, curr);
        curr.remove(curr.len() - 1);
    }
}

fn argument(
    term_vec: &Vec<Vec<GExpr>>,
    visited: &Vec<Vec<bool>>,
    i: usize,
    curr: &mut Vec<GExpr>,
    ret: &mut Vec<Vec<GExpr>>,
) {
    if i == term_vec.len() {
        ret.push(curr.clone());
    } else {
        for j in 0..term_vec[i].len() {
            if !visited[i][j] {
                curr.push((&term_vec[i][j]).clone());
                argument(term_vec, visited, i + 1, curr, ret);
                curr.remove(curr.len() - 1);
            }
        }
    }
}

fn match_all_io(a: &Vec<BaselineCounterExample>, b: &Vec<BaselineCounterExample>) -> bool {
    assert!(a.len() == b.len());
    let mut count = 0;
    for i in 0..a.len() {
        for j in 0..b.len() {
            if a[i].assignment == b[j].assignment {
                if a[i].output != b[j].output {
                    return false;
                } else {
                    count += 1;
                }
            }
        }
    }
    count == a.len()
}

fn match_any_io(a: &Vec<BaselineCounterExample>, b: &Vec<BaselineCounterExample>) -> bool {
    assert!(a.len() == b.len());
    for i in 0..a.len() {
        for j in 0..b.len() {
            if a[i].assignment == b[j].assignment {
                if a[i].output == b[j].output {
                    return true;
                }
            }
        }
    }
    false
}

impl<'a, S: Solver> BaselineEnumerator<'a, S> {
    pub fn new(solver: &'a S, grammar: &'a S::Grammar, counterexamples: &'a Vec<BaselineCounterExample>) -> Self {
        BaselineEnumerator {
            solver,
            grammar,
            counterexamples,
            cache: HashMap::new(),
            current_size: 0,
            index: 0,
            candidates: Vec::new(),
            oe_cache: HashMap::new(),
        }
    }

    fn grow(&mut self, non_terminal: &ProdName) {
        let size = self.current_size;
        if let Some(productions) = self.grammar.non_terminals().iter().find(|p| p.lhs == *non_terminal) {
            let mut generated_terms: Vec<GExpr> = Vec::new();
            self.terms(productions, size, &mut generated_terms);
            // Prune with Observational Equivalence
            // Compute output for each generated term on all pts
            // Skip those which match the output of one candidate in oe_cache on all pts
            let mut rem_terms: Vec<GExpr> = Vec::new();
            for gexpr in generated_terms {
                // compute this input-output
                let mut input_output: Vec<BaselineCounterExample> = Vec::new();
                for cex in self.counterexamples {
                    let vars: Vec<(String, Value)> =
                        cex.assignment.iter().map(|(k, v)| (k.clone(), v.clone())).collect();
                    let env = EvalEnv {
                        vars: vars,
                        funcs: Vec::new(),
                    };
                    let val = gexpr.to_expr().eval(&env);
                    input_output.push(BaselineCounterExample {
                        assignment: cex.assignment.clone(),
                        output: val,
                    });
                }
                // iterate over all candicates to find if there is an oe match
                let mut match_flag = false;
                for (_, cex_vec) in self.oe_cache.iter() {
                    if match_all_io(&input_output, &cex_vec) {
                        match_flag = true;
                        break;
                    }
                }
                if match_flag {
                    continue;
                }
                self.oe_cache.insert(format!("{:?}", gexpr.clone()), input_output);
                rem_terms.push(gexpr);
            }
            if rem_terms.len() != 0 {
                self.cache.entry((non_terminal.clone(), size)).or_insert(rem_terms);
            }
        }
    }

    fn terms(&self, productions: &Production, d: usize, all_expressions: &mut Vec<GExpr>) {
        let lhs = &productions.lhs;
        let rhs = &productions.rhs;
        for production in rhs {
            if d == 0 {
                match production {
                    GTerm::BfTerm(expr) => match expr {
                        GExpr::ConstBool(_)
                        | GExpr::ConstInt(_)
                        | GExpr::ConstBitVec(_)
                        | GExpr::ConstString(_)
                        | GExpr::Var(_, _)
                        | GExpr::BvConst(_, _) => all_expressions.push(expr.clone()),
                        _ => {}
                    },
                    _ => eprintln!("Unsupported Production Type!"),
                }
            } else {
                match production {
                    GTerm::BfTerm(expr) => {
                        let expr_vec = self.permutation(lhs, expr, &GExpr::UnknownGExpr, d);
                        all_expressions.extend(expr_vec.clone());
                    }
                    _ => eprintln!("Unsupported Rules!"),
                }
            }
        }
    }

    fn permutation(&self, prod_name: &ProdName, expr: &GExpr, father: &GExpr, d: usize) -> Vec<GExpr> {
        let mut ret: Vec<GExpr> = Vec::new();
        if d == 0 {
            match self.cache.get(&(prod_name.clone(), 0)) {
                Some(expressions) => {
                    ret.extend(expressions.clone());
                }
                None => {}
            }
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
                            }

                            _ => match self.cache.get(&(lhs.clone(), d)) {
                                // prevent empty cache with d
                                Some(expressions) => {
                                    ret.extend(expressions.clone());
                                }
                                None => {}
                            },
                        }
                    }
                }
            }
            GExpr::FuncApply(func_name, subs) | GExpr::GFuncApply(func_name, subs) => {
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
                                    match self.cache.get(&(symbol.clone(), d_i[k])) {
                                        Some(expressions) => {
                                            permutation_result.extend(expressions.clone());
                                        }
                                        None => {}
                                    }
                                } else {
                                    permutation_result.push(subs[k].clone())
                                }
                            }
                            _ => eprintln!("There are multiple op in RHS!"),
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
                        GExpr::FuncApply(_, _) => ret.push(GExpr::FuncApply(func_name.clone(), res.clone())),
                        GExpr::GFuncApply(_, _) => ret.push(GExpr::GFuncApply(func_name.clone(), res.clone())),
                        _ => unreachable!(),
                    }
                }
            }
            GExpr::Not(sub) | GExpr::BvNot(sub) | GExpr::BvNeg(sub) => {
                let terms = self.permutation(prod_name, sub, expr, d - 1);
                for t in terms.iter() {
                    //println!("{:?}", t);
                    match expr {
                        GExpr::Not(sub) => ret.push(GExpr::Not(Box::new(t.clone()))),
                        GExpr::BvNot(sub) => ret.push(GExpr::BvNot(Box::new(t.clone()))),
                        GExpr::BvNeg(sub) => ret.push(GExpr::BvNeg(Box::new(t.clone()))),
                        _ => unreachable!(),
                    }
                }
            }
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
            | GExpr::BvUlt(l_sub, r_sub) => {
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
                                GExpr::Equal(_, _) => {
                                    ret.push(GExpr::Equal(Box::new(lt.clone()), Box::new(rt.clone())))
                                }
                                GExpr::BvAnd(_, _) => {
                                    ret.push(GExpr::BvAnd(Box::new(lt.clone()), Box::new(rt.clone())))
                                }
                                GExpr::BvOr(_, _) => ret.push(GExpr::BvOr(Box::new(lt.clone()), Box::new(rt.clone()))),
                                GExpr::BvXor(_, _) => {
                                    ret.push(GExpr::BvXor(Box::new(lt.clone()), Box::new(rt.clone())))
                                }
                                GExpr::BvAdd(_, _) => {
                                    ret.push(GExpr::BvAdd(Box::new(lt.clone()), Box::new(rt.clone())))
                                }
                                GExpr::BvMul(_, _) => {
                                    ret.push(GExpr::BvMul(Box::new(lt.clone()), Box::new(rt.clone())))
                                }
                                GExpr::BvSub(_, _) => {
                                    ret.push(GExpr::BvSub(Box::new(lt.clone()), Box::new(rt.clone())))
                                }
                                GExpr::BvUdiv(_, _) => {
                                    ret.push(GExpr::BvUdiv(Box::new(lt.clone()), Box::new(rt.clone())))
                                }
                                GExpr::BvUrem(_, _) => {
                                    ret.push(GExpr::BvUrem(Box::new(lt.clone()), Box::new(rt.clone())))
                                }
                                GExpr::BvShl(_, _) => {
                                    ret.push(GExpr::BvShl(Box::new(lt.clone()), Box::new(rt.clone())))
                                }
                                GExpr::BvLshr(_, _) => {
                                    ret.push(GExpr::BvLshr(Box::new(lt.clone()), Box::new(rt.clone())))
                                }
                                GExpr::BvUlt(_, _) => {
                                    ret.push(GExpr::BvUlt(Box::new(lt.clone()), Box::new(rt.clone())))
                                }
                                _ => unreachable!(),
                            }
                        }
                    }
                }
            }
            GExpr::Ite(c, t, e) => {
                let mut depth = Vec::new();
                depth_of_subs(d, 3, &mut depth, &mut Vec::new());
                for d_i in depth {
                    let c_terms = self.permutation(prod_name, c, expr, d_i[0]);
                    let t_terms = self.permutation(prod_name, t, expr, d_i[1]);
                    let e_terms = self.permutation(prod_name, e, expr, d_i[2]);

                    for ct in c_terms.iter() {
                        for tt in t_terms.iter() {
                            for et in e_terms.iter() {
                                ret.push(GExpr::Ite(
                                    Box::new(ct.clone()),
                                    Box::new(tt.clone()),
                                    Box::new(et.clone()),
                                ));
                            }
                        }
                    }
                }
            }
            GExpr::ConstBool(_)
            | GExpr::ConstInt(_)
            | GExpr::ConstBitVec(_)
            | GExpr::ConstString(_)
            | GExpr::BvConst(_, _) => {}
            GExpr::Let(_, _) => eprintln!("Please implement Let"),
            _ => eprintln!("Unsupported: {:?}", expr),
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
            // grow
            if self.index >= self.candidates.len() {
                for non_terminal in self.grammar.non_terminals().iter().map(|p| &p.lhs) {
                    self.grow(non_terminal);
                    // add new candidates to the global candidate list
                    match self.cache.get(&(non_terminal.clone(), self.current_size)) {
                        Some(expressions) => {
                            self.candidates.extend(expressions.clone());
                        }
                        None => {}
                    }
                }
                self.current_size += 1;
                self.index = 0;
            }

            while self.index < self.candidates.len() {
                let program = &self.candidates[self.index];
                self.index += 1;
                if !match_any_io(
                    // does not match any counter-example
                    self.oe_cache.get(&format!("{:?}", program)).unwrap(),
                    self.counterexamples,
                ) {
                    return Option::from(program.to_expr());
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
        c: &'a Vec<BaselineCounterExample>,
    ) -> Box<dyn Iterator<Item = Self::Expr> + 'a> {
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
            self::FuncBody {
                name: func_name.to_string(),
                params: p.get_synth_func(func_name).unwrap().0.params.clone(),
                ret_sort: p.get_synth_func(func_name).unwrap().0.ret_sort.clone(),
                body: expr.clone(),
            },
        );

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
            let decl = z3::RecFuncDecl::new(
                &ctx,
                f_name.clone(),
                domain_references.as_slice(),
                &self.sort_to_z3_sort(f_ret_sort, &ctx),
            );

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
                        args.push(z3::ast::Dynamic::from(z3::ast::BV::new_const(
                            &ctx,
                            param.0.clone(),
                            w as u32,
                        )));
                    }
                    Sort::String => {
                        args.push(z3::ast::Dynamic::from(z3::ast::String::new_const(
                            &ctx,
                            param.0.clone(),
                        )));
                    }
                    _ => panic!("Unsupported sort"),
                }
            }

            let args_references: Vec<&dyn z3::ast::Ast> = args.iter().map(|arg| arg as &dyn z3::ast::Ast).collect();
            let mut local_var: HashMap<String, z3::ast::Dynamic> = HashMap::new();
            for param in f_params {
                local_var.insert(
                    param.0.clone(),
                    match param.1 {
                        Sort::Bool => z3::ast::Dynamic::from(z3::ast::Bool::new_const(&ctx, param.0.clone())),
                        Sort::Int => z3::ast::Dynamic::from(z3::ast::Int::new_const(&ctx, param.0.clone())),
                        Sort::BitVec(w) => {
                            z3::ast::Dynamic::from(z3::ast::BV::new_const(&ctx, param.0.clone(), w as u32))
                        }
                        Sort::String => z3::ast::Dynamic::from(z3::ast::String::new_const(&ctx, param.0.clone())),
                        _ => panic!("Unsupported sort"),
                    },
                );
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
            clauses.push(self.expr_to_smt(&c, &vars, &funcs, &ctx).as_bool().unwrap().not());
        }
        let clauses_references: Vec<&z3::ast::Bool<'_>> = clauses.iter().collect();
        solver.assert(&z3::ast::Bool::or(&ctx, clauses_references.as_slice()));

        // solver.check();
        // println!("{:?}", solver.get_model().unwrap());

        match solver.check() {
            z3::SatResult::Unsat => None, // no counter-example found
            z3::SatResult::Unknown => panic!("Unknown z3 solver result"),
            z3::SatResult::Sat => {
                // return value assignment where at least one of the constraints is violated
                let model = solver.get_model().unwrap();
                let mut assignment: HashMap<String, Value> = HashMap::new();
                for (name, sort) in declare_vars {
                    let interp = model.get_const_interp(&vars[&name.clone()]).unwrap();
                    assignment.insert(
                        name.clone(),
                        match sort {
                            Sort::Bool => Value::Bool(interp.as_bool().unwrap().as_bool().unwrap()),
                            Sort::Int => Value::Int(interp.as_int().unwrap().as_i64().unwrap()),
                            Sort::BitVec(_) => Value::BitVec(interp.as_bv().unwrap().as_u64().unwrap()),
                            Sort::Compound(_, _) => unimplemented!("Not supporting compound solving"),
                            Sort::String => Value::String(interp.as_string().unwrap().as_string().unwrap()),
                            Sort::None => panic!("Unsupported sort"),
                        },
                    );
                }

                let vars: Vec<(String, Value)> = assignment.iter().map(|(k, v)| (k.clone(), v.clone())).collect();
                let funcs: Vec<(String, FuncBody)> = define_funs.iter().map(|(k, v)| (k.clone(), v.clone())).collect();
                let env = EvalEnv {
                    vars: vars,
                    funcs: funcs,
                };
                return Some(Self::CounterExample {
                    assignment: assignment,
                    output: expr.eval(&env),
                });
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
            Expr::ConstBitVec(u) => Box::from(z3::ast::Dynamic::from(z3::ast::BV::from_i64(ctx, *u as i64, 32))),
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
            Expr::Equal(e1, e2) => Box::from(z3::ast::Dynamic::from(
                self.expr_to_smt(e1, vars, funcs, ctx)
                    ._eq(&self.expr_to_smt(e2, vars, funcs, ctx)),
            )),
            Expr::Ite(e1, e2, e3) => {
                let cond = self.expr_to_smt(e1, vars, funcs, ctx).as_bool().unwrap();
                let then = self.expr_to_smt(e2, vars, funcs, ctx);
                let els = self.expr_to_smt(e3, vars, funcs, ctx);
                Box::from(z3::ast::Dynamic::from(z3::ast::Bool::ite(&cond, &*then, &*els)))
            }
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
            Expr::BvConst(v, width) => Box::from(z3::ast::Dynamic::from(z3::ast::BV::from_i64(ctx, *v, *width as u32))),
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
    #![allow(warnings)]
    use std::fs;

    use crate::{parser::parse, solver::Solver};

    use super::{depth_of_subs, Expr, Sort};

    #[test]
    fn test_permutation_1() {
        let mut ans = Vec::new();
        let mut tmp = Vec::new();
        depth_of_subs(4, 3, &mut ans, &mut tmp);
        assert_eq!(
            ans,
            [
                [0, 0, 3],
                [0, 1, 2],
                [0, 2, 1],
                [0, 3, 0],
                [1, 0, 2],
                [1, 1, 1],
                [1, 2, 0],
                [2, 0, 1],
                [2, 1, 0],
                [3, 0, 0]
            ]
        )
    }

    #[test]
    fn test_permutation_2() {
        let mut ans = Vec::new();
        let mut tmp = Vec::new();
        depth_of_subs(2, 2, &mut ans, &mut tmp);
        assert_eq!(ans, [[0, 1], [1, 0]])
    }

    #[test]
    fn test_permutation_3() {
        let mut ans = Vec::new();
        let mut tmp = Vec::new();
        depth_of_subs(3, 2, &mut ans, &mut tmp);
        assert_eq!(ans, [[0, 2], [1, 1], [2, 0]])
    }

    #[test]
    fn test_oe() {
        unimplemented!()
    }

    #[test]
    fn test_verify_1() {
        let filename = format!("{}/{}", env!("CARGO_MANIFEST_DIR"), format!("test_bool_1.sl"));
        let input = fs::read_to_string(&filename).unwrap();
        let prog = match parse(&input) {
            Ok(res) => res,
            Err(e) => {
                panic!("Error parsing file: {}\nError: {:#?}", filename, e);
            }
        };
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
        let expr = Expr::And(
            Box::from(Expr::Var("a".to_string(), Sort::Bool)),
            Box::from(Expr::Var("b".to_string(), Sort::Bool)),
        );
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
            Box::from(Expr::Not(Box::from(Expr::Var("a".to_string(), Sort::Bool)))),
            Box::from(Expr::Var("b".to_string(), Sort::Bool)),
        );
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
        let expr = Expr::Not(Box::from(Expr::And(
            Box::from(Expr::Not(Box::from(Expr::Var("a".to_string(), Sort::Bool)))),
            Box::from(Expr::Not(Box::from(Expr::Var("b".to_string(), Sort::Bool)))),
        )));
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
        let prog = match parse(&input) {
            Ok(res) => res,
            Err(e) => {
                panic!("Error parsing file: {}\nError: {:#?}", filename, e);
            }
        };
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
        let expr = Expr::And(
            Box::from(Expr::Var("a".to_string(), Sort::Bool)),
            Box::from(Expr::Var("b".to_string(), Sort::Bool)),
        );
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
                Box::from(Expr::Var("b".to_string(), Sort::Bool)),
            )),
            Box::from(Expr::Var("c".to_string(), Sort::Bool)),
        );
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
        let expr = Expr::Not(Box::from(Expr::And(
            Box::from(Expr::And(
                Box::from(Expr::Not(Box::from(Expr::Var("a".to_string(), Sort::Bool)))),
                Box::from(Expr::Not(Box::from(Expr::Var("b".to_string(), Sort::Bool)))),
            )),
            Box::from(Expr::Not(Box::from(Expr::Var("c".to_string(), Sort::Bool)))),
        )));
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
