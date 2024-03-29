use std::cmp::max;
use std::collections::{HashMap, HashSet};
use std::hash::Hash;
use async_std::task::{self, yield_now};

use egg::{rewrite as rw, *};
use itertools::Itertools;
use z3_sys::Z3_eval_smtlib2_string;
use std::vec;

use strum_macros::Display;
use z3::ast::{
    Ast,
    Dynamic
};
use z3::{
    ast::Bool,
    Config,
    Context,
    Solver
};
use crate::parser::ast::{self, Expr, FuncName, Sort};

define_language! {
    pub enum SyGuSLanguage {
        ConstBool(bool),
        ConstInt(i64),
        ConstBitVec(u64),
        ConstString(String),
        Var(String),
        FuncApply(FuncName, Vec<Id>),
        BvConst(i64),
        // Bool
        "not" = Not([Id; 1]),
        "and" = And([Id; 2]),
        "or" = Or([Id; 2]),
        "xor" = Xor([Id; 2]),
        "iff" = Iff([Id; 2]),
        "=" = Equal([Id; 2]),
        // BitVec
        "bvand" = BvAnd([Id; 2]),
        "bvor" = BvOr([Id; 2]),
        "bvxor" = BvXor([Id; 2]),
        "bvnot" = BvNot([Id; 1]),
        "bvadd" = BvAdd([Id; 2]),
        "bvmul" = BvMul([Id; 2]),
        "bvsub" = BvSub([Id; 2]),
        "bvudiv" = BvUdiv([Id; 2]),
        "bvurem" = BvUrem([Id; 2]),
        "bvshl" = BvShl([Id; 2]),
        "bvlshr" = BvLshr([Id; 2]),
        "bvneg" = BvNeg([Id; 1]),
        "bvult" = BvUlt([Id; 2]),
    }
}
impl SyGuSLanguage {
    pub fn semantics(&self) -> Box<dyn Fn(&[(HashMap<String, i64>, i64)]) -> i64 + '_> {
        match self {
            SyGuSLanguage::ConstBool(b) => Box::new(move |_| if *b { 1 } else { 0 }),
            SyGuSLanguage::ConstInt(n) => Box::new(move |_| *n),
            SyGuSLanguage::ConstBitVec(n) => Box::new(move |_| *n as i64),
            SyGuSLanguage::ConstString(_) => unimplemented!("Semantics for ConstString not implemented"),
            SyGuSLanguage::Var(name) => Box::new(move |env| {
                env.iter()
                    .find_map(|(input, _)| input.get(name))
                    .cloned()
                    .unwrap_or(0)
            }),
            SyGuSLanguage::FuncApply(_, _) => unimplemented!("Semantics for FuncApply not implemented"),
            SyGuSLanguage::BvConst(n) => Box::new(move |_| *n),
            SyGuSLanguage::Not([_]) => Box::new(move |args| if args[0].1 == 0 { 1 } else { 0 }),
            SyGuSLanguage::And([_, _]) => Box::new(move |args| if args[0].1 != 0 && args[1].1 != 0 { 1 } else { 0 }),
            SyGuSLanguage::Or([_, _]) => Box::new(move |args| if args[0].1 != 0 || args[1].1 != 0 { 1 } else { 0 }),
            SyGuSLanguage::Xor([_, _]) => Box::new(move |args| if args[0].1 != args[1].1 { 1 } else { 0 }),
            SyGuSLanguage::Iff([_, _]) => Box::new(move |args| if args[0].1 == args[1].1 { 1 } else { 0 }),
            SyGuSLanguage::Equal([_, _]) => Box::new(move |args| if args[0].1 == args[1].1 { 1 } else { 0 }),
            SyGuSLanguage::BvAnd([_, _]) => Box::new(move |args| args[0].1 & args[1].1),
            SyGuSLanguage::BvOr([_, _]) => Box::new(move |args| args[0].1 | args[1].1),
            SyGuSLanguage::BvXor([_, _]) => Box::new(move |args| args[0].1 ^ args[1].1),
            SyGuSLanguage::BvNot([_]) => Box::new(move |args| !args[0].1),
            SyGuSLanguage::BvAdd([_, _]) => Box::new(move |args| args[0].1 + args[1].1),
            SyGuSLanguage::BvMul([_, _]) => Box::new(move |args| args[0].1 * args[1].1),
            SyGuSLanguage::BvSub([_, _]) => Box::new(move |args| args[0].1 - args[1].1),
            SyGuSLanguage::BvUdiv([_, _]) => Box::new(move |args| args[0].1 / args[1].1),
            SyGuSLanguage::BvUrem([_, _]) => Box::new(move |args| args[0].1 % args[1].1),
            SyGuSLanguage::BvShl([_, _]) => Box::new(move |args| args[0].1 << args[1].1),
            SyGuSLanguage::BvLshr([_, _]) => Box::new(move |args| (args[0].1 as u64 >> args[1].1) as i64),
            SyGuSLanguage::BvNeg([_]) => Box::new(move |args| -args[0].1),
            SyGuSLanguage::BvUlt([_, _]) => Box::new(move |args| if (args[0].1 as u64) < (args[1].1 as u64) { 1 } else { 0 }),
        }
    }
}
type ProdName = String;
type IOPairs = Vec<(HashMap<String, i64>, i64)>;
type IOPairsRef<'a> = Vec<(&'a HashMap<String, i64> , i64)>;

#[derive(Debug, Clone)]
struct Production {
    lhs: ProdName,
    lhs_type: String,
    rhs: Vec<ProdComponent>,
}

#[derive(Debug, Clone)]
enum ProdComponent {
    LhsName(ProdName),
    LanguageConstruct(SyGuSLanguage),
}

#[derive(Debug, Clone)]
struct Grammar {
    productions: Vec<Production>,
}

#[derive(Default, Debug, Clone)]
struct ObsEquiv {
    pts: IOPairs,
}

impl Analysis<SyGuSLanguage> for ObsEquiv {
    type Data = IOPairs;

    fn merge(&mut self, to: &mut Self::Data, from: Self::Data) -> DidMerge {
        // println!("<ObsEquiv::merge> to: {:?}, from: {:?}", to, from);
        let mut merged = false;
        for (from_input, from_output) in from {
            if !to.contains(&(from_input.clone(), from_output)) {
                to.push((from_input, from_output));
                merged = true;
            }
        }
        DidMerge(merged, false)
    }

    fn make(egraph: &EGraph<SyGuSLanguage, Self>, enode: &SyGuSLanguage) -> Self::Data {
        let pts: &Vec<(HashMap<String, i64>, i64)> = &egraph.analysis.pts;
        let sem = enode.semantics();
        let o = |i: &Id| &egraph[*i].data; // output
        match enode {
            // Constants
            SyGuSLanguage::ConstBool(b) => pts.iter().map(|(input, _)| (input.clone(), if *b { 1 } else { 0 })).collect(),
            SyGuSLanguage::ConstInt(n) => pts.iter().map(|(input, _)| (input.clone(), *n)).collect(),
            SyGuSLanguage::ConstBitVec(n) => pts.iter().map(|(input, _)| (input.clone(), *n as i64)).collect(),
            SyGuSLanguage::ConstString(_) => unimplemented!("Semantics for ConstString not implemented"),
            SyGuSLanguage::BvConst(n) => pts.iter().map(|(input, _)| (input.clone(), *n)).collect(),
            // Variables
            SyGuSLanguage::Var(name) => pts.iter().map(|(input, _)| (input.clone(), sem(&[(input.clone(), *input.get(name).unwrap_or(&0))]))).collect(),
            // Function application
            SyGuSLanguage::FuncApply(_, _) => unimplemented!("Semantics for FuncApply not implemented"),
            // Unary operators
            SyGuSLanguage::Not([id]) => o(id)
                .iter()
                .zip(pts)
                .map(|((input, output), _)| (input.clone(), sem(&[(input.clone(), *output)])))
                .collect(),
            SyGuSLanguage::BvNot([id]) => o(id)
                .iter()
                .zip(pts)
                .map(|((input, output), _)| (input.clone(), sem(&[(input.clone(), *output)])))
                .collect(),
            SyGuSLanguage::BvNeg([id]) => o(id)
                .iter()
                .zip(pts)
                .map(|((input, output), _)| (input.clone(), sem(&[(input.clone(), *output)])))
                .collect(),
            // Binary operators
            SyGuSLanguage::And([a, b]) | SyGuSLanguage::Or([a, b]) | SyGuSLanguage::Xor([a, b]) |
            SyGuSLanguage::Iff([a, b]) | SyGuSLanguage::Equal([a, b]) |
            SyGuSLanguage::BvAnd([a, b]) | SyGuSLanguage::BvOr([a, b]) | SyGuSLanguage::BvXor([a, b]) |
            SyGuSLanguage::BvAdd([a, b]) | SyGuSLanguage::BvMul([a, b]) | SyGuSLanguage::BvSub([a, b]) |
            SyGuSLanguage::BvUdiv([a, b]) | SyGuSLanguage::BvUrem([a, b]) | SyGuSLanguage::BvShl([a, b]) |
            SyGuSLanguage::BvLshr([a, b]) | SyGuSLanguage::BvUlt([a, b]) => o(a)
                .iter()
                .zip(o(b))
                .zip(pts)
                .map(|(((input_a, output_a), (input_b, output_b)), _)| {
                    let input = input_a.clone();
                    let output = sem(&[(input_a.clone(), *output_a), (input_b.clone(), *output_b)]);
                    (input, output)
                })
                .collect(),
        }
    }
    fn modify(egraph: &mut EGraph<SyGuSLanguage, Self>, id: Id) {
        // // We don't need to do anything here because the data consistency is guaranteed by the `enumerator.rebuild()`.
        // let io_pairs = egraph[id].data.clone();
        // if io_pairs.is_empty() || io_pairs.first().unwrap().0.is_empty() {
        //     return;
        // }
        // for (inputs, output) in io_pairs {
        //     let expr = egraph.id_to_expr(id);
        //     let new_id = egraph.add_expr(&expr);
        //     egraph.rebuild();
        //     let new_output = egraph[new_id].data.iter().find(|(i, _)| i == &inputs).map(|(_, o)| *o).unwrap();
        //     if new_output != output {
        //         // the second element of the IOPair
        //         let num_id = egraph.add(ArithLanguage::Num(new_output));
        //         egraph.union(id, num_id);
        //     }
        // }
    }
}

struct Enumerator<'a> {
    old_egraphs: Vec<EGraph<SyGuSLanguage, ObsEquiv>>,
    grammar: &'a Grammar,
    egraph: EGraph<SyGuSLanguage, ObsEquiv>,
    cache: HashMap<(ProdName, usize), HashSet<RecExpr<SyGuSLanguage>>>,
    checked_exprs: HashSet<RecExpr<SyGuSLanguage>>,
    current_size: usize,
}

impl<'a> Enumerator<'a> {
    fn new(grammar: &'a Grammar) -> Self {
        Enumerator {
            old_egraphs: Vec::new(),
            grammar,
            egraph: EGraph::new(ObsEquiv::default()).with_explanations_enabled(),
            cache: HashMap::new(),
            checked_exprs: HashSet::new(),
            current_size: 0,
        }
    }

    fn rebuild(&mut self, pts: &IOPairs) {
        let mut new_egraph = EGraph::new(ObsEquiv { pts: pts.clone() }).with_explanations_enabled();
        for (key, exprs) in &self.cache {
            for expr in exprs {
                new_egraph.add_expr(expr);
            }
        }
        self.merge_equivs();
        new_egraph.rebuild();
        self.egraph = new_egraph;
    }

    // fn rebuild(&mut self, pts: &IOPairs) {
    //     let mut inconsistent_eclasses = HashSet::new();
    //     // Identify inconsistent eclasses
    //     for (key, exprs) in &self.cache {
    //         for expr in exprs {
    //             let id = self.egraph.add_expr(expr);
    //             let eclass = &self.egraph[id];

    //             let mut output = HashSet::new();
    //             for (inputs, _) in pts {
    //                 for node in &eclass.nodes {
    //                     if let Some((i, o)) = ObsEquiv::make(&self.egraph, node).iter().find(|(i, _)| i == inputs) {
    //                         output.insert(*o);
    //                     }
    //                     if output.len() > 1 {
    //                         break;
    //                     }
    //                 }
    //                 if output.len() > 1 {
    //                     inconsistent_eclasses.insert(id);
    //                     break;
    //                 }
    //             }
    //         }
    //     }

    //     let new_egraph = EGraph::new(ObsEquiv { pts: pts.clone() }).with_explanations_enabled();
    //     let mut old_egraph = std::mem::replace(&mut self.egraph, new_egraph);

    //     // Add consistent eclasses to the new egraph
    //     for (key, exprs) in &self.cache {
    //         for expr in exprs {
    //             let id = old_egraph.add_expr(expr);
    //             if !inconsistent_eclasses.contains(&id) {
    //                 self.egraph.add_expr(expr);
    //             }
    //         }
    //     }
    //     // Add inconsistent eclasses to the new egraph
    //     for id in inconsistent_eclasses {
    //         for node in &old_egraph[id].nodes {
    //             let new_id = self.egraph.add(node.clone());
    //         }
    //     }
    //     // Merge equivalent expressions in the new egraph
    //     self.merge_equivs();
    //     self.egraph.rebuild();
    // }

    fn collect_equivs(&self) -> HashMap<Vec<(Vec<(String, i64)>, i64)>, HashSet<Id>> {
        let mut equivs: HashMap<Vec<(Vec<(String, i64)>, i64)>, HashSet<Id>> = HashMap::new();
        for eclass in self.egraph.classes() {
            let data: Vec<(Vec<(String, i64)>, i64)> = eclass.data.iter().map(|(i, o)| {
                let mut vec: Vec<(String, i64)> = i.clone().into_iter().collect();
                vec.sort();
                (vec, *o)
            }).collect::<Vec<_>>();
            let key = data;
            equivs.entry(key).or_insert(HashSet::new()).insert(eclass.id);
        }
        equivs
    }

    fn merge_equivs(&mut self) {
        let equivs = self.collect_equivs();
        // merge those equivs
        for (k, v) in equivs {
            let mut iter = v.into_iter();
            if let Some(first) = iter.next() {
                for id in iter {
                    self.egraph.union(first, id);
                }
            }
        }
    }

    /* async */ fn grow(&mut self) {
        let size = self.current_size + 1;
        // Base case: directly add constants and variables for size 1
        if size == 1 {
            for prod in &self.grammar.productions {
                for component in &prod.rhs {
                    if let ProdComponent::LanguageConstruct(lang_construct) = component {
                        println!("<Enumerator::grow> prod(size={}): ({}, {}) => {:?}", size, prod.lhs, prod.lhs_type, lang_construct);
                        let mut new_expressions: HashMap<(String, usize), HashSet<RecExpr<SyGuSLanguage>>> = HashMap::new();
                        match lang_construct {
                            SyGuSLanguage::ConstBool(_) | SyGuSLanguage::ConstInt(_) | SyGuSLanguage::ConstBitVec(_) |
                            SyGuSLanguage::ConstString(_) | SyGuSLanguage::BvConst(_) | SyGuSLanguage::Var(_) => {
                                let id = self.egraph.add(lang_construct.clone());
                                let expr = self.egraph.id_to_expr(id);
                                new_expressions.entry((prod.lhs.clone(), AstSize.cost_rec(&expr)))
                                               .or_insert_with(HashSet::new)
                                               .insert(expr);
                            },
                            _ => {}
                        }
                        for (key, exprs) in new_expressions {
                            self.cache.entry(key).or_insert_with(HashSet::new).extend(exprs);
                        }
                    }
                }
            }
        } else {
            // Composite expressions
            for prod in &self.grammar.productions {
                for component in &prod.rhs {
                    if let ProdComponent::LanguageConstruct(lang_construct) = component {
                        println!("<Enumerator::grow> prod(size={}): ({}, {}) => {:?}", size, prod.lhs, prod.lhs_type, lang_construct);
                        let mut new_expressions: HashMap<(String, usize), HashSet<RecExpr<SyGuSLanguage>>> = HashMap::new();
                        match lang_construct {
                            SyGuSLanguage::And(_) | SyGuSLanguage::Or(_) | SyGuSLanguage::Xor(_) | SyGuSLanguage::Iff(_) |
                            SyGuSLanguage::Equal(_) | SyGuSLanguage::BvAnd(_) | SyGuSLanguage::BvOr(_) | SyGuSLanguage::BvXor(_) |
                            SyGuSLanguage::BvAdd(_) | SyGuSLanguage::BvMul(_) | SyGuSLanguage::BvSub(_) | SyGuSLanguage::BvUdiv(_) |
                            SyGuSLanguage::BvUrem(_) | SyGuSLanguage::BvShl(_) | SyGuSLanguage::BvLshr(_) | SyGuSLanguage::BvUlt(_) => {
                                let num_nonterminals = lang_construct.children().len();
                                for left_size in 1..size {
                                    let right_size = size - left_size;
                                    let left_exprs = self.cache.get(&(prod.lhs.clone(), left_size)).cloned().unwrap();
                                    let right_exprs = self.cache.get(&(prod.lhs.clone(), right_size)).cloned().unwrap();
                                    for left_expr in &left_exprs {
                                        for right_expr in &right_exprs {
                                            let left_id = self.egraph.add_expr(left_expr);
                                            let right_id = self.egraph.add_expr(right_expr);
    
                                            let id = match lang_construct {
                                                SyGuSLanguage::And(_) => self.egraph.add(SyGuSLanguage::And([left_id, right_id])),
                                                SyGuSLanguage::Or(_) => self.egraph.add(SyGuSLanguage::Or([left_id, right_id])),
                                                SyGuSLanguage::Xor(_) => self.egraph.add(SyGuSLanguage::Xor([left_id, right_id])),
                                                SyGuSLanguage::Iff(_) => self.egraph.add(SyGuSLanguage::Iff([left_id, right_id])),
                                                SyGuSLanguage::Equal(_) => self.egraph.add(SyGuSLanguage::Equal([left_id, right_id])),
                                                SyGuSLanguage::BvAnd(_) => self.egraph.add(SyGuSLanguage::BvAnd([left_id, right_id])),
                                                SyGuSLanguage::BvOr(_) => self.egraph.add(SyGuSLanguage::BvOr([left_id, right_id])),
                                                SyGuSLanguage::BvXor(_) => self.egraph.add(SyGuSLanguage::BvXor([left_id, right_id])),
                                                SyGuSLanguage::BvAdd(_) => self.egraph.add(SyGuSLanguage::BvAdd([left_id, right_id])),
                                                SyGuSLanguage::BvMul(_) => self.egraph.add(SyGuSLanguage::BvMul([left_id, right_id])),
                                                SyGuSLanguage::BvSub(_) => self.egraph.add(SyGuSLanguage::BvSub([left_id, right_id])),
                                                SyGuSLanguage::BvUdiv(_) => self.egraph.add(SyGuSLanguage::BvUdiv([left_id, right_id])),
                                                SyGuSLanguage::BvUrem(_) => self.egraph.add(SyGuSLanguage::BvUrem([left_id, right_id])),
                                                SyGuSLanguage::BvShl(_) => self.egraph.add(SyGuSLanguage::BvShl([left_id, right_id])),
                                                SyGuSLanguage::BvLshr(_) => self.egraph.add(SyGuSLanguage::BvLshr([left_id, right_id])),
                                                SyGuSLanguage::BvUlt(_) => self.egraph.add(SyGuSLanguage::BvUlt([left_id, right_id])),
                                                _ => unreachable!(),
                                            };
    
                                            let expr = self.egraph.id_to_expr(id);
                                            new_expressions.entry((prod.lhs.clone(), AstSize.cost_rec(&expr)))
                                                           .or_insert_with(HashSet::new)
                                                           .insert(expr);
                                        }
                                    }
                                }
                            },
                            SyGuSLanguage::Not(_) | SyGuSLanguage::BvNot(_) | SyGuSLanguage::BvNeg(_) => {
                                for part_size in 1..size {
                                    let mut expr_parts = Vec::new();
                                    if let Some(exprs) = self.cache.get(&(prod.lhs.clone(), part_size)) {
                                        expr_parts.push(exprs.iter().cloned().collect::<Vec<_>>());
                                    }
                                    for expr in expr_parts.into_iter().flatten() {
                                        let mut new_expr = expr.clone();
                                        let id = new_expr.add(lang_construct.clone()); // we don't care the id of a unary operator
                                        self.egraph.add_expr(&new_expr);
                                        new_expressions.entry((prod.lhs.clone(), AstSize.cost_rec(&new_expr)))
                                                        .or_insert_with(HashSet::new)
                                                        .insert(new_expr);
                                    }
                                }
                            },
                            _ => {}
                        }
                        for (key, exprs) in new_expressions {
                            self.cache.entry(key).or_insert_with(HashSet::new).extend(exprs);
                        }
                    }
                }
            }
        }
        self.merge_equivs();
        self.egraph.rebuild();
        self.current_size = size;
    }


    /* async */fn enumerate(&mut self, size: usize, pts: &IOPairs) -> Vec<RecExpr<SyGuSLanguage>> {
        println!("<Enumerator::enumerate> current size: {}", self.current_size);
        if self.egraph.analysis.pts.len() != pts.len() {
            println!("<Enumerator::enumerate> egraph.analysis.pts: {:?}", self.egraph.analysis.pts);
            self.rebuild(pts);
            println!("<Enumerator::enumerate> egraph.analysis.pts: {:?}", self.egraph.analysis.pts);
        }
        let mut result = vec![];
        while self.current_size <= size {
            if self.current_size < size {
                println!("<Enumerator::enumerate> Growing to size: {}", self.current_size + 1);
                self.grow()/*.await*/;
            }
            let start_nonterminal = &self.grammar.productions.first().unwrap().lhs;
            let cache_max_size = self.cache.iter().map(|(k, _)| k.1).max().unwrap();
            let exprs = self.cache.iter().filter(|(k, _)| &k.0 == start_nonterminal && k.1 <= cache_max_size).map(|(_, v)| v).flatten().collect::<Vec<_>>();
            result.clear();
            for expr in exprs {
                if self.satisfies_pts(expr, pts) {
                    println!("<Enumerator::enumerate> expr: {:?} satisfies pts: {:?}", expr.pretty(100), pts);
                    result.push(expr.clone());
                }
            }
            if !result.is_empty() {
                break;
            }
            if self.current_size == size {
                break;
            }
        }
        result
    }


    fn satisfies_pts(&self, expr: &RecExpr<SyGuSLanguage>, pts: &IOPairs) -> bool {
        // println!("<Enumerator::satisfies_pts> expr: {:?}", expr.pretty(100));
        if pts.is_empty() || pts.first().unwrap().0.is_empty() {
            return true;
        }
        let mut egraph = EGraph::new(ObsEquiv { pts: pts.clone() }).with_explanations_enabled();
        let id = egraph.add_expr(expr);
        egraph.rebuild();
        // if format!("{}",expr.pretty(100)) == "(+ y (* x 2))" {
        //     let new_id = egraph.add_expr(&expr);
        //     println!("<Enumerator::satisfies_pts> egraph[id].data: {:?}", egraph[id].data);
        //     println!("<Enumerator::satisfies_pts> self.egraph[id].data: {:?}", self.egraph[new_id].data);
        //     println!("<Enumerator::satisfies_pts> pts: {:?}", pts);
        // }
        let mut result = true;
        for (inputs, output) in pts {
            if egraph[id].data.iter().any(|(i, o)| &i == &inputs && &o == &output) {
                result &= true;
            } else {
                result &= false;
                break;
            }
        }
        result
    }
}

struct EggSolver<'a> {
    enumerator: Enumerator<'a>,
}

impl<'a> EggSolver<'a> {
    fn new(enumerator: Enumerator<'a>) -> Self {
        EggSolver { 
            enumerator
        }
    }

    /*async */fn synthesize(&mut self, max_size: usize, pts_all: &IOPairs, var_decls: &[(String, Sort)], target_fn: &dyn Fn(&HashMap<String, i64>) -> i64) -> Vec<RecExpr<SyGuSLanguage>> {
        let mut pts = vec![];
        let start = std::time::Instant::now();
        println!("<EggSolver::synthesize> Start time: {:?}", start);
        while self.enumerator.current_size <= max_size && start.elapsed().as_secs() <= 120 {
            let exprs = self.enumerator.enumerate(max_size, &pts)/*.await */;
            let exprs_sat = exprs.iter().filter(|expr| self.verify(expr, var_decls, target_fn).is_ok()).collect::<Vec<_>>();
            // let cexs: Option<_> = exprs.iter().map(|expr| self.verify(expr, pts_all)).fold(None, |acc, cex| acc.or(cex));
            // let cexs: Vec<_> = exprs.iter().map(|expr| self.verify(expr, pts_all)).filter(|cex| cex.is_some()).map(|cex| cex.unwrap()).unique_by(|(inputs, output)| inputs.iter
            if !exprs_sat.is_empty() {
                // target found
                // println!("Target found!");
                // println!("{}", pretty_cache(&enumerator.cache, 2));
                println!("{}", pretty_egraph(&self.enumerator.egraph, 2));
                // println!("Time elapsed: {:?}", start.elapsed());
                return exprs_sat.iter().cloned().map(|expr| expr.clone()).collect();
            } else if !exprs.is_empty() {
                // cannot find target within current size
                let expr = exprs.first().unwrap();
                if let Some(cex) = self.verify(expr, var_decls, target_fn).err() {
                    pts.push((cex.clone(), target_fn(&cex)));
                } else {
                    // unreachable
                }
            } {
                // unreachable
            }
        }
        vec![]
    }

    fn verify(
        &self,
        expr: &RecExpr<SyGuSLanguage>,
        var_decls: &[(String, Sort)],
        target_fn: &dyn Fn(&HashMap<String, i64>) -> i64,
    ) -> Result<(), HashMap<String, i64>> {
        let cfg = Config::new();
        let ctx = Context::new(&cfg);
        let solver = Solver::new(&ctx);
        // TODO: use z3 to parse the expression and evaluate it
        // let return_val = unsafe {
        //     Z3_eval_smtlib2_string(
        //         ctx, 
        //         self.expr_str(expr).as_ptr()
        //     )
        // };
        Ok(())
    }

    fn expr_str(&self, expr: &RecExpr<SyGuSLanguage>) -> String {
        expr.pretty(100)
    }
}
#[async_std::main]
async fn main() {
    let grammar = Grammar {
        productions: vec![
            Production {
                lhs: "S".to_string(),
                lhs_type: "Op".to_string(),
                rhs: vec![
                    ProdComponent::LhsName("S".to_string()),
                    ProdComponent::LanguageConstruct(SyGuSLanguage::BvAdd(Default::default())),
                    ProdComponent::LhsName("S".to_string()),
                ]
            },
            Production {
                lhs: "S".to_string(),
                lhs_type: "Op".to_string(),
                rhs: vec![
                    ProdComponent::LhsName("S".to_string()),
                    ProdComponent::LanguageConstruct(SyGuSLanguage::BvMul(Default::default())),
                    ProdComponent::LhsName("S".to_string()),
                ]
            },
            Production {
                lhs: "S".to_string(),
                lhs_type: "Var".to_string(),
                rhs: vec![
                    ProdComponent::LanguageConstruct(SyGuSLanguage::Var("x".to_string())),
                    ProdComponent::LanguageConstruct(SyGuSLanguage::Var("y".to_string())),
                ]
            },
            Production {
                lhs: "S".to_string(),
                lhs_type: "Const".to_string(),
                rhs: vec![ProdComponent::LanguageConstruct(SyGuSLanguage::BvConst(0))]
            },
            Production {
                lhs: "S".to_string(),
                lhs_type: "Const".to_string(),
                rhs: vec![ProdComponent::LanguageConstruct(SyGuSLanguage::BvConst(1))]
            },
            Production {
                lhs: "S".to_string(),
                lhs_type: "Const".to_string(),
                rhs: vec![ProdComponent::LanguageConstruct(SyGuSLanguage::BvConst(2))]
            },
        ]
    };

    let enumerator = Enumerator::new(&grammar);
    let mut solver = EggSolver::new(enumerator);
    let max_size = 7;

    let var_decls = vec![
        ("x".to_string(), Sort::BitVec(32)),("y".to_string(), Sort::BitVec(32)),
        ];

    let target_fn = |inputs: &HashMap<String, i64>| {
        let x = *inputs.get("x").unwrap() as u64;
        let y = *inputs.get("y").unwrap() as u64;
        ((x * x) + (y * 2)) as i64
    };

    let pts = vec![
        (HashMap::from([("x".to_string(), 1), ("y".to_string(), 2)]), 5), // 1 * 1 + 2 * 2 = 5
        (HashMap::from([("x".to_string(), 3), ("y".to_string(), 4)]), 17), // 3 * 3 + 4 * 2 = 17
        (HashMap::from([("x".to_string(), 4), ("y".to_string(), 1)]), 18), // 4 * 4 + 1 * 2 = 18
        (HashMap::from([("x".to_string(), 5), ("y".to_string(), 3)]), 31), // 5 * 5 + 3 * 2 = 31
    ];

    let var_decls = vec![
        ("x".to_string(), Sort::BitVec(32)),
        ("y".to_string(), Sort::BitVec(32)),
    ];

    let exprs = solver.synthesize(max_size, &pts, &var_decls, &target_fn)/*.await*/;
    if exprs.is_empty() {
        println!("No expression could be synthesized.");
        assert!(false);
    } else {
        for expr in &exprs {
            println!("Synthesized expression: {}", expr.pretty(100));
            match solver.verify(expr, &var_decls, &target_fn) {
                Ok(()) => println!("Verification succeeded"),
                Err(counter_example) => {
                    println!("Verification failed with counter-example: {:?}", counter_example);
                    assert!(false);
                }
            }
        }
    }
}

fn pretty_cache(cache: &HashMap<(ProdName, usize), HashSet<RecExpr<SyGuSLanguage>>>, starting_space: usize)  -> String {
    let mut result = String::new();
    result.push_str("Cache:\n");
    for (key, exprs) in cache {
        for expr in exprs {
            // println!("[{}, {}]: {}", &key.0, &key.1, expr.pretty(100));
            // println!("{}[{}, {}]: {}", " ".repeat(starting_space), &key.0, &key.1, expr.pretty(100));
            result.push_str(&format!("{}[{}, {}]: {}\n", " ".repeat(starting_space), &key.0, &key.1, expr.pretty(100)));
        }
    }
    result.trim().to_string()
}

fn pretty_egraph(egraph: &EGraph<SyGuSLanguage, ObsEquiv>, starting_space: usize) -> String {
    let mut result = String::new();
    result.push_str("EGraph:\n");
    for eclass in egraph.classes() {
        let expr = egraph.id_to_expr(eclass.id);
        result.push_str(&format!("{}{}:[{}] {:?}\n", " ".repeat(starting_space), eclass.id, expr.pretty(100), egraph[eclass.id].data.iter().map(| (inputs, output) | output).collect::<Vec<_>>()));
        for eqc in eclass.iter() {
            result.push_str(&format!("{}<{}>\n", " ".repeat(starting_space * 2), eqc));
        }
        for (inputs, output) in &egraph[eclass.id].data {
            result.push_str(&format!("{}{} -> {}\n", " ".repeat(starting_space * 2), inputs.iter().map(|(k, v)| format!("{:?}: {}", k, v)).collect::<Vec<_>>().join(", "), output));
        }
    }
    result.trim().to_string()
}

fn merge_egraphs<L: Language, N: Analysis<L>>(source_egraph: &EGraph<L, N>, target_egraph: &mut EGraph<L, N>) {
    for id in source_egraph.classes().map(|e| e.id) {
        let expr = source_egraph.id_to_expr(id);
        target_egraph.add_expr(&expr);
    }
    target_egraph.rebuild();
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_main() {
        main();
    }
}
