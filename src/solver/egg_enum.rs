use std::cmp::max;
use std::collections::{HashMap, HashSet, VecDeque};
use std::hash::Hash;
use async_std::task::{self, yield_now};

use cfg::earley::grammar;
use egg::{rewrite as rw, *};
use indexmap::IndexSet;
use itertools::{enumerate, Itertools};

use crate::parser::ast::Expr;

define_language! {
    pub enum ArithLanguage {
        Num(i32),
        Var(String),
        "+" = Add([Id; 2]),
        "-" = Sub([Id; 2]),
        "*" = Mul([Id; 2]),
        "-" = Neg([Id; 1]),
    }
}
impl ArithLanguage {
    // Credit to egsolver
    pub fn semantics(&self) -> Box<dyn Fn(&[(HashMap<String, i32>, i32)]) -> i32 + '_> {
        match self {
            ArithLanguage::Num(n) => Box::new(move |_| *n),
            ArithLanguage::Var(name) => Box::new(move |env| {
                env.iter()
                    .find_map(|(input, _)| input.get(name))
                    .cloned()
                    .unwrap_or(0)
            }),
            ArithLanguage::Neg([_]) => Box::new(move |args| -args[0].1),
            ArithLanguage::Add([_, _]) => Box::new(move |args| args[0].1 + args[1].1),
            ArithLanguage::Sub([_, _]) => Box::new(move |args| args[0].1 - args[1].1),
            ArithLanguage::Mul([_, _]) => Box::new(move |args| args[0].1 * args[1].1),
        }
    }
    pub fn assign_ids(&self, ids: &[Id]) -> Self {
        match self {
            ArithLanguage::Num(n) => ArithLanguage::Num(*n),
            ArithLanguage::Var(name) => ArithLanguage::Var(name.clone()),
            ArithLanguage::Neg([_]) => ArithLanguage::Neg([ids[0]]),
            ArithLanguage::Add([_, _]) => ArithLanguage::Add([ids[0], ids[1]]),
            ArithLanguage::Sub([_, _]) => ArithLanguage::Sub([ids[0], ids[1]]),
            ArithLanguage::Mul([_, _]) => ArithLanguage::Mul([ids[0], ids[1]]),
        }
    }
}
type ProdName = String;
type IOPairs = Vec<(HashMap<String, i32>, i32)>;
type IOPairsRef<'a> = Vec<(&'a HashMap<String, i32> , i32)>;

#[derive(Debug, Clone)]
struct Production {
    lhs: ProdName,
    lhs_type: String,
    rhs: Vec<ProdComponent>,
}

#[derive(Debug, Clone)]
enum ProdComponent {
    LhsName(ProdName),
    LanguageConstruct(ArithLanguage),
}

#[derive(Debug, Clone)]
struct Grammar {
    productions: Vec<Production>,
}

#[derive(Default, Debug, Clone)]
struct ObsEquiv {
    pts: IOPairs,
}

#[derive(Default, Debug, Clone, PartialEq, PartialOrd, Ord, Eq, Hash)]
struct ExprAstSize {
    ast_size: usize,
    prod_name: String,
}
// struct OperatorCostFn;

// impl CostFunction<ArithLanguage> for OperatorCostFn {
//     type Cost = f64;

//     fn cost<C>(&mut self, enode: &ArithLanguage, mut costs: C) -> Self::Cost
//     where
//         C: FnMut(Id) -> Self::Cost,
//     {
//         let op_cost = match enode {
//             ArithLanguage::Num(_) => -1.0,
//             ArithLanguage::Var(_) => -1.0,
//             ArithLanguage::Add(_) | ArithLanguage::Sub(_) => -2.0,
//             ArithLanguage::Mul(_) => -3.0,
//             ArithLanguage::Neg(_) => -1.5,
//         };

//         let children_cost: f64 = enode.fold(0.0, |sum, id| sum + costs(id));
//         op_cost + children_cost
//     }
// }

// struct VarietyCostFn<'a> {
//     egraph: &'a EGraph<ArithLanguage, ObsEquiv>,
// }

// impl<'a> CostFunction<ArithLanguage> for VarietyCostFn<'a>  {
//     type Cost = f64;

//     fn cost<C>(&mut self, enode: &ArithLanguage, mut costs: C) -> Self::Cost
//     where
//         C: FnMut(Id) -> Self::Cost,
//     {
//         let mut variables:HashSet<String> = HashSet::new();

//         let variety_cost = match enode {
//             ArithLanguage::Num(_) => -1.0,
//             ArithLanguage::Var(name) => {
//                 variables.insert(name.clone());
//                 -2.0
//             },
//             _ => -0.0,
//         };

//         let children_cost: f64 = enode.fold(0.0, |sum, id| {
//             let child_cost = costs(id);
//             variables.extend(self.get_variables(id));
//             sum + child_cost
//         });

//         // variety_cost + children_cost +
//         -(variables.len() as f64)
//     }
// }

// impl VarietyCostFn<'_> {
//     fn get_variables(&self, id: Id) -> HashSet<String> {
//         let mut variables = HashSet::new();
//         if let Some(var_name) = self.egraph[id].nodes.iter().find_map(|n| match n {
//             ArithLanguage::Var(name) => Some(name.clone()),
//             _ => None,
//         }) {
//             variables.insert(var_name);
//         }
//         variables
//     }
// }
// TODO:
// 1. IOPairs -> IOPair
// 2. Implement methods for IOPair
//    (1) fn eval(&self, egraph: &EGraph<ArithLanguage, ObsEquiv>, expr: &RecExpr<ArithLanguage>) -> i64
// 3. Remove cache for storing all expressions, only store equivs, when the egraph eclasses split, new exprs will be added to the cache
impl Analysis<ArithLanguage> for ObsEquiv {
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

    fn make(egraph: &EGraph<ArithLanguage, Self>, enode: &ArithLanguage) -> Self::Data {
        let pts: &Vec<(HashMap<String, i32>, i32)> = &egraph.analysis.pts;
        let sem = enode.semantics();
        let o = |i: &Id| &egraph[*i].data; // output
        match enode {
            // Constants
            ArithLanguage::Num(n) => pts.iter().map(|(input, output)| (input.clone(), *n)).collect(),
            // Varaibles
            ArithLanguage::Var(name) => pts.iter().map(|(input, output)| (input.clone(), sem(&[(input.clone(), *input.get(name).unwrap())]))).collect(),
            // Unary operators
            ArithLanguage::Neg([id]) => o(id)
                .iter()
                .zip(pts)
                .map(|((input, output), _)| (input.clone(), sem(&[(input.clone(), *output)])))
                .collect(),
            // Binary operators
            ArithLanguage::Add([a, b]) | ArithLanguage::Sub([a, b]) | ArithLanguage::Mul([a, b]) => o(a)
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
    fn modify(egraph: &mut EGraph<ArithLanguage, Self>, id: Id) {
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

impl Analysis<ArithLanguage> for ExprAstSize {
    type Data = ExprAstSize;

    fn merge(&mut self, to: &mut Self::Data, from: Self::Data) -> DidMerge {
        let merged = to.ast_size == from.ast_size;
        if merged {
            *to = from;
        }
        DidMerge(merged, false)
    }

    fn make(egraph: &EGraph<ArithLanguage, Self>, enode: &ArithLanguage) -> Self::Data {
        // println!("<ExprAstSize::make> enode: {:?}", enode);
        if let Some(id) = egraph.lookup(enode.clone()) {
            let ExprAstSize { ast_size, prod_name} = egraph[id].data.clone();
            ExprAstSize { ast_size, prod_name }
        } else {
            let ast_size = match enode {
                ArithLanguage::Num(_) | ArithLanguage::Var(_) => 1,
                ArithLanguage::Neg([id]) => egraph[*id].data.ast_size + 1,
                ArithLanguage::Add([a, b]) | ArithLanguage::Sub([a, b]) | ArithLanguage::Mul([a, b]) => {
                    egraph[*a].data.ast_size + egraph[*b].data.ast_size + 1
                },
            };
            let prod_name = "PLACE_HOLDER".to_string();
            ExprAstSize { ast_size, prod_name }
        }
    }
    fn modify(egraph: &mut EGraph<ArithLanguage, Self>, id: Id) {
        // We don't need to do anything here because the data consistency is guaranteed by the `enumerator.rebuild()`.
    }
}

// TODO: Can we use e-matching to obtain certain size of expressions?
// I think we can use AstSize somewhere.
// Can we connect data with certain e-node, rather than e-class? In this way, we can keep track of expr size.
struct Enumerator<'a> {
    grammar: &'a Grammar,
    egraph: EGraph<ArithLanguage, ObsEquiv>,
    // cache: HashMap<(ProdName, usize), HashSet<RecExpr<ArithLanguage>>>,
    ecache: EGraph<ArithLanguage, ExprAstSize>,
    current_size: usize,
}

impl<'a> Enumerator<'a> {
    fn new(grammar: &'a Grammar) -> Self {
        Enumerator {
            grammar,
            egraph: EGraph::new(ObsEquiv::default()).with_explanations_enabled(),
            // cache: HashMap::new(),
            ecache: EGraph::new(ExprAstSize::default()).with_explanations_enabled(),
            current_size: 0,
        }
    }

    fn rebuild(&mut self, pts: &IOPairs) {
        // TODO: construct another egraph using all the possible expr in one egraph? (Non-incremental)
        // TODO: split eclasses by certain standard, enodes eval to the same value will be in the new same eclass, one is retained as orignal, the other are newly created.
        let start = std::time::Instant::now();
        let mut new_egraph = EGraph::new(ObsEquiv { pts: pts.clone() }).with_explanations_enabled();
        for eclass in self.egraph.classes() {  // level = 1
            let mut exprs = HashSet::new();
            // collect all exprs in the eclass
            for node in &eclass.nodes {
                let expr = node.join_recexprs(|id| self.egraph.id_to_expr(id));
                exprs.insert(expr);
            }
            // TODO: cannot build new cache because of the `prod_name` required in the key
            // add all exprs to the new egraph
            for expr in exprs {
                new_egraph.add_expr(&expr);
            }
        }
        // let exprs_set = collect_all_equivs(&self.egraph); // level = 10
        // for expr in exprs_set {
        //     new_egraph.add_expr(&expr);
        // }
        self.merge_equivs();
        new_egraph.rebuild();
        self.egraph = new_egraph;
        println!("<Enumerator::rebuild> Time elapsed: {:?}", start.elapsed().as_secs_f64());
    }

    fn get_all_exprs_from_ecache(&self, key: ExprAstSize) -> HashSet<RecExpr<ArithLanguage>> {
        // println!("<Enumerator::get_all_exprs_from_ecache> key: {:?}", key);
        let exprs = collect_all_equivs2(&self.ecache, key);
        exprs
    }

    fn get_sat_exprs_from_egraph(&self, pts: &IOPairs) -> Vec<RecExpr<ArithLanguage>> {
        let mut result: Vec<RecExpr<ArithLanguage>> = vec![];
        for eclass in self.egraph.classes() {
            if eclass.data.iter().all(|(i, o)| pts.contains(&(i.clone(), *o))) {
                result.push(self.egraph.id_to_expr(eclass.id));
            }
        }
        result
    }
    
    fn exists_in_egraph(&self, expr: &RecExpr<ArithLanguage>) -> bool {
        let id = self.egraph.lookup_expr(expr);
        id.is_some()
    }
    fn exists_in_cache(&self, key: &(ProdName, usize)) -> bool {
        // self.cache.contains_key(key)
        false
    }
    fn collect_equivs(&self) -> HashMap<Vec<(Vec<(String, i32)>, i32)>, HashSet<Id>> {
        let mut equivs: HashMap<Vec<(Vec<(String, i32)>, i32)>, HashSet<Id>> = HashMap::new();
        for eclass in self.egraph.classes() {
            let data: Vec<(Vec<(String, i32)>, i32)> = eclass.data.iter().map(|(i, o)| {
                let mut vec: Vec<(String, i32)> = i.clone().into_iter().collect();
                vec.sort();
                (vec, *o)
            }).collect::<Vec<_>>();
            let key = data;
            equivs.entry(key).or_insert(HashSet::new()).insert(eclass.id);
        }
        equivs
    }

    fn collect_equivs_ecache(&self) -> HashMap<ExprAstSize, HashSet<Id>> {
        let mut equivs = HashMap::new();
        for eclass in self.ecache.classes() {
            let data = eclass.data.clone();
            let key = data;
            equivs.entry(key).or_insert(HashSet::new()).insert(eclass.id);
        }
        equivs
    }
    fn merge_equivs(&mut self) {
        let start = std::time::Instant::now();
        let equivs = self.collect_equivs();
        // merge those equivs
        // println!("<Enumerator::merge_equivs> equivs.keys().len(): {}", equivs.keys().len());
        for (k, v) in equivs {
            // println!("<Enumerator::merge_equivs> data: {:?}", k.iter().map(|t| t.1).collect::<Vec<_>>());
            let mut iter = v.into_iter();
            if let Some(first) = iter.next() {
                // println!("<Enumerator::merge_equivs> first id: {}", first);
                // println!("<Enumerator::merge_equivs> eclass({})(nodes.len = {}): {:?}", first, self.egraph[first].nodes.len(), self.egraph[first]);
                for id in iter {
                    // println!("<Enumerator::merge_equivs> id: {}", id);
                    // println!("<Enumerator::merge_equivs> eclass({})(nodes.len = {}): {:?}", id, self.egraph[id].nodes.len(), self.egraph[id]);
                    self.egraph.union(first, id);
                    // println!("<Enumerator::merge_equivs> eclass after union({})(nodes.len = {}): {:?}", first, self.egraph[first].nodes.len(), self.egraph[first]);
                }
            }
        }
        println!("<Enumerator::merge_equivs> Time elapsed: {:?}", start.elapsed().as_secs_f64());
    }

    fn merge_equivs_ecache(&mut self) {
        let start = std::time::Instant::now();
        let equivs = self.collect_equivs_ecache();
        // merge those equivs
        // println!("<Enumerator::merge_equivs> equivs.keys().len(): {}", equivs.keys().len());
        for (k, v) in equivs {
            // println!("<Enumerator::merge_equivs> data: {:?}", k.iter().map(|t| t.1).collect::<Vec<_>>());
            let mut iter = v.into_iter();
            if let Some(first) = iter.next() {
                // println!("<Enumerator::merge_equivs> first id: {}", first);
                // println!("<Enumerator::merge_equivs> eclass({})(nodes.len = {}): {:?}", first, self.egraph[first].nodes.len(), self.egraph[first]);
                for id in iter {
                    // println!("<Enumerator::merge_equivs> id: {}", id);
                    // println!("<Enumerator::merge_equivs> eclass({})(nodes.len = {}): {:?}", id, self.egraph[id].nodes.len(), self.egraph[id]);
                    self.ecache.union(first, id);
                    // println!("<Enumerator::merge_equivs> eclass after union({})(nodes.len = {}): {:?}", first, self.egraph[first].nodes.len(), self.egraph[first]);
                }
            }
        }
    }

    /* async */ fn grow(&mut self) {
        let start = std::time::Instant::now();
        let size = self.current_size + 1;
        // Base case: directly add numbers and variables for size 1
        if size == 1 {
            for prod in &self.grammar.productions {
                for component in &prod.rhs {
                    if let ProdComponent::LanguageConstruct(lang_construct) = component {
                        println!("<Enumerator::grow> prod(size={}): ({}, {}) => {:?}", size, prod.lhs, prod.lhs_type, lang_construct);
                        // let mut new_expressions: HashMap<(String, usize), HashSet<RecExpr<ArithLanguage>>> = HashMap::new();
                        match lang_construct {
                            ArithLanguage::Num(_) | ArithLanguage::Var(_) => {
                                // Logically, this is not sufficient, because the eclass may contains some equivalent expressions
                                // but this is the leave node, so it's fine.

                                // let (id, expr) = self.egraph.add_with_recexpr_return(lang_construct.clone());
                                let id = self.egraph.add(lang_construct.clone());
                                let expr = self.egraph.id_to_expr(id);

                                let id = self.ecache.add_expr(&expr);
                                self.ecache.set_analysis_data(id, ExprAstSize { ast_size: 1, prod_name: prod.lhs.clone() });
                                self.merge_equivs_ecache();

                                // let mut old_expr_iter = [lang_construct.clone()].into_iter();
                                // let mut old_expr_iter = [left_expr.clone(), right_expr.clone()].into_iter();
                                // let expr =  lang_construct.join_recexprs(|_| old_expr_iter.next().unwrap());
                                // println!("<Enumerator::grow> lc: {:?}, expr: {:?}", lang_construct, expr.pretty(100));

                                // new_expressions.entry((prod.lhs.clone(), AstSize.cost_rec(&expr)))
                                //                .or_insert_with(HashSet::new)
                                //                .insert(expr);
                            },
                            _ => {}
                        }
                        // self.merge_equivs();
                        // self.egraph.rebuild();
                        // for (key, exprs) in new_expressions {
                        //     // for expr in &exprs {
                        //     //     self.egraph.add_expr(&expr);
                        //     // }
                        //     // let exprs = exprs.into_iter().filter(|expr| !self.exists_in_egraph(expr) || !self.exists_in_cache(&key)).collect::<Vec<_>>();
                        //     self.cache.entry(key).or_insert_with(HashSet::new).extend(exprs);
                        // }
                    }
                }
            }
        } else {
            // Composite expressions
            for prod in &self.grammar.productions {
                for component in &prod.rhs {
                    if let ProdComponent::LanguageConstruct(lang_construct) = component {
                        println!("<Enumerator::grow> prod(size={}): ({}, {}) => {:?}", size, prod.lhs, prod.lhs_type, lang_construct);
                        // let mut new_expressions: HashMap<(String, usize), HashSet<RecExpr<ArithLanguage>>> = HashMap::new();
                        match lang_construct {
                            ArithLanguage::Add(_) | ArithLanguage::Sub(_) | ArithLanguage::Mul(_) => {
                                let num_nonterminals = lang_construct.children().len();
                                for left_size in 1..(size-1) {
                                    let right_size = (size-1) - left_size;
                                    // println!("<Enumerator::grow> left_size: {}, right_size: {}", left_size, right_size);

                                    // let mut left_expr_parts = self.cache
                                    //     .get(&(prod.lhs.clone(), left_size)).cloned().unwrap_or_default()
                                    //     .into_iter().collect::<Vec<_>>();
                                    let left_expr_parts = self.get_all_exprs_from_ecache(ExprAstSize { ast_size: left_size, prod_name: prod.lhs.clone() })
                                        .into_iter().collect::<Vec<_>>();
                                    // println!("<Enumerator::grow> left_expr_parts.len(): {}", left_expr_parts.len());
                                    // for entry in &new_expressions {
                                    //     let ((prod_name, expr_size), exprs) = entry;
                                    //     if prod_name == &prod.lhs && expr_size == &left_size {
                                    //         left_expr_parts.extend(exprs.iter().cloned().collect::<Vec<_>>());
                                    //     }
                                    // }

                                    // let mut right_expr_parts = self.cache
                                    //     .get(&(prod.lhs.clone(), right_size)).cloned().unwrap_or_default()
                                    //     .into_iter().collect::<Vec<_>>();
                                    let right_expr_parts = self.get_all_exprs_from_ecache(ExprAstSize { ast_size: right_size, prod_name: prod.lhs.clone() })
                                        .into_iter().collect::<Vec<_>>();
                                    // println!("<Enumerator::grow> right_expr_parts.len(): {}", right_expr_parts.len());
                                    // for entry in &new_expressions {
                                    //     let ((prod_name, expr_size), exprs) = entry;
                                    //     if prod_name == &prod.lhs && expr_size == &right_size {
                                    //         right_expr_parts.extend(exprs.iter().cloned().collect::<Vec<_>>());
                                    //     }
                                    // }

                                    for left_expr in &left_expr_parts {
                                        for right_expr in &right_expr_parts {
                                            // println!("<Enumerator::grow> op: {:?}, left_expr: {:?}, right_expr: {:?}", pretty_op(lang_construct), left_expr.pretty(100), right_expr.pretty(100));
                                            let left_id = self.egraph.add_expr(left_expr);
                                            let right_id = self.egraph.add_expr(right_expr);

                                            let id = match lang_construct {
                                                ArithLanguage::Add(_) => {
                                                    self.egraph.add(ArithLanguage::Add([left_id, right_id]))
                                                },
                                                ArithLanguage::Sub(_) => {
                                                    self.egraph.add(ArithLanguage::Sub([left_id, right_id]))
                                                },
                                                ArithLanguage::Mul(_) => {
                                                    self.egraph.add(ArithLanguage::Mul([left_id, right_id]))
                                                },
                                                _ => unreachable!(),
                                            };

                                            let mut old_expr_iter = [left_expr.clone(), right_expr.clone()].into_iter();
                                            let expr = lang_construct.join_recexprs(|id| old_expr_iter.next().unwrap());
                                            // println!("<Enumerator::grow> lc: {:?}, expr: {:?}", lang_construct, expr.pretty(100));

                                            let id = self.ecache.add_expr(&expr);
                                            self.ecache.set_analysis_data(id, ExprAstSize { ast_size: AstSize.cost_rec(&expr), prod_name: prod.lhs.clone() });
                                            self.merge_equivs_ecache();
                                            // new_expressions.entry((prod.lhs.clone(), AstSize.cost_rec(&expr)))
                                            //                .or_insert_with(HashSet::new)
                                            //                .insert(expr);
                                        }
                                    }
                                }
                            },
                            ArithLanguage::Neg(_) => {
                                let right_size = size - 1;
                                // println!("<Enumerator::grow> right_size: {}", right_size);

                                // let mut right_expr_parts = self.cache
                                //     .get(&(prod.lhs.clone(), right_size)).cloned().unwrap_or_default()
                                //     .into_iter().collect::<Vec<_>>();
                                let right_expr_parts = self.get_all_exprs_from_ecache(ExprAstSize { ast_size: right_size, prod_name: prod.lhs.clone() })
                                    .into_iter().collect::<Vec<_>>();
                                // println!("<Enumerator::grow> right_expr_parts.len(): {}", right_expr_parts.len());
                                // for entry in &new_expressions {
                                //     let ((prod_name, expr_size), exprs) = entry;
                                //     if prod_name == &prod.lhs && expr_size == &right_size {
                                //         right_expr_parts.extend(exprs.iter().cloned().collect::<Vec<_>>());
                                //     }
                                // } // This is necessary, otherwise, some exprs will be never taken into consideration due to the laziness
                                for right_expr in &right_expr_parts {
                                    let right_id = self.egraph.add_expr(right_expr);
                                    // let (id, expr) = self.egraph.add_with_recexpr_return(ArithLanguage::Neg([right_id]));
                                    let id = self.egraph.add(ArithLanguage::Neg([right_id]));
                                    // let expr = self.egraph.id_to_expr(id);
                                    let expr = lang_construct.join_recexprs(|_| right_expr.clone());
                                    // println!("<Enumerator::grow> lc: {:?}, expr: {:?}", lang_construct, expr.pretty(100));

                                    let id = self.ecache.add_expr(&expr);
                                    self.ecache.set_analysis_data(id, ExprAstSize { ast_size: AstSize.cost_rec(&expr), prod_name: prod.lhs.clone() });
                                    self.merge_equivs_ecache();

                                    // new_expressions.entry((prod.lhs.clone(), AstSize.cost_rec(&expr)))
                                    //                .or_insert_with(HashSet::new)
                                    //                .insert(expr);
                                }
                            },
                            _ => {}
                        }
                        // println!("<Enumerator::grow> new_expressions<({}, {})>.len(): {}", prod.lhs, size, new_expressions.values().flatten().count());
                        // self.merge_equivs();
                        // self.egraph.rebuild();
                        // for (key, exprs) in new_expressions {
                        //     // for expr in &exprs {
                        //     //     self.egraph.add_expr(&expr);
                        //     // }
                        //     // let exprs = exprs.into_iter().filter(|expr| !self.exists_in_egraph(expr) || !self.exists_in_cache(&key)).collect::<Vec<_>>();
                        //     self.cache.entry(key).or_insert_with(HashSet::new).extend(exprs);
                        // }
                        // task::yield_now().await;
                    }
                }
            }
        }
        // // Update cache with new expressions
        // for (key, exprs) in new_expressions {
        //     for expr in &exprs {
        //         self.egraph.add_expr(&expr);
        //     }
        //     self.cache.entry(key).or_insert_with(HashSet::new).extend(exprs);
        // }
        self.merge_equivs();
        // self.merge_equivs_ecache();
        self.egraph.rebuild();
        self.ecache.rebuild();

        // println!("{}", pretty_cache(&self.cache, 2));
        // println!("{}", pretty_egraph(&self.egraph, 2));
        // println!("<Enumerator::grow> Pretty ecache: {}", pretty_ecache(&self.ecache, 2));
        self.current_size = size;
        println!("<Enumerator::grow> Time elapsed: {:?}", start.elapsed().as_secs_f64());
    }


    /* async */fn enumerate(&mut self, size: usize, pts: &IOPairs) -> Vec<RecExpr<ArithLanguage>> {
        if self.egraph.analysis.pts.len() != pts.len() {
            // println!("<Enumerator::enumerate> egraph.analysis.pts: {:?}", self.egraph.analysis.pts);
            self.rebuild(pts);
            // println!("<Enumerator::enumerate> egraph.analysis.pts: {:?}", self.egraph.analysis.pts);
        }
        println!("<Enumerator::enumerate> current size: {}, pts: {:?}", self.current_size, pts);
        let mut result: Vec<RecExpr<ArithLanguage>> = vec![];
        while self.current_size <= size {
            if self.current_size < size {
                println!("<Enumerator::enumerate> Growing to size: {}", self.current_size + 1);
                self.grow()/*.await*/;
                println!("<Enumerator::enumerate> Finished growing to size: {}", self.current_size);
                println!("<Enumerator::enumerate> ECache size: {}", self.ecache.total_size());
                println!("<Enumerator::enumerate> EGraph size: {}", self.egraph.total_size());
                println!("<Enumerator::enumerate> EGraph nodes: {}", self.egraph.total_number_of_nodes());
                println!("<Enumerator::enumerate> EGraph classes: {}", self.egraph.number_of_classes());
            }
            let start_nonterminal = &self.grammar.productions.first().unwrap().lhs;
            let ecache_max_size = self.ecache.classes().map(|eclass| eclass.data.ast_size).max().unwrap();
            let exprs = self.get_sat_exprs_from_egraph(pts);
            result.clear();
            for expr in exprs {
                if self.satisfies_pts(&expr, pts) {
                    // println!("<Enumerator::enumerate> expr: {:?} satisfies pts: {:?}", expr.pretty(100), pts);
                    result.push(expr.clone());
                }
            }
            if !result.is_empty() {
                println!("<Enumerator::enumerate> Found {} expressions that satisfy pts: {:?}", result.len(), pts);
                break;
            } else {
                println!("<Enumerator::enumerate> No expressions satisfy pts: {:?}", pts);
            }
            for expr in &result {
                println!("<Enumerator::enumerate> expr: {:?}", expr.pretty(100));
            }
            if self.current_size == size {
                break;
            }
        }
        result
    }


    fn satisfies_pts(&self, expr: &RecExpr<ArithLanguage>, pts: &IOPairs) -> bool {
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

struct EggSolver {
    grammar: Grammar,
}

impl EggSolver {
    fn new(grammar: Grammar) -> Self {
        EggSolver { grammar }
    }

    /*async */fn synthesize(&self, max_size: usize, pts_all: &IOPairs) -> Vec<RecExpr<ArithLanguage>> {
        let mut enumerator = Enumerator::new(&self.grammar);
        let mut pts = vec![];
        let start = std::time::Instant::now();
        // println!("<EggSolver::synthesize> Start time: {:?}", start);
        while enumerator.current_size < max_size {
            let mut exprs = enumerator.enumerate(max_size, &pts)/*.await */;
            let mut equiv_exprs = HashSet::new();
            for expr in &exprs {
                if equiv_exprs.contains(expr) {
                    println!("<EggSolver::synthesize> Expr: {:?} already in equiv_exprs", expr.pretty(100));
                    continue;
                }
                // use equiv_exprs to get all equivalent expressions
                let equivs = get_equiv_exprs(&enumerator.egraph, expr);
                println!("<EggSolver::synthesize> Expr: {:?} has {} equivs", expr.pretty(100), equivs.len());
                // let equivs_more = equivs.iter().fold(HashSet::new(), |mut acc, equiv| {
                //     acc.extend(get_equiv_exprs(&enumerator.egraph, equiv));
                //     acc
                // });
                // for equiv in &equivs {
                //     // println!("<EggSolver::synthesize> equiv: {}", equiv.pretty(100));
                // }
                equiv_exprs.extend(equivs);
                // equiv_exprs.extend(equivs_more);
            }
            exprs = equiv_exprs.into_iter().collect::<Vec<_>>();
            println!("<EggSolver::synthesize> Total synthesized exprs: {}", exprs.len());
            for expr in &exprs {
                println!("<EggSolver::synthesize> Equiv expr: {}", expr.pretty(100));
            }
            let mut exprs_sat = vec![];
            let mut cex_unsat = None;
            // let var_extractor = Extractor::new(&enumerator.egraph, VarietyCostFn { egraph: &enumerator.egraph });
            // let var_extractor = Extractor::new(&enumerator.egraph, AstSize);
            for expr in &exprs {
                // let (var_cost, var_best_expr) = var_extractor.find_best(enumerator.egraph.lookup_expr(expr).unwrap());
                // let expr = var_best_expr /*self.egraph.id_to_expr(id)*/;
                if let Some(cex) = self.verify(&expr, pts_all){
                    if cex_unsat.is_none() {
                        cex_unsat = Some(cex);
                    }
                } else {
                    exprs_sat.push(expr.clone());
                }
            }
            if !exprs_sat.is_empty() {
                // target found
                // println!("{}", pretty_cache(&enumerator.cache, 2));
                // println!("{}", pretty_egraph(&enumerator.egraph, 2));
                // println!("Time elapsed: {:?}", start.elapsed().as_secs_f64());
                return exprs_sat;
            }
            if !exprs.is_empty() {
                // cannot find target within current size
                pts.push(cex_unsat.unwrap());
            }
        }
        vec![]
    }

    fn verify(&self, expr: &RecExpr<ArithLanguage>, pts_all: &IOPairs) -> Option<(HashMap<String, i32>, i32)> {
        // println!("<EggSolver::verify> expr: {:?} = {:?}", expr.pretty(100), expr);
        for (inputs, expected_output) in pts_all {
            // println!("<EggSolver::verify> inputs: {:?}, expected_output: {:?}", inputs, expected_output);
            let mut egraph = EGraph::new(ObsEquiv { pts: vec![(inputs.clone(), *expected_output)] }).with_explanations_enabled();
            let expr = expr.clone();
            // println!("<EggSolver::verify> pretty_egraph: {}", pretty_egraph(&egraph, 2));
            let id = egraph.add_expr(&expr);
            // println!("<EggSolver::verify> expr: {:?} = {:?}", expr.pretty(100), expr);
            // println!("{}", pretty_egraph(&egraph, 2));
            egraph.rebuild();
            // println!("<EggSolver::verify> egraph[id].data: {:?}", egraph[id].data);
            let actual_output = egraph[id].data[0].1;
            // println!("<EggSolver::verify> actual_output: {:?}", actual_output);
            if actual_output != *expected_output {
                return Some((inputs.clone(), *expected_output));
            }
        }
        None
    }
}

fn pretty_op(op: &ArithLanguage, ends: bool) -> String {
    let op_str = match op {
        ArithLanguage::Num(n) => format!("{}", n),
        ArithLanguage::Var(name) => name.clone(),
        ArithLanguage::Add(_) => "+".to_string(),
        ArithLanguage::Sub(_) => "-".to_string(),
        ArithLanguage::Mul(_) => "*".to_string(),
        ArithLanguage::Neg(_) => "-".to_string(),
    };
    if ends && !matches!(op, ArithLanguage::Num(_) | ArithLanguage::Var(_)){
        format!("{} ", op_str)
    } else {
        format!("{}", op_str)
    }
}

fn pretty_cache(cache: &HashMap<(ProdName, usize), HashSet<RecExpr<ArithLanguage>>>, starting_space: usize)  -> String {
    let mut result = String::new();
    result.push_str("Cache:\n");
    let cache = cache.into_iter().sorted_by(|(k1, _), (k2, _)| Ord::cmp(k1, k2));
    for (key, exprs) in cache {
        let exprs = exprs.into_iter().sorted_by(|e1, e2| Ord::cmp(&AstSize.cost_rec(e1), &AstSize.cost_rec(e2)));
        for expr in exprs {
            // println!("[{}, {}]: {}", &key.0, &key.1, expr.pretty(100));
            // println!("{}[{}, {}]: {}", " ".repeat(starting_space), &key.0, &key.1, expr.pretty(100));
            result.push_str(&format!("{}[{}, {}]: {}\n", " ".repeat(starting_space), &key.0, &key.1, expr.pretty(100)));
        }
    }
    result.trim().to_string()
}

fn pretty_egraph(egraph: &EGraph<ArithLanguage, ObsEquiv>, starting_space: usize) -> String {
    let mut result = String::new();
    result.push_str("EGraph:\n");
    for eclass in egraph.classes() {
        let expr = egraph.id_to_expr(eclass.id);
        result.push_str(&format!("{}<{}>: [{}] -> {:?}\n", " ".repeat(starting_space), eclass.id, expr.pretty(100), egraph[eclass.id].data.iter().map(| (inputs, output) | output).collect::<Vec<_>>()));
        // Data
        result.push_str(&format!("{}<data>:\n", " ".repeat(starting_space * 2)));
        for (inputs, output) in &egraph[eclass.id].data {
            result.push_str(&format!("{}[{}] -> {}\n", " ".repeat(starting_space * 3), inputs.iter().sorted().map(|(k, v)| format!("{}: {}", k, v)).collect::<Vec<_>>().join(", "), output));
        }
        // Nodes
        result.push_str(&format!("{}<nodes>:\n", " ".repeat(starting_space * 2)));
        let align_space = eclass.nodes.iter().map(|node| format!("{:?}", node)).map(|s| s.len()).max().unwrap_or_default();
        for node in &eclass.nodes {
            let node_str = format!("{:?}", node);
            result.push_str(&format!("{}{}:", " ".repeat(starting_space * 3), node_str));
            result.push_str(" ".repeat(align_space + 1 - node_str.len()).as_str());
            let child_expr = node.join_recexprs(|id| egraph.id_to_expr(id));
            result.push_str(&format!("{}\n", child_expr.pretty(100)));
        }
        // // expr ids
        // result.push_str(&format!("{}exprs:\n", " ".repeat(starting_space * 2)));
        // let ids = egraph.lookup_expr_ids(&expr).unwrap_or_default();
        // for expr_id in ids {
        //     result.push_str(&format!("{}{}\n", " ".repeat(starting_space * 2), egraph.id_to_expr(expr_id).pretty(100)));
        // }
    }
    result.trim().to_string()
}

fn pretty_ecache(egraph: &EGraph<ArithLanguage, ExprAstSize>, starting_space: usize) -> String {
    let mut result = String::new();
    result.push_str("EGraph Cache:\n");
    for eclass in egraph.classes() {
        let expr = egraph.id_to_expr(eclass.id);
        let ExprAstSize { ast_size, prod_name } = &egraph[eclass.id].data;
        result.push_str(&format!("{}<{}>: ({}, {}) => [{}]\n", " ".repeat(starting_space), eclass.id, prod_name, ast_size, expr.pretty(100)));
        
        // Nodes
        result.push_str(&format!("{}<nodes>:\n", " ".repeat(starting_space * 2)));
        let align_space = eclass.nodes.iter().map(|node| format!("{:?}", node)).map(|s| s.len()).max().unwrap_or_default();
        for node in &eclass.nodes {
            let node_str = format!("{:?}", node);
            result.push_str(&format!("{}{}:", " ".repeat(starting_space * 3), node_str));
            result.push_str(" ".repeat(align_space + 1 - node_str.len()).as_str());
            let child_expr = node.join_recexprs(|id| egraph.id_to_expr(id));
            result.push_str(&format!("{}\n", child_expr.pretty(100)));
        }
    }
    result.trim().to_string()
}

fn _build_recexpr<F>(root: &ArithLanguage, mut get_node: F) -> Option<RecExpr<ArithLanguage>>
where F: FnMut(Id) -> Option<ArithLanguage>
{
    let mut set = IndexSet::<ArithLanguage>::default();
    let mut ids = HashMap::<Id, Id>::default();
    let mut todo = root.children().to_vec();
    let mut visited = HashSet::<Id>::default();
    println!("<_build_recexpr> root: {:?}", root);
    println!("<_build_recexpr> todo init: {:?}", todo);

    while let Some(id) = todo.pop() {
        if visited.contains(&id) {
            continue;
        }
        if let Some(node) = get_node(id) {
            visited.insert(id);
            for child in node.children() {
                if !todo.contains(child) {
                    todo.push(*child);
                }
            }
        }
    }
    println!("<_build_recexpr> todo: {:?}", todo);
    println!("<_build_recexpr> visited: {:?}", visited);

    for id in visited {
        let node = get_node(id).unwrap();
        let new_id = set.insert_full(node).0;
        ids.insert(id, Id::from(new_id));
        println!("<_build_recexpr> ids: {:?}", ids);
    }

    // while let Some(id) = todo.last().copied() {
    //     cnt += 1;
    //     println!("<_build_recexpr> loop cnt: {}, id: {:?}", cnt, id);
    //     if ids.contains_key(&id) {
    //         todo.pop();
    //         continue;
    //     }

    //     let node = get_node(id);
    //     println!("<_build_recexpr> node: {:?}", node);

    //     // check to see if we can do this node yet
    //     let mut ids_has_all_children = true;
    //     for child in node.children() {
    //         if !ids.contains_key(child) {
    //             ids_has_all_children = false;
    //             if !todo.contains(child) {
    //                 println!("<_build_recexpr> todo push: {:?} (todo.len: {})", child, todo.len());
    //                 todo.push(*child);
    //             }
    //         }
    //     }

    //     println!("<_build_recexpr> ids_has_all_children: {}", ids_has_all_children);
    //     println!("<_build_recexpr> todo: {:?}", todo);
    //     println!("<_build_recexpr> ids: {:?}", ids);

    //     // all children are processed, so we can lookup this node safely
    //     if ids_has_all_children {
    //         let node = node.map_children(|id| ids[&id]);
    //         let new_id = set.insert_full(node).0;
    //         ids.insert(id, Id::from(new_id));
    //         todo.pop();
    //     }
    //     println!("<_build_recexpr> ids: {:?}", ids);
    //     println!("<_build_recexpr> todo: {:?}", todo);
    // }
    // finally, add the root node and create the expression
    let mut nodes: Vec<ArithLanguage> = set.into_iter().collect();
    nodes.push(root.clone().map_children(|id| ids[&id]));
    Some(RecExpr::from(nodes))
}

fn get_equiv_exprs(egraph: &EGraph<ArithLanguage, ObsEquiv>, expr: &RecExpr<ArithLanguage>) -> Vec<RecExpr<ArithLanguage>> {
    // println!("<get_equiv_exprs> representative expr: {}", expr.pretty(100));
    let mut exprs = vec![];
    if let Some(id) = egraph.lookup_expr(expr) {
        // println!("<get_equiv_exprs> id: {:?}", id);
        let eclass = &egraph[id];
        // println!("<get_equiv_exprs> eclass: {:?}", eclass);
        for node in &eclass.nodes {
            // println!("<get_equiv_exprs> node: {:?}", node); // type: &ArithLanguage
            // TODO: not only build from the representative expr, but also from other nodes in the eclass
            let new_expr = node.join_recexprs(|id| egraph.id_to_expr(id));
            // println!("<get_equiv_exprs> new_expr: {}", new_expr.pretty(100));
            exprs.push(new_expr);
        }
    }
    exprs
}

fn collect_all_equivs(egraph: &EGraph<ArithLanguage, ObsEquiv>) -> HashSet<RecExpr<ArithLanguage>> {
    let mut exprs_set = HashSet::new();
    for eclass in egraph.classes() {
        collect_all_equivs_rec(egraph, eclass.id, &mut exprs_set, 0);
    }
    exprs_set
}

fn collect_all_equivs_rec(egraph: &EGraph<ArithLanguage, ObsEquiv>, root_id: Id, exprs_set: &mut HashSet<RecExpr<ArithLanguage>>, level: usize) {
    if level > 10 {
        return;
    }
    let eclass = &egraph[root_id];
    for node in &eclass.nodes {
        let new_expr = node.join_recexprs(|id| egraph.id_to_expr(id));
        exprs_set.insert(new_expr.clone());
        for id in node.children() {
            collect_all_equivs_rec(egraph, *id, exprs_set, level + 1);
        }
    }
}

fn collect_all_equivs2(egraph: &EGraph<ArithLanguage, ExprAstSize>, key: ExprAstSize) -> HashSet<RecExpr<ArithLanguage>> {
    let mut exprs_set = HashSet::new();
    for eclass in egraph.classes() {
        let ExprAstSize { ast_size, prod_name } = &egraph[eclass.id].data;
        if ast_size != &key.ast_size || prod_name != &key.prod_name {
            continue;
        }
        // println!("<collect_all_equivs2> pretty_ecache(egraph, 2): {}", pretty_ecache(egraph, 2));
        // println!("<collect_all_equivs2> ast_size: {:?}", ast_size);
        collect_all_equivs_rec2(egraph, eclass.id, &mut exprs_set);
        // println!("<collect_all_equivs2> exprs_set.len(): {}", exprs_set.len());
    }
    exprs_set
}

fn collect_all_equivs_rec2(egraph: &EGraph<ArithLanguage, ExprAstSize>, root_id: Id, exprs_set: &mut HashSet<RecExpr<ArithLanguage>>) {
    let eclass = &egraph[root_id];
    let mut expr_count = 0;
    for (i, node) in enumerate(&eclass.nodes) {
        let children = node.children();
        // println!("<collect_all_equivs_rec2> children: {:?}", children);
        if children.is_empty() {
            // leaf node
            // println!("<collect_all_equivs_rec2> leaf node: {:?}", node);
            let lc = node.assign_ids(&[]);
            let lcs = vec![lc];
            let expr = RecExpr::from(lcs);
            // println!("<collect_all_equivs_rec2> leaf node expr: {}", expr.pretty(100));
            exprs_set.insert(expr);
            expr_count += 1;
        } else if children.len() == 1 {
            // unary op
            // println!("<collect_all_equivs_rec2> unary op: {:?}", node);
            let child = children[0];
            // println!("<collect_all_equivs_rec2> child: {:?}", child);
            let mut vec_lcs: Vec<Vec<ArithLanguage>> = vec![];
            collect_all_subexprs(&egraph, &mut vec_lcs, &child);
            // now the vec_lcs is updated, we can convert it to exprs
            for lcs in vec_lcs {
                let mut lcs = lcs;
                // let ids = lcs.iter().map(|lc| lc.children()).flatten().collect::<Vec<_>>();
                let lc_id = Id::from(lcs.len() - 1);
                let lc = node.assign_ids(&[lc_id]);
                lcs.push(lc);
                // println!("<collect_all_equivs_rec2> lcs: {:?}", lcs);
                let expr = RecExpr::from(lcs);
                // println!("<collect_all_equivs_rec2> unary op expr: {}", expr.pretty(100));
                exprs_set.insert(expr);
                expr_count += 1;
            }
        } else if children.len() == 2 {
            // binary op
            // println!("<collect_all_equivs_rec2> binary op: {:?}", node);
            let left_child = children[0];
            let mut vec_lcs_left: Vec<Vec<ArithLanguage>> = vec![];
            // println!("<collect_all_equivs_rec2> binary op left start: {:?}", left_child);
            collect_all_subexprs(&egraph, &mut vec_lcs_left, &left_child);
            // println!("<collect_all_equivs_rec2> binary op left end: {:?}", vec_lcs_left);
            // now the vec_lcs is updated, we can convert it to exprs
            let right_child = children[1];
            let mut vec_lcs_right: Vec<Vec<ArithLanguage>> = vec![];
            // println!("<collect_all_equivs_rec2> binary op right start: {:?}", right_child);
            collect_all_subexprs(&egraph, &mut vec_lcs_right, &right_child);
            // println!("<collect_all_equivs_rec2> binary op right end: {:?}", vec_lcs_right);
            // println!("<collect_all_equivs_rec2> right: {:?}", vec_lcs_right);
            // now the vec_lcs is updated, we can convert it to exprs
            // use cartesian product to get all possible combinations
            for comb in vec_lcs_left.into_iter().cartesian_product(vec_lcs_right.into_iter()) {
                let expr_left = RecExpr::from(comb.0);
                let expr_right = RecExpr::from(comb.1);
                let mut exprs_iter = [expr_left.clone(), expr_right.clone()].into_iter();
                let join_recexprs_fn = |id| {
                    exprs_iter.next().unwrap()
                };
                // let ids = [Id::from(0), Id::from(1)];
                // let lc = node.assign_ids(&[Id::from(0), Id::from(1)]);
                // println!("<collect_all_equivs_rec2> left: {:?}, right: {:?}", expr_left.pretty(100), expr_right.pretty(100));
                let expr = node.join_recexprs(join_recexprs_fn);
                // println!("<collect_all_equivs_rec2> binary op expr: {}", expr.pretty(100));
                exprs_set.insert(expr);
                expr_count += 1;
            }
        } else {
            unreachable!("Language construct with more than 2 children");
        }
    }
    // println!("<collect_all_equivs_rec2> expr_count: {}", expr_count);
}

fn collect_all_subexprs(egraph: &EGraph<ArithLanguage, ExprAstSize>, vec_lcs: &mut Vec<Vec<ArithLanguage>>, child_id: &Id) {
    let mut expr_count = 0;
    let child_eclass = &egraph[*child_id];
    for (i, child_node) in enumerate(&child_eclass.nodes) {
        let key = (child_eclass.data.clone(), i);
        let child_children = child_node.children();
        if child_children.is_empty() {
            // leaf node
            // println!("<collect_all_subexprs> <{}> leaf node: {:?}", child_id, child_node);
            let new_lc = child_node.assign_ids(&[]);
            let new_lcs = vec![new_lc];
            // println!("<collect_all_subexprs> <{}> new_lcs: {:?}", child_id, new_lcs);
            // println!("<collect_all_subexprs> <{}> leaf node expr: {}", child_id, RecExpr::from(new_lcs.clone()).pretty(100));
            vec_lcs.push(new_lcs);
            expr_count += 1;
            // println!("<collect_all_subexprs> <{}> vec_lcs: {:?}", child_id, vec_lcs);
        } else if child_children.len() == 1 {
            // unary op
            // println!("<collect_all_subexprs> <{}> unary op: {:?}", child_id, child_node);
            let grandchild_id = child_children[0];
            let mut new_vec_lcs = vec![];
            collect_all_subexprs(egraph, &mut new_vec_lcs, &grandchild_id);
            for lcs in new_vec_lcs {
                let mut new_lcs = lcs;
                let new_lc_id = Id::from(new_lcs.len() - 1);
                let new_lc = child_node.assign_ids(&[new_lc_id]);
                new_lcs.push(new_lc);
                // println!("<collect_all_subexprs> <{}> new_lcs: {:?}", child_id, new_lcs);
                // println!("<collect_all_subexprs> <{}> unary op expr: {}", child_id, RecExpr::from(new_lcs.clone()).pretty(100));
                vec_lcs.push(new_lcs);
                expr_count += 1;
            }
        } else if child_children.len() == 2 {
            // binary op
            // println!("<collect_all_subexprs> <{}> binary op: {:?}", child_id, child_node);
            let left_grandchild_id = child_children[0];
            let mut new_vec_lcs_left = vec![];
            // println!("<collect_all_subexprs> <{}> binary op left start: {:?}", child_id, left_grandchild_id);
            collect_all_subexprs(egraph, &mut new_vec_lcs_left, &left_grandchild_id);
            // println!("<collect_all_subexprs> <{}> binary op left end: {:?}", child_id, new_vec_lcs_left);

            let right_grandchild_id = child_children[1];
            let mut new_vec_lcs_right = vec![];
            // println!("<collect_all_subexprs> <{}> binary op right start: {:?}", child_id, right_grandchild_id);
            collect_all_subexprs(egraph, &mut new_vec_lcs_right, &right_grandchild_id);
            // println!("<collect_all_subexprs> <{}> binary op right end: {:?}", child_id, new_vec_lcs_right);

            for comb in new_vec_lcs_left.into_iter().cartesian_product(new_vec_lcs_right.into_iter()) {
                let new_lc_left = comb.0;
                let new_lc_right = comb.1;
                let mut new_lcs = vec![];
                let mut new_lcs_iter = [new_lc_left.clone(), new_lc_right.clone()].into_iter();
                let join_recexpr_fn = |id| {
                    RecExpr::from(new_lcs_iter.next().unwrap())
                };
                new_lcs.extend(new_lc_left);
                new_lcs.extend(new_lc_right);
                let expr = child_node.join_recexprs(join_recexpr_fn);
                // println!("<collect_all_subexprs> <{}> expr: {}", child_id, expr.pretty(100));
                let new_lcs = expr.as_ref().to_vec();
                // println!("<collect_all_subexprs> <{}> binary op expr: {}", child_id, expr.pretty(100));
                vec_lcs.push(new_lcs);
                expr_count += 1;
            }
        } else {
            unreachable!("Language construct with more than 2 children");
        }
    }
    // println!("<collect_all_subexprs> <{}> expr_count: {}", child_id, expr_count);
}

fn merge_egraphs<L: Language, N: Analysis<L>>(source_egraph: &EGraph<L, N>, target_egraph: &mut EGraph<L, N>) {
    for id in source_egraph.classes().map(|e| e.id) {
        let expr = source_egraph.id_to_expr(id);
        target_egraph.add_expr(&expr);
    }
    target_egraph.rebuild();
}

#[async_std::main]
async fn main() {
    let mut grammar = Grammar {
        productions: vec![
            Production {
                lhs: "S".to_string(),
                lhs_type: "Op".to_string(),
                rhs: vec![
                    ProdComponent::LhsName("S".to_string()),
                    ProdComponent::LanguageConstruct(ArithLanguage::Add(Default::default())),
                    ProdComponent::LhsName("S".to_string()),
                ]
            },
            Production {
                lhs: "S".to_string(),
                lhs_type: "Op".to_string(),
                rhs: vec![
                    ProdComponent::LhsName("S".to_string()),
                    ProdComponent::LanguageConstruct(ArithLanguage::Sub(Default::default())),
                    ProdComponent::LhsName("S".to_string()),
                ]
            },
            Production {
                lhs: "S".to_string(),
                lhs_type: "Op".to_string(),
                rhs: vec![
                    ProdComponent::LhsName("S".to_string()),
                    ProdComponent::LanguageConstruct(ArithLanguage::Mul(Default::default())),
                    ProdComponent::LhsName("S".to_string()),
                ]
            },
            Production {
                lhs: "S".to_string(),
                lhs_type: "Op".to_string(),
                rhs: vec![
                    ProdComponent::LanguageConstruct(ArithLanguage::Neg(Default::default())),
                    ProdComponent::LhsName("S".to_string()),
                ]
            },
            Production {
                lhs: "S".to_string(),
                lhs_type: "Var".to_string(),
                rhs: vec![
                    ProdComponent::LanguageConstruct(ArithLanguage::Var("x".to_string())),
                    ProdComponent::LanguageConstruct(ArithLanguage::Var("y".to_string())),
                ]
            },
            Production {
                lhs: "S".to_string(),
                lhs_type: "Num".to_string(),
                rhs: vec![ProdComponent::LanguageConstruct(ArithLanguage::Num(0))]
            },
            Production {
                lhs: "S".to_string(),
                lhs_type: "Num".to_string(),
                rhs: vec![ProdComponent::LanguageConstruct(ArithLanguage::Num(1))]
            },
            Production {
                lhs: "S".to_string(),
                lhs_type: "Num".to_string(),
                rhs: vec![ProdComponent::LanguageConstruct(ArithLanguage::Num(2))]
            },
        ]
    };
    let solver = EggSolver::new(grammar.clone());
    let max_size = 9;

    let start = std::time::Instant::now();
    let pts = vec![
        // x * x + y * 2
        // (+ (* x x) (* y 2)
        (HashMap::from([("x".to_string(), 1), ("y".to_string(), 2)]), 5), // 1 * 1 + 2 * 2 = 5
        (HashMap::from([("x".to_string(), 3), ("y".to_string(), 4)]), 17), // 3 * 3 + 4 * 2 = 17
        (HashMap::from([("x".to_string(), 4), ("y".to_string(), 1)]), 18), // 4 * 4 + 1 * 2 = 18
        (HashMap::from([("x".to_string(), 5), ("y".to_string(), 3)]), 31), // 5 * 5 + 3 * 2 = 31
    ];

    // let pts = vec![
    //     // x * x + y * 2 + x * y
    // sizes_corrected = [1, 2, 3, 4, 5, 6, 7, 8, 9]
    // cache_sizes_corrected = [5, 85, 210, 938, 1798, 5892, 10334, 31672, 380988]
    // egraph_classes_corrected = [1, 29, 39, 133, 167, 613, 744, 2691, 6280]
    // egraph_nodes_corrected = [5, 85, 205, 853, 1573, 5215, 8875, 27847, 61021]

    //     (HashMap::from([("x".to_string(), 11), ("y".to_string(), 23)]), 420), // 11 * 11 + 23 * 2 + 11 * 23 = 420
    //     (HashMap::from([("x".to_string(), 1), ("y".to_string(), 2)]), 16), // 1 * 1 + 2 * 2 = 5
    //     (HashMap::from([("x".to_string(), 3), ("y".to_string(), 4)]), 29), // 3 * 3 + 4 * 2 = 17
    //     (HashMap::from([("x".to_string(), 4), ("y".to_string(), 1)]), 22), // 4 * 4 + 1 * 2 = 18
    //     (HashMap::from([("x".to_string(), 5), ("y".to_string(), 3)]), 46), // 5 * 5 + 3 * 2 = 31
    // ];


    let start = std::time::Instant::now();
    let pts = vec![
        // x * y
        (HashMap::from([("x".to_string(), 1), ("y".to_string(), 2)]), 2), // 1 * 2 = 2
        (HashMap::from([("x".to_string(), 3), ("y".to_string(), 4)]), 12), // 3 * 4 = 12
        (HashMap::from([("x".to_string(), 4), ("y".to_string(), 11)]), 44), // 4 * 11 = 44
    ];

    // x * y + z
    // grammar.productions.push(Production {
    //     lhs: "S".to_string(),
    //     lhs_type: "Var".to_string(),
    //     rhs: vec![
    //         ProdComponent::LanguageConstruct(ArithLanguage::Add(Default::default())),
    //     ]
    // });
    grammar.productions.push(
        Production {
            lhs: "S".to_string(),
            lhs_type: "Var".to_string(),
            rhs: vec![
                ProdComponent::LanguageConstruct(ArithLanguage::Var("z".to_string())),
            ]
        },
    );
    let max_size = 5;
    let start = std::time::Instant::now();
    let solver = EggSolver::new(grammar.clone());
    let pts = vec![
        (HashMap::from([("x".to_string(), 1), ("y".to_string(), 2), ("z".to_string(), 3)]), 5), // 1 * 2 + 3 = 5
        (HashMap::from([("x".to_string(), 3), ("y".to_string(), 4), ("z".to_string(), 5)]), 17), // 3 * 4 + 5 = 17
        (HashMap::from([("x".to_string(), 4), ("y".to_string(), 11), ("z".to_string(), 2)]), 46), // 4 * 11 + 2 = 46
    ];

    let exprs = solver.synthesize(max_size, &pts);
    if exprs.is_empty(){
        println!("No expression could be synthesized.");
        assert!(false);
    } else {
        println!("-------------Synthesized Successfully----------------");
        println!("Test case: {:?}", pts);
        // println!("Target program: {}", "x * x + y * 2");
        // println!("Target program: {}", "x * x + y * 2 + x * y");
        println!("Target program: {}", "x * y + z");
        for expr in exprs {
            println!("Program: {}", expr.pretty(100));
        }
        println!("Elapsed time: {:?}ms", start.elapsed().as_millis());
        println!("-----------------------------------------------------");
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_egg_enum() {
        main();
    }

    #[test]
    fn test_pbe_egs_3() {
        let grammar = Grammar {
            productions: vec![
                Production {
                    lhs: "S".to_string(),
                    lhs_type: "Var".to_string(),
                    rhs: vec![
                        ProdComponent::LanguageConstruct(ArithLanguage::Var("x".to_string())),
                        ProdComponent::LanguageConstruct(ArithLanguage::Var("y".to_string())),
                        ProdComponent::LanguageConstruct(ArithLanguage::Var("z".to_string())),
                    ]
                },
                Production {
                    lhs: "S".to_string(),
                    lhs_type: "Const".to_string(),
                    rhs: vec![
                        ProdComponent::LanguageConstruct(ArithLanguage::Num(0)),
                        ProdComponent::LanguageConstruct(ArithLanguage::Num(1)),
                        ProdComponent::LanguageConstruct(ArithLanguage::Num(2)),
                    ]
                },
                Production {
                    lhs: "S".to_string(),
                    lhs_type: "Op".to_string(),
                    rhs: vec![
                        ProdComponent::LanguageConstruct(ArithLanguage::Neg(Default::default())),
                        ProdComponent::LhsName("S".to_string()),
                    ]
                },
                Production {
                    lhs: "S".to_string(),
                    lhs_type: "Op".to_string(),
                    rhs: vec![
                        ProdComponent::LanguageConstruct(ArithLanguage::Add(Default::default())),
                        ProdComponent::LhsName("S".to_string()),
                        ProdComponent::LhsName("S".to_string()),
                    ]
                },
                Production {
                    lhs: "S".to_string(),
                    lhs_type: "Op".to_string(),
                    rhs: vec![
                        ProdComponent::LanguageConstruct(ArithLanguage::Sub(Default::default())),
                        ProdComponent::LhsName("S".to_string()),
                        ProdComponent::LhsName("S".to_string()),
                    ]
                },
                Production {
                    lhs: "S".to_string(),
                    lhs_type: "Op".to_string(),
                    rhs: vec![
                        ProdComponent::LanguageConstruct(ArithLanguage::Mul(Default::default())),
                        ProdComponent::LhsName("S".to_string()),
                        ProdComponent::LhsName("S".to_string()),
                    ]
                },
            ]
        };
        let start = std::time::Instant::now();
        let solver = EggSolver::new(grammar.clone());
        let pts = vec![
            // (x + (y - (x * (y - z)))) - 2 * z
            // (- (+ x (- y (* x (- y z)) (* 2 z))
            (HashMap::from([("x".to_string(), 7), ("y".to_string(), 12), ("z".to_string(), 23)]), 50), // 7 + 12 - (7 * (12 - 23)) - 2 * 23 = 50
            (HashMap::from([("x".to_string(), 11), ("y".to_string(), 13), ("z".to_string(), 29)]), 142), // 11 + 13 - (11 * (13 - 29)) - 2 * 29 = 142
            (HashMap::from([("x".to_string(), 3), ("y".to_string(), 7), ("z".to_string(), 13)]), 2), // 3 + 7 - (3 * (7 - 13)) - 2 * 13 = 2
        ];

        let max_size = 9;
        let exprs = solver.synthesize(max_size, &pts)/*.await*/;
        if exprs.is_empty(){
            println!("No expression could be synthesized.");
            assert!(false);
        } else {
            println!("-------------Synthesized Successfully----------------");
            println!("Test case: {:?}", pts);
            println!("Target program: {}", "x + y - (x * (y - z) ) - 2 * z");
            for expr in exprs {
                println!("Program: {}", expr.pretty(100));
            }
            println!("Elapsed time: {:?}ms", start.elapsed().as_millis());
            println!("-----------------------------------------------------");
        }
    }

    #[test]
    fn test_pbe_egs() {
        let grammar = Grammar {
            productions: vec![
                Production {
                    lhs: "S".to_string(),
                    lhs_type: "Op".to_string(),
                    rhs: vec![
                        ProdComponent::LanguageConstruct(ArithLanguage::Var("x".to_string())),
                        ProdComponent::LanguageConstruct(ArithLanguage::Var("y".to_string())),
                    ]
                },
                Production {
                    lhs: "S".to_string(),
                    lhs_type: "Op".to_string(),
                    rhs: vec![
                        ProdComponent::LanguageConstruct(ArithLanguage::Num(0)),
                    ]
                },
                Production {
                    lhs: "S".to_string(),
                    lhs_type: "Op".to_string(),
                    rhs: vec![
                        ProdComponent::LanguageConstruct(ArithLanguage::Num(1)),
                    ]
                },
                Production {
                    lhs: "S".to_string(),
                    lhs_type: "Op".to_string(),
                    rhs: vec![
                        ProdComponent::LanguageConstruct(ArithLanguage::Num(2)),
                    ]
                },
                Production {
                    lhs: "S".to_string(),
                    lhs_type: "Op".to_string(),
                    rhs: vec![
                        ProdComponent::LanguageConstruct(ArithLanguage::Add(Default::default())),
                        ProdComponent::LhsName("S".to_string()),
                        ProdComponent::LhsName("S".to_string()),
                    ]
                },
                Production {
                    lhs: "S".to_string(),
                    lhs_type: "Op".to_string(),
                    rhs: vec![
                        ProdComponent::LanguageConstruct(ArithLanguage::Sub(Default::default())),
                        ProdComponent::LhsName("S".to_string()),
                        ProdComponent::LhsName("S".to_string()),
                    ]
                },
                Production {
                    lhs: "S".to_string(),
                    lhs_type: "Op".to_string(),
                    rhs: vec![
                        ProdComponent::LanguageConstruct(ArithLanguage::Mul(Default::default())),
                        ProdComponent::LhsName("S".to_string()),
                        ProdComponent::LhsName("S".to_string()),
                    ]
                },
            ]
        };
        let solver = EggSolver::new(grammar.clone());
        let max_size = 7;


        let start = std::time::Instant::now();
        let pts = vec![
            // 2 * x + y
            // (+ (* 2 x) y)
            (HashMap::from([("x".to_string(), 1), ("y".to_string(), 2)]), 4), // 2 * 1 + 2 = 4
            (HashMap::from([("x".to_string(), 3), ("y".to_string(), 4)]), 10), // 2 * 3 + 4 = 10
            (HashMap::from([("x".to_string(), 4), ("y".to_string(), 1)]), 9), // 2 * 4 + 1 = 9
            (HashMap::from([("x".to_string(), 5), ("y".to_string(), 3)]), 13), // 2 * 5 + 3 = 13
            (HashMap::from([("x".to_string(), 6), ("y".to_string(), 2)]), 14), // 2 * 6 + 2 = 14
        ];

        let exprs = solver.synthesize(max_size, &pts);
        if exprs.is_empty(){
            println!("No expression could be synthesized.");
            assert!(false);
        } else {
            println!("-------------Synthesized Successfully----------------");
            println!("Test case: {:?}", pts);
            println!("Target program: {}", "2 * x + y");
            for expr in exprs {
                println!("Program: {}", expr.pretty(100));
            }
            println!("Elapsed time: {:?}ms", start.elapsed().as_millis());
            println!("-----------------------------------------------------");
        }

        let start = std::time::Instant::now();
        let pts = vec![
            // x * y + x
            // (+ (* x y) x)
            (HashMap::from([("x".to_string(), 1), ("y".to_string(), 2)]), 3), // 1 * 2 + 1 = 3
            (HashMap::from([("x".to_string(), 3), ("y".to_string(), 4)]), 15), // 3 * 4 + 3 = 15
            (HashMap::from([("x".to_string(), 4), ("y".to_string(), 1)]), 8), // 4 * 1 + 4 = 8
            (HashMap::from([("x".to_string(), 5), ("y".to_string(), 3)]), 20), // 5 * 3 + 5 = 20
            (HashMap::from([("x".to_string(), 6), ("y".to_string(), 2)]), 18), // 6 * 2 + 6 = 18
        ];

        let exprs = solver.synthesize(max_size, &pts)/*.await*/;
        if exprs.is_empty(){
            println!("No expression could be synthesized.");
            assert!(false);
        } else {
            println!("-------------Synthesized Successfully----------------");
            println!("Test case: {:?}", pts);
            println!("Target program: {}", "x * y + x");
            for expr in exprs {
                println!("Program: {}", expr.pretty(100));
            }
            println!("Elapsed time: {:?}ms", start.elapsed().as_millis());
            println!("-----------------------------------------------------");
        }

        let start = std::time::Instant::now();
        let pts = vec![
            // x * y - x - y
            (HashMap::from([("x".to_string(), 1), ("y".to_string(), 2)]), -1), // 1 * 2 - 1 - 2 = -1
            (HashMap::from([("x".to_string(), 3), ("y".to_string(), 4)]), 5), // 3 * 4 - 3 - 4 = 5
            (HashMap::from([("x".to_string(), 4), ("y".to_string(), 1)]), -1), // 4 * 1 - 4 - 1 = -1
            (HashMap::from([("x".to_string(), 5), ("y".to_string(), 3)]), 7), // 5 * 3 - 5 - 3 = 7
        ];

        let exprs = solver.synthesize(max_size, &pts)/*.await*/;
        if exprs.is_empty(){
            println!("No expression could be synthesized.");
            assert!(false);
        } else {
            println!("-------------Synthesized Successfully----------------");
            println!("Test case: {:?}", pts);
            println!("Target program: {}", "x * y - x - y");
            for expr in exprs {
                println!("Program: {}", expr.pretty(100));
            }
            println!("Elapsed time: {:?}ms", start.elapsed().as_millis());
            println!("-----------------------------------------------------");
        }

        let start = std::time::Instant::now();
        let pts = vec![
            // x * (x - y) + 2 * x
            (HashMap::from([("x".to_string(), 1), ("y".to_string(), 2)]), 1), // 1 * (1 - 2) + 2 * 1 = 1
            (HashMap::from([("x".to_string(), 32), ("y".to_string(), 9)]), 800), // 32 * (32 - 9) + 2 * 32 = 800
            (HashMap::from([("x".to_string(), 29), ("y".to_string(), 23)]), 232), // 4 * (4 - 1) + 2 * 4 = 10
            (HashMap::from([("x".to_string(), 2338), ("y".to_string(), 293)]), 4785886), // 2338 * (2338 - 293) + 2 * 2338 = 4785886
        ];
        let exprs = solver.synthesize(max_size, &pts)/*.await*/;
        if exprs.is_empty(){
            println!("No expression could be synthesized.");
            assert!(false);
        } else {
            println!("-------------Synthesized Successfully----------------");
            println!("Test case: {:?}", pts);
            println!("Target program: {}", "x * (x - y) + 2 * x");
            for expr in exprs {
                println!("Program: {}", expr.pretty(100));
            }
            println!("Elapsed time: {:?}ms", start.elapsed().as_millis());
            println!("-----------------------------------------------------");
        }
    }
}
