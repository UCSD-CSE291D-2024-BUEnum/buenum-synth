use async_std::stream::{self, Stream};
use std::cmp::max;
use std::collections::{HashMap, HashSet};
use std::fmt;
use std::hash::Hash;
use async_std::task;

use anyhow::Error;
use egg::{rewrite as rw, *};
use itertools::Itertools;

define_language! {
    pub enum ArithLanguage {
        Num(i32),
        Var(String),
        "+" = Add([Id; 2]),
        "-" = Sub([Id; 2]),
        "*" = Mul([Id; 2]),
        "neg" = Neg([Id; 1]),
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
            ArithLanguage::Num(n) => pts.iter().map(|(input, output)| (input.clone(), *n)).collect(),
            ArithLanguage::Var(name) => pts.iter().map(|(input, output)| (input.clone(), sem(&[(input.clone(), *input.get(name).unwrap())]))).collect(),
            ArithLanguage::Neg([id]) => o(id)
                .iter()
                .zip(pts)
                .map(|((input, output), _)| (input.clone(), sem(&[(input.clone(), *output)])))
                .collect(),
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
        let io_pairs = egraph[id].data.clone();
        if io_pairs.is_empty() || io_pairs.first().unwrap().0.is_empty() {
            return;
        }
        for (inputs, output) in io_pairs {
            let expr = egraph.id_to_expr(id);
            let new_id = egraph.add_expr(&expr);
            egraph.rebuild();
            let new_output = egraph[new_id].data.iter().find(|(i, _)| i == &inputs).map(|(_, o)| *o).unwrap_or(output);
            if new_output != output {
                let num_id = egraph.add(ArithLanguage::Num(new_output));
                egraph.union(id, num_id);
            }
        }
    }
}

struct Enumerator<'a> {
    grammar: &'a Grammar,
    egraph: EGraph<ArithLanguage, ObsEquiv>,
    cache: HashMap<(ProdName, usize), HashSet<RecExpr<ArithLanguage>>>,
    checked_exprs: HashSet<RecExpr<ArithLanguage>>,
    current_size: usize,
}

impl<'a> Enumerator<'a> {
    fn new(grammar: &'a Grammar) -> Self {
        Enumerator {
            grammar,
            egraph: EGraph::new(ObsEquiv::default()).with_explanations_enabled(),
            cache: HashMap::new(),
            checked_exprs: HashSet::new(),
            current_size: 0,
        }
    }

    fn rebuild(&mut self, pts: &IOPairs) {
        // println!("<Enumerator::rebuild> self.egraph.analysis.pts: {:?}, pts: {:?}", self.egraph.analysis.pts, pts);
        let mut new_egraph = EGraph::new(ObsEquiv { pts: pts.clone() }).with_explanations_enabled();
        for (key, exprs) in &self.cache {
            for expr in exprs {
                new_egraph.add_expr(expr);
            }
        }
        // println!("{}", pretty_egraph(&new_egraph, 2));
        new_egraph.rebuild();
        self.egraph = new_egraph;
    }

    async fn grow(&mut self) {
        let size = self.current_size + 1;
        let mut expr_size = self.current_size;
        // Base case: directly add numbers and variables for size 1
        if size == 1 {
            for prod in &self.grammar.productions {
                for component in &prod.rhs {
                    if let ProdComponent::LanguageConstruct(lang_construct) = component {
                        println!("<Enumerator::grow> prod(size={}): ({}, {}) => {:?}", size, prod.lhs, prod.lhs_type, lang_construct);
                        let mut new_expressions: HashMap<(String, usize), HashSet<RecExpr<ArithLanguage>>> = HashMap::new();
                        match lang_construct {
                            ArithLanguage::Num(_) | ArithLanguage::Var(_) => {
                                // let mut expr = RecExpr::default();
                                let id = self.egraph.add(lang_construct.clone());
                                let expr = self.egraph.id_to_expr(id);
                                expr_size = max(AstSize.cost_rec(&expr), expr_size);
                                // expr.add(lang_construct.clone());
                                new_expressions.entry((prod.lhs.clone(), AstSize.cost_rec(&expr)))
                                               .or_insert_with(HashSet::new)
                                               .insert(expr);
                            },
                            _ => {}
                        }
                        println!("<Enumerator::grow> new_expressions<({}, {})>.len(): {}", prod.lhs, size, new_expressions.values().flatten().count());
                        for (key, exprs) in new_expressions {
                            // for expr in &exprs {
                            //     self.egraph.add_expr(&expr);
                            // }
                            self.cache.entry(key).or_insert_with(HashSet::new).extend(exprs);
                        }
                        task::yield_now().await;
                    }
                }
            }
        } else {
            // Composite expressions
            for prod in &self.grammar.productions {
                for component in &prod.rhs {
                    if let ProdComponent::LanguageConstruct(lang_construct) = component {
                        println!("<Enumerator::grow> prod(size={}): ({}, {}) => {:?}", size, prod.lhs, prod.lhs_type, lang_construct);
                        let mut new_expressions: HashMap<(String, usize), HashSet<RecExpr<ArithLanguage>>> = HashMap::new();
                        match lang_construct {
                            ArithLanguage::Add(_) | ArithLanguage::Sub(_) | ArithLanguage::Mul(_) => {
                                let num_nonterminals = lang_construct.children().len();
                                let partitions: itertools::Combinations<std::ops::Range<usize>> = (1..size).combinations(num_nonterminals - 1);
                                for partition in partitions {
                                    let mut sizes = vec![1];
                                    sizes.extend(partition);
                                    sizes.push(size - sizes.iter().sum::<usize>());
    
                                    let mut expr_parts = Vec::new();
                                    for part_size in &sizes {
                                        if let Some(exprs) = self.cache.get(&(prod.lhs.clone(), *part_size)) {
                                            expr_parts.push(exprs.iter().cloned().collect::<Vec<_>>());
                                        }
                                    }
                                    for expr_combination in expr_parts.into_iter().multi_cartesian_product() {
                                        let mut ids = Vec::new();
    
                                        for e in expr_combination {
                                            ids.push(self.egraph.add_expr(&e));
                                        }
                                        // println!("<Enumerator::grow> lang_construct: {:?}", lang_construct);
                                        let id = match lang_construct {
                                            ArithLanguage::Add(_) => {
                                                self.egraph.add(ArithLanguage::Add([ids[0], ids[1]]))
                                            },
                                            ArithLanguage::Sub(_) => {
                                                self.egraph.add(ArithLanguage::Sub([ids[0], ids[1]]))
                                            },
                                            ArithLanguage::Mul(_) => {
                                                self.egraph.add(ArithLanguage::Mul([ids[0], ids[1]]))
                                            },
                                            _ => unreachable!(),
                                        };
                                        let expr = self.egraph.id_to_expr(id);
                                        expr_size = max(AstSize.cost_rec(&expr), expr_size);

                                        // if let ArithLanguage::Add(_) = lang_construct {
                                        //     println!("<Enumerator::grow> expr: {}", expr.pretty(100));
                                        // }
                                        // println!("<Enumerator::grow> expr: {}", expr.pretty(100));
    
                                        new_expressions.entry((prod.lhs.clone(), AstSize.cost_rec(&expr)))
                                                       .or_insert_with(HashSet::new)
                                                       .insert(expr);
                                    }
                                }
                            },
                            ArithLanguage::Neg(_) => {
                                for part_size in 1..size {
                                    if let Some(exprs) = self.cache.get(&(prod.lhs.clone(), part_size)) {
                                        for expr in exprs {
                                            let mut new_expr = expr.clone();
                                            let id = new_expr.add(lang_construct.clone()); // we don't care the id of a unary operator
                                            expr_size = max(AstSize.cost_rec(&new_expr), expr_size);
                                            // new_expr.add(ArithLanguage::Neg([id]));
                                            // println!("<Enumerator::grow> expr: {}", new_expr.pretty(100));
                                            new_expressions.entry((prod.lhs.clone(), AstSize.cost_rec(&new_expr)))
                                                           .or_insert_with(HashSet::new)
                                                           .insert(new_expr);
                                        }
                                    }
                                }
                            },
                            _ => {}
                        }
                        println!("<Enumerator::grow> new_expressions<({}, {})>.len(): {}", prod.lhs, size, new_expressions.values().flatten().count());
                        for (key, exprs) in new_expressions {
                            // for expr in &exprs {
                            //     self.egraph.add_expr(&expr);
                            // }
                            self.cache.entry(key).or_insert_with(HashSet::new).extend(exprs);
                        }
                        task::yield_now().await;
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
        self.egraph.rebuild();

        // println!("{}", pretty_cache(&self.cache, 2));
        // println!("{}", pretty_egraph(&self.egraph, 2));
        self.current_size = size;
    }


    async fn enumerate(&mut self, size: usize, pts: &IOPairs) -> Option<RecExpr<ArithLanguage>> {
        println!("<Enumerator::enumerate> size: {}, pts: {:?}", size, pts);
        if self.egraph.analysis.pts.len() != pts.len() {
            println!("<Enumerator::enumerate> egraph.analysis.pts: {:?}", self.egraph.analysis.pts);
            self.rebuild(pts);
            println!("<Enumerator::enumerate> egraph.analysis.pts: {:?}", self.egraph.analysis.pts);
        }
        while self.current_size <= size {
            if self.current_size < size {
                println!("<Enumerator::enumerate> Growing to size: {}", self.current_size + 1);
                self.grow().await;
            }
            let start_nonterminal = &self.grammar.productions.first().unwrap().lhs;
            let cache_max_size = self.cache.iter().map(|(k, _)| k.1).max().unwrap();
            if let Some(exprs) = self.cache.get(&(start_nonterminal.clone(), cache_max_size)) {
                for expr in exprs {
                    if self.satisfies_pts(expr, pts) {
                        println!("<Enumerator::enumerate> expr: {:?} satisfies pts: {:?}", expr.pretty(100), pts);
                        return Some(expr.clone());
                    }
                }
            }
            if self.current_size == size {
                break;
            }
            task::yield_now().await;
        }
        None
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

    
async fn synthesize(&self, max_size: usize, pts_all: &IOPairs) -> Option<RecExpr<ArithLanguage>> {
    let mut enumerator = Enumerator::new(&self.grammar);
    let mut pts = vec![];
    let start = std::time::Instant::now();
    println!("<EggSolver::synthesize> Start time: {:?}", start);
    while enumerator.current_size <= max_size && start.elapsed().as_secs() <= 120 {
        if let Some(expr) = enumerator.enumerate(max_size, &pts).await {
            if let Some(cex) = self.verify(&expr, pts_all) {
                println!("<EggSolver::synthesize> Found unique counterexample: {:?}", cex);
                pts.push(cex);
            } else {
                println!("<EggSolver::synthesize> Found expression: {:?} within {:?} seconds", expr.pretty(100), start.elapsed().as_secs());
                return Some(expr);
            }
        }
    }
    None
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

#[async_std::main]
async fn main() {
    let grammar = Grammar {
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

    let solver = EggSolver::new(grammar);
    let max_size = 7;

    let pts = vec![
        // x * x + y * 2
        // (+ (* x x) (* y 2)
        (HashMap::from([("x".to_string(), 1), ("y".to_string(), 2)]), 5), // 1 * 1 + 2 * 2 = 5
        (HashMap::from([("x".to_string(), 3), ("y".to_string(), 4)]), 17), // 3 * 3 + 4 * 2 = 17
        (HashMap::from([("x".to_string(), 4), ("y".to_string(), 1)]), 18), // 4 * 4 + 1 * 2 = 18
        (HashMap::from([("x".to_string(), 5), ("y".to_string(), 3)]), 31), // 5 * 5 + 3 * 2 = 31
    ];
    // let pts = vec![
    //     // 2 * x + y
    //     // (+ (* 2 x) y)
    //     (HashMap::from([("x".to_string(), 1), ("y".to_string(), 2)]), 4), // 2 * 1 + 2 = 4
    //     (HashMap::from([("x".to_string(), 3), ("y".to_string(), 4)]), 10), // 2 * 3 + 4 = 10
    //     (HashMap::from([("x".to_string(), 4), ("y".to_string(), 1)]), 9), // 2 * 4 + 1 = 9
    // ];

    if let Some(expr) = solver.synthesize(max_size, &pts).await {
        println!("Synthesized expression: {}", expr.pretty(100));
    } else {
        println!("No expression could be synthesized.");
        assert!(false);
    }
}

fn pretty_cache(cache: &HashMap<(ProdName, usize), HashSet<RecExpr<ArithLanguage>>>, starting_space: usize)  -> String {
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

fn pretty_egraph(egraph: &EGraph<ArithLanguage, ObsEquiv>, starting_space: usize) -> String {
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
    fn test_egg_enum() {
        main();
    }
}
