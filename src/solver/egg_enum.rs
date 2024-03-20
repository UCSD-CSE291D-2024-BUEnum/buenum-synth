use std::collections::{
    HashMap,
    HashSet
};
use std::fmt;

use anyhow::Error;
use egg::{
    rewrite as rw,
    *
};
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

type ProdName = String;

#[derive(Debug, Clone)]
struct Production {
    lhs: ProdName,
    lhs_type: String,
    rhs: Vec<ProdComponent>
}

#[derive(Debug, Clone)]
enum ProdComponent {
    LhsName(ProdName),
    LanguageConstruct(ArithLanguage)
}

#[derive(Debug, Clone)]
struct Grammar {
    productions: Vec<Production>
}

#[derive(Default, Debug, Clone)]
struct ObsEquiv {
    pts: Vec<(HashMap<String, i32>, i32)>
}

impl Analysis<ArithLanguage> for ObsEquiv {
    type Data = i32;

    fn merge(&mut self, to: &mut Self::Data, from: Self::Data) -> DidMerge {
        if *to != from {
            *to = from;
            DidMerge(true, false)
        } else {
            DidMerge(false, false)
        }
    }

    fn make(egraph: &EGraph<ArithLanguage, Self>, enode: &ArithLanguage) -> Self::Data {
        let x = |i: &Id| egraph[*i].data;
        match enode {
            ArithLanguage::Num(n) => *n,
            ArithLanguage::Var(name) => {
                if let Some((inputs, _)) = egraph.analysis.pts.last() {
                    inputs.get(name).cloned().unwrap_or(0)
                } else {
                    0
                }
            }
            ArithLanguage::Neg([a]) => -x(a),
            ArithLanguage::Add([a, b]) => x(a) + x(b),
            ArithLanguage::Sub([a, b]) => x(a) - x(b),
            ArithLanguage::Mul([a, b]) => x(a) * x(b),
        }
    }

    fn modify(egraph: &mut EGraph<ArithLanguage, Self>, id: Id) {
        if let n = egraph[id].data {
            let added = egraph.add(ArithLanguage::Num(n));
            egraph.union(id, added);
        }
    }
}

struct Enumerator<'a> {
    grammar: &'a Grammar,
    egraph: EGraph<ArithLanguage, ObsEquiv>,
    cache: HashMap<(ProdName, usize), HashSet<RecExpr<ArithLanguage>>>,
    current_size: usize
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
        result.push_str(&format!("{}{}:[{}]\n", " ".repeat(starting_space), eclass.id, expr.pretty(100)));
        for eqc in eclass.iter() {
            result.push_str(&format!("{}<{}>: {}\n", " ".repeat(starting_space * 2), eqc, egraph[eclass.id].data));
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

impl<'a> Enumerator<'a> {
    fn new(grammar: &'a Grammar) -> Self {
        Enumerator {
            grammar,
            egraph: EGraph::new(ObsEquiv::default()).with_explanations_enabled(),
            cache: HashMap::new(),
            current_size: 0
        }
    }

    fn rebuild(&mut self, pts: &[(HashMap<String, i32>, i32)]) {
        self.egraph.analysis.pts = pts.to_vec();
        self.egraph.rebuild();

        let mut to_remove = HashMap::new();
        for (key, exprs) in &self.cache {
            for expr in exprs {
                if !self.satisfies_counter_examples(expr, pts) {
                    to_remove.entry(key.clone()).or_insert_with(HashSet::new).insert(expr.clone());
                }
            }
        }
        for (key, exprs) in to_remove {
            self.cache.entry(key).or_insert_with(HashSet::new).retain(|expr| !exprs.contains(expr));
        }
        // TODO: now, we removed them, how to construct the new expressions?
        // for eclass in self.egraph.classes() {
        //     let expr = self.egraph.id_to_expr(eclass.id);
        //     let size = expr.
        //     let key = (expr.as_ref().clone(), size);
        //     self.cache.entry(key).or_insert_with(HashSet::new).insert(expr);
        // }
    }
    fn grow(&mut self, pts: &[(HashMap<String, i32>, i32)]) {
        let size = self.current_size + 1;
        let mut new_expressions = HashMap::new();

        println!("------------- Growing to size {} -------------", size);
        println!("Points: {:?}", &pts);
        // println!("Previous points: {:?}", &self.egraph.analysis.pts);
        // println!("EGraph before rebuild():\n{}", pretty_egraph(&self.egraph, 2));
        // println!("Before rebuild() cache:\n{}", pretty_cache(&self.cache, 2));
        // self.rebuild(pts);
        // println!("Current points: {:?}", &self.egraph.analysis.pts);
        // println!("EGraph after rebuild():\n{}", pretty_egraph(&self.egraph, 2));
        // println!("After rebuild() cache:\n{}", pretty_cache(&self.cache, 2));
        

        if size == 1 {
            for prod in &self.grammar.productions {
                for component in &prod.rhs {
                    if let ProdComponent::LanguageConstruct(lang_construct) = component {
                        let is_terminal = |lc: &ArithLanguage| match lc {
                            ArithLanguage::Num(_) | ArithLanguage::Var(_) => true,
                            _ => false
                        };
                        if is_terminal(lang_construct) {
                            self.egraph.add(lang_construct.clone());
                            new_expressions
                                .entry((prod.lhs.clone(), size))
                                .or_insert_with(HashSet::new)
                                .insert(RecExpr::from(vec![lang_construct.clone()]));
                        }
                    }
                }
            }
        } else {
            for prod in &self.grammar.productions {
                for left_size in 1..size {
                    let right_size = size - left_size;
                    if let Some(left_exprs) = self.cache.get(&(prod.lhs.clone(), left_size)) {
                        if let Some(right_exprs) = self.cache.get(&(prod.lhs.clone(), right_size)) {
                            for left_expr in left_exprs {
                                for right_expr in right_exprs {
                                    let mut new_egraph = self.egraph.clone().with_explanations_enabled();
                                    let left_id = new_egraph.add_expr(left_expr);
                                    let right_id = new_egraph.add_expr(right_expr);
                                    for component in &prod.rhs {
                                        if let ProdComponent::LanguageConstruct(lang_construct) = component {
                                            match lang_construct {
                                                ArithLanguage::Neg(_) => {
                                                    let expr = ArithLanguage::Neg([left_id]);
                                                    new_egraph.add(expr);
                                                }
                                                ArithLanguage::Add(_) => {
                                                    let expr = ArithLanguage::Add([left_id, right_id]);
                                                    new_egraph.add(expr);
                                                }
                                                ArithLanguage::Sub(_) => {
                                                    let expr = ArithLanguage::Sub([left_id, right_id]);
                                                    new_egraph.add(expr);
                                                }
                                                ArithLanguage::Mul(_) => {
                                                    let expr = ArithLanguage::Mul([left_id, right_id]);
                                                    new_egraph.add(expr);
                                                }
                                                _ => {}
                                            }
                                        }
                                    }
                                    merge_egraphs(&new_egraph, &mut self.egraph);
                                    for eclass in new_egraph.classes() {
                                        let expr = new_egraph.id_to_expr(eclass.id);
                                        new_expressions
                                            .entry((prod.lhs.clone(), size))
                                            .or_insert_with(HashSet::new)
                                            .insert(expr);
                                    }
                                }
                            }
                        }
                    }
                }
            }
        }

        for (key, exprs) in new_expressions {
            // for expr in &exprs {
            //     println!("[{}, {}]: {}", &key.0, &key.1, expr.pretty(100));
            // }
            self.cache.entry(key).or_insert_with(HashSet::new).extend(exprs);
        }

        self.current_size = size;
    }

    fn enumerate(&mut self, size: usize, pts: &[(HashMap<String, i32>, i32)]) -> Vec<RecExpr<ArithLanguage>> {
        // if new points are given, we need to start from scratch
        if self.egraph.analysis.pts.len() != pts.len() {
            self.current_size = 0;
            self.egraph = EGraph::new(ObsEquiv { pts: pts.to_vec() }).with_explanations_enabled();
            self.cache = HashMap::new();
        }
        let mut result = Vec::new();
        while self.current_size < size && result.is_empty(){
            self.grow(pts);
            for eclass in self.egraph.classes() {
                let expr = self.egraph.id_to_expr(eclass.id);
                if self.satisfies_counter_examples(&expr, pts) {
                    println!("{}: {}", self.current_size, expr.pretty(100));
                    result.push(expr);
                }
            }
            }
        result
    }

    fn satisfies_counter_examples(&self, expr: &RecExpr<ArithLanguage>, pts: &[(HashMap<String, i32>, i32)]) -> bool {
        for pt in pts {
            if !self.satisfies_counter_example(expr, pt) {
                return false;
            }
        }
        true
    }

    fn satisfies_counter_example(&self, expr: &RecExpr<ArithLanguage>, pt: &(HashMap<String, i32>, i32)) -> bool {
        let mut egraph = EGraph::new(ObsEquiv { pts: vec![pt.clone()] });
        let id = egraph.add_expr(expr);
        egraph.rebuild();
        let value = egraph[id].data;
        value == pt.1
    }
}

struct EggSolver {
    grammar: Grammar
}

impl EggSolver {
    fn new(grammar: Grammar) -> Self {
        EggSolver { grammar }
    }

    fn synthesize(&self, max_size: usize, pts_all: &[(HashMap<String, i32>, i32)]) -> Option<RecExpr<ArithLanguage>> {
        let mut enumerator = Enumerator::new(&self.grammar);

        let mut pts = vec![];
        // set a timeout
        let start = std::time::Instant::now();
        loop {
            let exprs = enumerator.enumerate(max_size, &pts);
            for expr in exprs {
                if let Some(pt) = self.verify(&expr, pts_all) {
                    pts.push(pt);
                } else {
                    println!("{}: {}", AstSize.cost_rec(&expr), expr.pretty(100));
                    return Some(expr);
                }
            }
            if start.elapsed().as_secs() > 120 { // 2 minutes
                break;
            }
        }
        None
    }
    fn verify(&self, expr: &RecExpr<ArithLanguage>, pts_all: &[(HashMap<String, i32>, i32)]) -> Option<(HashMap<String, i32>, i32)> {
        for pt in pts_all {
            let mut egraph = EGraph::new(ObsEquiv { pts: vec![pt.clone()] });
            let id = egraph.add_expr(expr);
            egraph.rebuild();
            let value = egraph[id].data;
            if value != pt.1 {
                return Some(pt.clone());
            }
        }
        None
    }
}

fn main() {
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
    let max_size = 3;

    // let pts = vec![
    //     // x * x + y * 2
    //     (HashMap::from([("x".to_string(), 1), ("y".to_string(), 2)]), 5), // 1 * 1 + 2 * 2 = 5
    //     (HashMap::from([("x".to_string(), 3), ("y".to_string(), 4)]), 17), // 3 * 3 + 4 * 2 = 17
    //     (HashMap::from([("x".to_string(), 4), ("y".to_string(), 1)]), 18), // 4 * 4 + 1 * 2 = 18
    //     (HashMap::from([("x".to_string(), 5), ("y".to_string(), 3)]), 31), // 5 * 5 + 3 * 2 = 31
    // ];
    let pts = vec![
        // 2 * x + y
        // (+ (* 2 x) y)
        (HashMap::from([("x".to_string(), 1), ("y".to_string(), 2)]), 4), // 2 * 1 + 2 = 4
        (HashMap::from([("x".to_string(), 3), ("y".to_string(), 4)]), 10), // 2 * 3 + 4 = 10
        (HashMap::from([("x".to_string(), 4), ("y".to_string(), 1)]), 9), // 2 * 4 + 1 = 9
    ];

    if let Some(expr) = solver.synthesize(max_size, &pts) {
        println!("Synthesized expression: {}", expr.pretty(100));
    } else {
        println!("No expression could be synthesized.");
        assert!(false);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_egg_enum() {
        main();
    }
}
