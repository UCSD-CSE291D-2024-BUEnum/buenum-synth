use std::collections::HashMap;
use std::vec;

use egg::{
    rewrite as rw,
    *
};
use strum_macros::Display;

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
    pts: Vec<(HashMap<String, i32>, i32)>,
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
                let mut sum = 0;
                for (inputs, output) in &egraph.analysis.pts {
                    if let Some(value) = inputs.iter().find(|(var, _)| *var == name).map(|(_, val)| *val) {
                        sum += value;
                    }
                }
                sum
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
    runner: Runner<ArithLanguage, ObsEquiv>,
    cache: HashMap<(ProdName, usize), EGraph<ArithLanguage, ObsEquiv>>,
    current_size: usize
}

// target_egrraph.egraph_union(&source_egraph); // TODO: Error in library function implementation
fn merge_egraphs<L: Language, N: Analysis<L>>(source_egraph: &EGraph<L, N>, target_egraph: &mut EGraph<L, N>) {
    // Iterate through all e-classes in the source e-graph
    for id in source_egraph.classes().map(|e| e.id) {
        let expr = source_egraph.id_to_expr(id);
        target_egraph.add_expr(&expr);
    }
    target_egraph.rebuild();
}

fn pretty_egraph<L: Language, N: Analysis<L>>(egraph: &EGraph<L, N>) 
where L: std::fmt::Display
{
    println!("EGraph size: {}", egraph.number_of_classes());
    for class in egraph.classes() {
        println!("Class {}: {:?} [{}]", class.id, egraph[class.id].data, egraph.id_to_expr(class.id).pretty(80));
    }
}

macro_rules! handle_language_construct {
    ($lang_construct:expr, $egraph:ident, $new_expressions:ident, $prod:ident, $size:ident, $pts:ident, $is_terminal:expr) => {
        if $is_terminal($lang_construct) {
            $egraph.add($lang_construct.clone());
            merge_egraphs(&$egraph, &mut $new_expressions.entry(($prod.lhs.clone(), $size)).or_insert(EGraph::new(ObsEquiv { pts: $pts.to_vec() }).with_explanations_enabled()));
        }
    };
}

macro_rules! handle_binary_op {
    ($op:expr, $left_mapping:ident, $right_mapping:ident, $new_egraph:ident, $new_expressions:ident, $prod:ident, $size:ident, $pts:ident) => {
        for &left_id in $left_mapping.keys() {
            for &right_id in $right_mapping.keys() {
                let new_left_id = $left_mapping[&left_id];
                let new_right_id = $right_mapping[&right_id];
                let expr = $op([new_left_id, new_right_id]);
                $new_egraph.add(expr);
                merge_egraphs(&$new_egraph, &mut $new_expressions.entry(($prod.lhs.clone(), $size)).or_insert(EGraph::new(ObsEquiv { pts: $pts.to_vec() }).with_explanations_enabled()));
            }
        }
    };
}

impl<'a> Enumerator<'a> {
    fn new(grammar: &'a Grammar) -> Self {
        Enumerator {
            grammar,
            runner: Runner::default(),
            cache: HashMap::new(),
            current_size: 0
        }
    }

    fn grow(&mut self, pts: &[(HashMap<String, i32>, i32)]) {
        let size = self.current_size + 1;
        let mut new_expressions = HashMap::new();
    
        if size == 1 {
            for prod in &self.grammar.productions {
                let mut egraph = EGraph::new(ObsEquiv { pts: pts.to_vec() }).with_explanations_enabled();
                for component in &prod.rhs {
                    if let ProdComponent::LanguageConstruct(lang_construct) = component {
                        let is_terminal = |lc: &ArithLanguage| match lc {
                            ArithLanguage::Num(_) | ArithLanguage::Var(_) => true,
                            _ => false,
                        };
                        handle_language_construct!(lang_construct, egraph, new_expressions, prod, size, pts, is_terminal);
                    }
                }
            }
            pretty_egraph(&new_expressions[&("S".to_string(), 1)]);
        } else {
            for prod in &self.grammar.productions {
                for left_size in 1..size {
                    let right_size = size - left_size;
                    if let Some(left_egraph) = self.cache.get(&(prod.lhs.clone(), left_size)) {
                        if let Some(right_egraph) = self.cache.get(&(prod.lhs.clone(), right_size)) {
                            for left_enode in left_egraph.classes() {
                                for right_enode in right_egraph.classes() {
                                    // println!("Merging left and right egraphs:");
                                    let mut new_egraph = left_egraph.clone().with_explanations_enabled();
                                    let left_mapping: HashMap<_, _> = left_egraph.classes().map(|c| (c.id, new_egraph.add_expr(&left_egraph.id_to_expr(c.id)))).collect();
                                    let right_mapping: HashMap<_, _> = right_egraph.classes().map(|c| (c.id, new_egraph.add_expr(&right_egraph.id_to_expr(c.id)))).collect();
                                    // println!("Left egraph:");
                                    // pretty_egraph(&new_egraph);
                                    // println!("Right egraph:");
                                    // pretty_egraph(&right_egraph);
                                    // println!("Merging...");
                                    // merge_egraphs(right_egraph, &mut new_egraph);
                                    // println!("Merged egraph:");
                                    // pretty_egraph(&new_egraph);
                                    for component in &prod.rhs {
                                        if let ProdComponent::LanguageConstruct(lang_construct) = component {
                                            match lang_construct {
                                                ArithLanguage::Neg(_) => {
                                                    let mut unary_egraph = new_egraph.clone().with_explanations_enabled();
                                                    for eclass in new_egraph.classes() {
                                                        let new_id = unary_egraph.add_expr(&new_egraph.id_to_expr(eclass.id));
                                                        let expr = ArithLanguage::Neg([new_id]);
                                                        unary_egraph.add(expr);
                                                    }
                                                    merge_egraphs(&unary_egraph, &mut new_expressions.entry((prod.lhs.clone(), size)).or_insert(EGraph::new(ObsEquiv { pts: pts.to_vec() }).with_explanations_enabled()));
                                                }
                                                ArithLanguage::Add(_) => handle_binary_op!(ArithLanguage::Add, left_mapping, right_mapping, new_egraph, new_expressions, prod, size, pts),
                                                ArithLanguage::Sub(_) => handle_binary_op!(ArithLanguage::Sub, left_mapping, right_mapping, new_egraph, new_expressions, prod, size, pts),
                                                ArithLanguage::Mul(_) => handle_binary_op!(ArithLanguage::Mul, left_mapping, right_mapping, new_egraph, new_expressions, prod, size, pts),
                                                _ => {}
                                            }
                                        }
                                    }
                                }
                            }
                        }
                    }
                }
            }
            pretty_egraph(&new_expressions[&("S".to_string(), size)]);
        }
        for (key, egraph) in new_expressions {
            self.cache.entry(key).or_insert(egraph);
        }
    
        self.current_size = size;
    }

    fn enumerate(&mut self, size: usize, pts: &[(HashMap<String, i32>, i32)]) -> Vec<RecExpr<ArithLanguage>> {
        while self.current_size < size {
            self.grow(pts);
        }

        let mut result = Vec::new();
        for ((_, expr_size), egraph) in &self.cache {
            if *expr_size == size {
                for eclass in egraph.classes() {
                    let expr = egraph.id_to_expr(eclass.id);
                    if self.satisfies_counter_examples(&expr, pts) {
                        result.push(expr);
                    }
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
    grammar: Grammar,
}

impl EggSolver {
    fn new(grammar: Grammar) -> Self {
        EggSolver { grammar }
    }

    fn synthesize(&self, max_size: usize, pts: &[(HashMap<String, i32>, i32)]) -> Option<RecExpr<ArithLanguage>> {
        let mut enumerator = Enumerator::new(&self.grammar);
        for size in 1..=max_size {
            let exprs = enumerator.enumerate(size, pts);
            if let Some(expr) = exprs.into_iter().next() {
                return Some(expr);
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
            // add unary op
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
    let max_size = 4;

    let pts = vec![ // x * 2 + y
        (HashMap::from([("x".to_string(), 1), ("y".to_string(), 2)]), 4),
        (HashMap::from([("x".to_string(), 3), ("y".to_string(), 4)]), 10),
        (HashMap::from([("x".to_string(), 4), ("y".to_string(), 1)]), 9),
    ];

    if let Some(expr) = solver.synthesize(max_size, &pts) {
        println!("Synthesized expression: {}", expr.pretty(100));
    } else {
        println!("No expression could be synthesized.");
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_main() {
        main();
        /*
running 1 test
EGraph size: 5
Class 0: 8 [x]
Class 5: 1 [1]
Class 2: 7 [y]
Class 4: 0 [0]
Class 6: 2 [2]
EGraph size: 23
Class 0: 1 [1]
Class 32: 0 [(- 1 1)]
Class 64: 4 [(* 2 2)]
Class 23: 3 [(+ 2 1)]
Class 70: -2 [(neg 2)]
Class 49: -5 [(- 2 y)]
Class 11: 8 [x]
Class 72: -1 [(neg 1)]
Class 37: 5 [(- y 2)]
Class 69: -8 [(neg x)]
Class 63: 14 [(* 2 y)]
Class 51: -6 [(- 2 x)]
Class 48: 6 [(- x 2)]
Class 71: -7 [(neg y)]
Class 68: 16 [(* x 2)]
Class 1: 7 [y]
Class 27: 10 [(+ 2 x)]
Class 24: 9 [(+ 2 y)]
Class 59: 49 [(* y y)]
Class 65: 64 [(* x x)]
Class 15: 15 [(+ x y)]
Class 6: 2 [2]
Class 67: 56 [(* x y)]
test solver::egg_enum::tests::test_main has been running for over 60 seconds
EGraph size: 88
Class 0: 8 [x]
Class 469: 21 [(* (+ 2 1) y)]
Class 457: -42 [(* (- 2 x) y)]
Class 506: -9 [(neg (+ 2 y))]
Class 183: 71 [(+ (* x x) y)]
Class 49: 4 [(+ 2 2)]
Class 177: 17 [(+ (* x 2) 1)]
Class 500: -49 [(neg (* y y))]
Class 482: 12 [(* (- x 2) 2)]
Class 348: 57 [(- (* x x) y)]
Class 476: 32 [(* (* 2 2) x)]
Class 470: 112 [(* (* 2 y) x)]
Class 458: 392 [(* (* y y) x)]
Class 7: -8 [(neg x)]
Class 196: -57 [(- y (* x x))]
Class 62: -5 [(+ 2 (neg y))]
Class 446: 40 [(* (- y 2) x)]
Class 184: 58 [(+ (* x y) 2)]
Class 245: -62 [(- 2 (* x x))]
Class 501: -3 [(neg (+ 2 1))]
Class 233: -63 [(- 1 (* x x))]
Class 452: -40 [(* (- 2 y) x)]
Class 483: 42 [(* (- x 2) y)]
Class 32: 56 [(* x y)]
Class 160: 51 [(+ (* y y) 2)]
Class 26: 3 [(+ 2 1)]
Class 20: 15 [(+ x y)]
Class 337: 54 [(- (* x y) 2)]
Class 477: 28 [(* (* 2 2) y)]
Class 2: 2 [2]
Class 130: 0 [(+ x (neg x))]
Class 502: -14 [(neg (* 2 y))]
Class 179: 23 [(+ (* x 2) y)]
Class 496: -56 [(neg (* x y))]
Class 484: 512 [(* (* x x) x)]
Class 161: 50 [(+ (* y y) 1)]
Class 472: 98 [(* (* 2 y) y)]
Class 15: 1 [1]
Class 460: 343 [(* (* y y) y)]
Class 9: 7 [y]
Class 326: 47 [(- (* y y) 2)]
Class 3: 10 [(+ 2 x)]
Class 454: -35 [(* (- 2 y) y)]
Class 448: 35 [(* (- y 2) y)]
Class 314: -13 [(- (- 2 y) x)]
Class 52: 16 [(+ 2 (* 2 y))]
Class 180: 66 [(+ (* x x) 2)]
Class 46: 5 [(+ 2 (+ 2 1))]
Class 491: 72 [(* (+ 2 y) x)]
Class 40: -2 [(neg 2)]
Class 241: -55 [(- 1 (* x y))]
Class 497: -4 [(neg (* 2 2))]
Class 485: 128 [(* (* x x) 2)]
Class 503: -10 [(neg (+ 2 x))]
Class 22: 14 [(* 2 y)]
Class 16: 9 [(+ 2 y)]
Class 461: 120 [(* (+ x y) x)]
Class 455: -48 [(* (- 2 x) x)]
Class 467: 24 [(* (+ 2 1) x)]
Class 449: 80 [(* (+ 2 x) x)]
Class 254: -54 [(- 2 (* x y))]
Class 504: -16 [(neg (* x 2))]
Class 181: 65 [(+ (* x x) 1)]
Class 492: 18 [(* (+ 2 y) 2)]
Class 486: 448 [(* (* x x) y)]
Class 352: 13 [(- (+ x y) 2)]
Class 346: 62 [(- (* x x) 2)]
Class 145: 22 [(+ (+ x y) y)]
Class 462: 30 [(* (+ x y) 2)]
Class 456: -12 [(* (- 2 x) 2)]
Class 5: 6 [(- x 2)]
Class 322: 41 [(- (* y y) x)]
Class 249: -47 [(- 2 (* y y))]
Class 450: 20 [(* (+ 2 x) 2)]
Class 499: -64 [(neg (* x x))]
Class 505: -15 [(neg (+ x y))]
Class 493: 63 [(* (+ 2 y) y)]
Class 42: 64 [(* x x)]
Class 170: 11 [(+ (* 2 2) y)]
Class 36: -6 [(- 2 x)]
Class 481: 48 [(* (- x 2) x)]
Class 30: 49 [(* y y)]
Class 24: -1 [(neg 1)]
Class 213: -41 [(- x (* y y))]
Class 18: -7 [(neg y)]
Class 335: 55 [(- (* x y) 1)]
Class 463: 105 [(* (+ x y) y)]
Class 451: 70 [(* (+ 2 x) y)]
Synthesized expression: (+ (* x 2) y)
*/
    }
}