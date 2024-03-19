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
        "+" = Add([Id; 2]),
        "-" = Sub([Id; 2]),
        "*" = Mul([Id; 2]),
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
    pts: Vec<HashMap<String, i32>>,
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

impl<'a> Enumerator<'a> {
    fn new(grammar: &'a Grammar) -> Self {
        Enumerator {
            grammar,
            runner: Runner::default(),
            cache: HashMap::new(),
            current_size: 0
        }
    }

    fn grow(&mut self, pts: &[HashMap<String, i32>]) {
        let size = self.current_size + 1;
        let mut new_expressions = HashMap::new();
    
        if size == 1 {
            for prod in &self.grammar.productions {
                let mut egraph = EGraph::new(ObsEquiv { pts: pts.to_vec() }).with_explanations_enabled();
                for component in &prod.rhs {
                    if let ProdComponent::LanguageConstruct(lang_construct) = component {
                        match lang_construct {
                            ArithLanguage::Num(_) => {
                                egraph.add(lang_construct.clone());
                                merge_egraphs(&egraph, &mut new_expressions.entry((prod.lhs.clone(), size)).or_insert(EGraph::new(ObsEquiv { pts: pts.to_vec() }).with_explanations_enabled()));
                            }
                            _ => {}
                        }
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
                                    let mut new_egraph = left_egraph.clone();
                                    merge_egraphs(right_egraph, &mut new_egraph);
                                    for component in &prod.rhs {
                                        if let ProdComponent::LanguageConstruct(lang_construct) = component {
                                            match lang_construct {
                                                ArithLanguage::Add(_) => {
                                                    let expr = ArithLanguage::Add([left_enode.id, right_enode.id]);
                                                    new_egraph.add(expr);
                                                    merge_egraphs(&new_egraph, &mut new_expressions.entry((prod.lhs.clone(), size)).or_insert(EGraph::new(ObsEquiv { pts: pts.to_vec() }).with_explanations_enabled()));
                                                }
                                                ArithLanguage::Sub(_) => {
                                                    let expr = ArithLanguage::Sub([left_enode.id, right_enode.id]);
                                                    new_egraph.add(expr);
                                                    merge_egraphs(&new_egraph, &mut new_expressions.entry((prod.lhs.clone(), size)).or_insert(EGraph::new(ObsEquiv { pts: pts.to_vec() }).with_explanations_enabled()));
                                                }
                                                ArithLanguage::Mul(_) => {
                                                    let expr = ArithLanguage::Mul([left_enode.id, right_enode.id]);
                                                    new_egraph.add(expr);
                                                    merge_egraphs(&new_egraph, &mut new_expressions.entry((prod.lhs.clone(), size)).or_insert(EGraph::new(ObsEquiv { pts: pts.to_vec() }).with_explanations_enabled()));
                                                }
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

    fn enumerate(&mut self, size: usize, pts: &[HashMap<String, i32>]) -> Vec<RecExpr<ArithLanguage>> {
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

    fn satisfies_counter_examples(&self, expr: &RecExpr<ArithLanguage>, pts: &[HashMap<String, i32>]) -> bool {
        for pt in pts {
            if !self.satisfies_counter_example(expr, pt) {
                return false;
            }
        }
        true
    }

    fn satisfies_counter_example(&self, expr: &RecExpr<ArithLanguage>, pt: &HashMap<String, i32>) -> bool {
        let mut egraph = EGraph::new(ObsEquiv { pts: vec![pt.clone()] });
        let id = egraph.add_expr(expr);
        egraph.rebuild();
        let value = egraph[id].data;
        pt.values().all(|&v| v == value)
    }
}

struct EggSolver {
    grammar: Grammar,
}

impl EggSolver {
    fn new(grammar: Grammar) -> Self {
        EggSolver { grammar }
    }

    fn synthesize(&self, max_size: usize, pts: &[HashMap<String, i32>]) -> Option<RecExpr<ArithLanguage>> {
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

    let pts = vec![
        HashMap::from([("x".to_string(), 1), ("y".to_string(), 2)]),
        HashMap::from([("x".to_string(), 3), ("y".to_string(), 4)]),
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
    }
}