use std::collections::HashMap;
use std::vec;

use egg::{
    rewrite as rw,
    *
};

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

struct Enumerator<'a> {
    grammar: &'a Grammar,
    runner: Runner<ArithLanguage, ()>,
    cache: HashMap<(ProdName, usize), EGraph<ArithLanguage, ()>>,
    current_size: usize
}

// target_egrraph.egraph_union(&source_egraph); // TODO: Error in library function implementation
fn merge_egraphs<L: Language, N: Analysis<L>>(source_egraph: &EGraph<L, N>, target_egraph: &mut EGraph<L, N>) {
    // Iterate through all e-classes in the source e-graph
    for id in source_egraph.classes().map(|e| e.id) {
        // Convert each e-class id to a representative expression
        let expr = source_egraph.id_to_expr(id);
        
        // Add the expression to the target e-graph
        target_egraph.add_expr(&expr);
    }
    
    // Rebuild the target e-graph to restore invariants
    target_egraph.rebuild();
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

    fn grow(&mut self) {
        let size = self.current_size + 1;
        let mut new_expressions = HashMap::new();
    
        // Base case: directly add numbers for size 1
        if size == 1 {
            for prod in &self.grammar.productions {
                let mut egraph = EGraph::new(()).with_explanations_enabled();
                for component in &prod.rhs {
                    if let ProdComponent::LanguageConstruct(lang_construct) = component {
                        match lang_construct {
                            ArithLanguage::Num(_) => {
                                egraph.add(lang_construct.clone());
                                merge_egraphs(&egraph, &mut new_expressions.entry((prod.lhs.clone(), size)).or_insert(EGraph::new(()).with_explanations_enabled()));
                            }
                            _ => {}
                        }
                    }
                }
            }
            // println!("New expressions: {:?}", new_expressions);
        } else {
            // Composite expressions
            for prod in &self.grammar.productions {
                // let mut egraph = EGraph::new(()).with_explanations_enabled();
                for left_size in 1..size {
                    let right_size = size - left_size;
                    if let Some(left_egraph) = self.cache.get(&(prod.lhs.clone(), left_size)) {
                        if let Some(right_egraph) = self.cache.get(&(prod.lhs.clone(), right_size)) {
                            for left_enode in left_egraph.classes() {
                                for right_enode in right_egraph.classes() {
                                    let mut new_egraph = left_egraph.clone();
                                    // new_egraph.egraph_union(right_egraph);
                                    merge_egraphs(right_egraph, &mut new_egraph);
                                    for component in &prod.rhs {
                                        if let ProdComponent::LanguageConstruct(lang_construct) = component {
                                            match lang_construct {
                                                ArithLanguage::Add(_) => {
                                                    let expr = ArithLanguage::Add([left_enode.id, right_enode.id]);
                                                    new_egraph.add(expr);
                                                    merge_egraphs(&new_egraph, &mut new_expressions.entry((prod.lhs.clone(), size)).or_insert(EGraph::new(()).with_explanations_enabled()));
                                                    // new_expressions.entry((prod.lhs.clone(), size)).or_insert(EGraph::new(()).with_explanations_enabled()).add(expr);
                                                }
                                                ArithLanguage::Sub(_) => {
                                                    let expr = ArithLanguage::Sub([left_enode.id, right_enode.id]);
                                                    new_egraph.add(expr);
                                                    merge_egraphs(&new_egraph, &mut new_expressions.entry((prod.lhs.clone(), size)).or_insert(EGraph::new(()).with_explanations_enabled()));
                                                    // new_expressions.entry((prod.lhs.clone(), size)).or_insert(EGraph::new(()).with_explanations_enabled()).add(expr);
                                                }
                                                ArithLanguage::Mul(_) => {
                                                    let expr = ArithLanguage::Mul([left_enode.id, right_enode.id]);
                                                    new_egraph.add(expr);
                                                    merge_egraphs(&new_egraph, &mut new_expressions.entry((prod.lhs.clone(), size)).or_insert(EGraph::new(()).with_explanations_enabled()));
                                                    // new_expressions.entry((prod.lhs.clone(), size)).or_insert(EGraph::new(()).with_explanations_enabled()).add(expr);
                                                }
                                                _ => {}
                                            }
                                            // new_expressions.insert((prod.lhs.clone(), size), new_egraph.clone());
                                        }
                                    }
                                    // println!("New expressions: {:?}", new_expressions);
                                }
                            }
                        }
                    }
                }
                // new_expressions.insert((prod.lhs.clone(), size), egraph);
            }
        }
        // println!("New expressions Finally: {:?}", new_expressions);
        // Update cache with new expressions
        for (key, egraph) in new_expressions {
            self.cache.entry(key).or_insert(egraph);
        }
    
        self.current_size = size;
    }

    fn enumerate(&mut self, size: usize) -> Vec<RecExpr<ArithLanguage>> {
        while self.current_size < size {
            self.grow();
        }

        // Access the cache directly for expressions of the given size.
        // This ensures we only collect expressions that match the size criteria.
        let mut result = Vec::new();
        for ((_, expr_size), egraph) in &self.cache {
            if *expr_size == size {
                for eclass in egraph.classes() {
                    let expr = egraph.id_to_expr(eclass.id);
                    result.push(expr);
                }
            }
        }
        result
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

    let mut enumerator = Enumerator::new(&grammar);
    let max_size = 3; // Adjust this value based on the depth of enumeration you desire

    for size in 1..=max_size {
        println!("Enumerating programs of size {}", size);
        let exprs = enumerator.enumerate(size);
        for expr in exprs {
            println!("{}", expr.pretty(100));
        }
        println!(); // Just to have a clear separation for each size's output
    }
    println!("Done!");
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_main() {
        main();
    }
}
