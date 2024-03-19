use std::cell::RefCell;
use std::collections::HashMap;

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
    cache: HashMap<(ProdName, usize), EGraph<ArithLanguage, ()>>,
    current_size: usize
}

impl<'a> Enumerator<'a> {
    fn get_expr(&self, key: &(ProdName, usize), id: Id) -> RecExpr<ArithLanguage> {
        self.cache.get(&key).unwrap().id_to_expr(id)
    }
    fn new(grammar: &'a Grammar) -> Self {
        Enumerator {
            grammar,
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
                for component in &prod.rhs {
                    if let ProdComponent::LanguageConstruct(lang_construct) = component {
                        match lang_construct {
                            ArithLanguage::Num(n) => {
                                let mut egraph = EGraph::new(());
                                egraph.add(lang_construct.clone());
                                new_expressions.insert((prod.lhs.clone(), size), egraph);
                            }
                            _ => {}
                        }
                    }
                }
            }
        } else {
            // Composite expressions
            for prod in &self.grammar.productions {
                for left_size in 1..size {
                    let right_size = size - left_size;
                    if let Some(left_egraph) = self.cache.get(&(prod.lhs.clone(), left_size)) {
                        if let Some(right_egraph) = self.cache.get(&(prod.lhs.clone(), right_size)) {
                            for left_ec in left_egraph.classes() {
                                for right_ec in right_egraph.classes() {
                                    let left_id = left_ec.id;
                                    let right_id = right_ec.id;
                                    let mut egraph = EGraph::new(());
                                    // Add the operation based on production rule
                                    for component in &prod.rhs {
                                        if let ProdComponent::LanguageConstruct(lang_construct) = component {
                                            match lang_construct {
                                                ArithLanguage::Add(_) => {
                                                    egraph.add(ArithLanguage::Add([left_id, right_id]));
                                                }
                                                ArithLanguage::Sub(_) => {
                                                    egraph.add(ArithLanguage::Sub([left_id, right_id]));
                                                }
                                                ArithLanguage::Mul(_) => {
                                                    egraph.add(ArithLanguage::Mul([left_id, right_id]));
                                                }
                                                _ => unreachable!()
                                            }
                                        }
                                    }
                                    new_expressions
                                        .entry((prod.lhs.clone(), size))
                                        .or_insert_with(|| EGraph::new(()))
                                        .egraph_union(&egraph);
                                }
                            }
                        }
                    }
                }
            }
        }

        // Update cache with new expressions
        for (key, egraph) in new_expressions {
            self.cache.insert(key, egraph);
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
        for ((prod_name, expr_size), egraph) in &self.cache {
            if *expr_size == size {
                for eclass in egraph.classes() {
                    let id = eclass.id;
                    let expr = egraph.id_to_expr(id);
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

    // let enumerator = RefCell::new(Enumerator::new(&grammar));
    let mut enumerator = Enumerator::new(&grammar);
    let max_size = 5; // Adjust this value based on the depth of enumeration you desire

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
