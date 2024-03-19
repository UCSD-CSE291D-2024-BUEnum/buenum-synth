use std::collections::{HashMap, HashSet};

use egg::{
    rewrite as rw,
    *
};
use std::fmt;
use itertools::Itertools;

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
    cache: HashMap<(ProdName, usize), HashSet<RecExpr<ArithLanguage>>>,
    current_size: usize
}

impl<'a> Enumerator<'a> {
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
                let mut expr = RecExpr::default();
                for component in &prod.rhs {
                    if let ProdComponent::LanguageConstruct(lang_construct) = component {
                        match lang_construct {
                            ArithLanguage::Num(_) => {
                                expr.add(lang_construct.clone());
                                new_expressions.entry((prod.lhs.clone(), size))
                                               .or_insert_with(HashSet::new)
                                               .insert(expr.clone());
                            },
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
                    if let Some(left_exprs) = self.cache.get(&(prod.lhs.clone(), left_size)) {
                        if let Some(right_exprs) = self.cache.get(&(prod.lhs.clone(), right_size)) {
                            for left_expr in left_exprs {
                                for right_expr in right_exprs {
                                    let mut expr = RecExpr::default();
                                    // Clone and add all nodes from left_expr to the new expr
                                    let lhs_ids: Vec<Id> = left_expr.as_ref().iter().map(|node| expr.add(node.clone())).collect();
                                    // Clone and add all nodes from right_expr to the new expr
                                    let rhs_ids: Vec<Id> = right_expr.as_ref().iter().map(|node| expr.add(node.clone())).collect();
    
                                    // Use the last ids from lhs_ids and rhs_ids for the arithmetic operation
                                    let lhs_id = *lhs_ids.last().unwrap();
                                    let rhs_id = *rhs_ids.last().unwrap();
    
                                    // Add the operation based on production rule
                                    for component in &prod.rhs {
                                        if let ProdComponent::LanguageConstruct(lang_construct) = component {
                                            match lang_construct {
                                                ArithLanguage::Add(_) => {
                                                    expr.add(ArithLanguage::Add([lhs_id, rhs_id]));
                                                },
                                                ArithLanguage::Sub(_) => {
                                                    expr.add(ArithLanguage::Sub([lhs_id, rhs_id]));
                                                },
                                                ArithLanguage::Mul(_) => {
                                                    expr.add(ArithLanguage::Mul([lhs_id, rhs_id]));
                                                },
                                                _ => {}
                                            }
                                        }
                                    }
    
                                    new_expressions.entry((prod.lhs.clone(), size))
                                                   .or_insert_with(HashSet::new)
                                                   .insert(expr);
                                }
                            }
                        }
                    }
                }
            }
        }
    
        // Update cache with new expressions
        for (key, exprs) in new_expressions {
            self.cache.entry(key).or_insert_with(HashSet::new).extend(exprs);
        }
    
        self.current_size = size;
    }
    

    fn enumerate(&mut self, size: usize) -> Vec<RecExpr<ArithLanguage>> {
        while self.current_size < size {
            self.grow();
        }
        // Access the cache directly for expressions of the given size.
        // This ensures we only collect expressions that match the size criteria.
        self.cache.iter()
            .filter_map(|((_, expr_size), exprs)| {
                if *expr_size == size {
                    Some(exprs.clone())
                } else {
                    None
                }
            })
            .flatten()
            .collect()
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
        let programs = enumerator.enumerate(size);
        for program in &programs {
            println!("{}", program.pretty(100));
        }
        println!(); // Just to have a clear separation for each size's output
    }
    println!("Done!");
    println!("Total number of programs enumerated: {}", enumerator.cache.values().map(|s| s.len()).sum::<usize>());
    println!("Cache contents:");
    // for (key, value) in &enumerator.cache {
    //     println!("Key: {:?}, Value: {:?}", key, value);
    // }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_main() {
        main();
    }
}
