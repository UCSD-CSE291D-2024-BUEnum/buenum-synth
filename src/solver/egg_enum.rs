use std::collections::HashMap;
use std::vec;

use egg::{
    rewrite as rw,
    *
};
use strum_macros::Display;

use z3::{ast::Bool, Config, Context, Solver};
use z3::ast::Ast;


fn verify_expression(expr: &RecExpr<BoolLanguage>, var_map: &HashMap<String, z3::ast::Bool>, ctx: &Context, custom_fn: &dyn Fn(&HashMap<String, bool>) -> bool) -> Result<(), HashMap<String, bool>> {
    let solver = Solver::new(ctx);

    // Convert the synthesized expression to a z3 expression
    let z3_expr = convert_to_z3_expr(&expr, var_map, ctx);

    // Add the constraint that the synthesized expression should be equivalent to the custom function
    solver.assert(&z3_expr._eq(&Bool::new_const(ctx, "custom_fn")));

    if solver.check() == z3::SatResult::Sat {
        Ok(())
    } else {
        let model = solver.get_model().unwrap();
        let mut counter_example = HashMap::new();
        for (var, _) in var_map {
            let value = model.eval(&var_map[var], true).unwrap().as_bool().unwrap();
            counter_example.insert(var.clone(), value);
        }
        Err(counter_example)
    }
}

fn convert_to_z3_expr<'ctx>(expr: &'ctx RecExpr<BoolLanguage>, var_map: &'ctx HashMap<String, Bool<'ctx>>, ctx: &'ctx Context) -> Bool<'ctx> {
    match expr.as_ref().first() {
        Some(BoolLanguage::And([a, b])) => {
            let a_expr = convert_to_z3_expr(expr, var_map, ctx).clone();
            let b_expr = convert_to_z3_expr(expr, var_map, ctx).clone();
            Bool::and(&ctx, &[&a_expr, &b_expr])
        }
        Some(BoolLanguage::Or([a, b])) => {
            let a_expr = convert_to_z3_expr(expr, var_map, ctx).clone();
            let b_expr = convert_to_z3_expr(expr, var_map, ctx).clone();
            Bool::or(&ctx, &[&a_expr, &b_expr])
        }
        Some(BoolLanguage::Not([a])) => {
            let a_expr = convert_to_z3_expr(expr, var_map, ctx).clone();
            a_expr.not()
        }
        Some(BoolLanguage::Var(name)) => var_map[name].clone(),
        Some(BoolLanguage::Const(val)) => Bool::from_bool(ctx, *val),
        _ => panic!("Unexpected node type"),
    }
}

define_language! {
    pub enum ArithLanguage {
        Num(i32),
        Var(String),
        "+" = Add([Id; 2]),
        "-" = Sub([Id; 2]),
        "*" = Mul([Id; 2]),
        "|" = Or([Id; 2]),
        "&" = And([Id; 2]),
        "^" = Xor([Id; 2]),
        "<<" = Shl([Id; 2]),
        "neg" = Neg([Id; 1]),
    }
}

define_language! {
    pub enum BoolLanguage {
        "and" = And([Id; 2]),
        "or" = Or([Id; 2]),
        "not" = Not([Id; 1]),
        Var(String),
        Const(bool),
    }
}

#[derive(Debug, Clone, PartialEq, Eq, Hash, Display)]
pub enum SyGuSLanguage {
    BoolLanguage(BoolLanguage),
    ArithLanguage(ArithLanguage),
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
    LanguageConstruct(BoolLanguage)
}

#[derive(Debug, Clone)]
struct Grammar {
    productions: Vec<Production>
}

#[derive(Default, Debug, Clone)]
struct ObsEquiv {
    pts: Vec<(HashMap<String, bool>, bool)>,
}

impl Analysis<BoolLanguage> for ObsEquiv {
    // type Data = i32;
    type Data = bool;

    fn merge(&mut self, to: &mut Self::Data, from: Self::Data) -> DidMerge {
        if *to != from {
            *to = from;
            DidMerge(true, false)
        } else {
            DidMerge(false, false)
        }
    }

    fn make(egraph: &EGraph<BoolLanguage, Self>, enode: &BoolLanguage) -> Self::Data {
        let x = |i: &Id| egraph[*i].data;
        match enode {
            // ArithLanguage::Num(n) => *n,
            // ArithLanguage::Var(name) => {
            //     let mut sum = 0;
            //     for (inputs, output) in &egraph.analysis.pts {
            //         if let Some(value) = inputs.iter().find(|(var, _)| *var == name).map(|(_, val)| *val) {
            //             sum += value;
            //         }
            //     }
            //     sum
            // }
            // ArithLanguage::Neg([a]) => -x(a),
            // ArithLanguage::Add([a, b]) => x(a) + x(b),
            // ArithLanguage::Sub([a, b]) => x(a) - x(b),
            // ArithLanguage::Mul([a, b]) => x(a) * x(b),
            // ArithLanguage::Shl([a, b]) => x(a) << x(b),
            // ArithLanguage::Or([a, b]) => x(a) | x(b),
            // ArithLanguage::And([a, b]) => x(a) & x(b),
            // ArithLanguage::Xor([a, b]) => x(a) ^ x(b),
            BoolLanguage::And([a, b]) => x(a) & x(b),
            BoolLanguage::Or([a, b]) => x(a) | x(b),
            BoolLanguage::Not([a]) => !x(a),
            BoolLanguage::Var(name) => {
                let mut sum = false;
                for (inputs, output) in &egraph.analysis.pts {
                    if let Some(value) = inputs.iter().find(|(var, _)| *var == name).map(|(_, val)| *val) {
                        sum &= value;
                    }
                }
                sum
            }
            BoolLanguage::Const(n) => *n,
        }
    }

    fn modify(egraph: &mut EGraph<BoolLanguage, Self>, id: Id) {
        if let n = egraph[id].data {
            let added = egraph.add(BoolLanguage::Const(n));
            egraph.union(id, added);
        }
    }
}

struct Enumerator<'a> {
    grammar: &'a Grammar,
    runner: Runner<BoolLanguage, ObsEquiv>,
    cache: HashMap<(ProdName, usize), EGraph<BoolLanguage, ObsEquiv>>,
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

    fn grow(&mut self, pts: &[(HashMap<String, bool>, bool)]) {
        let size = self.current_size + 1;
        let mut new_expressions = HashMap::new();
    
        if size == 1 {
            for prod in &self.grammar.productions {
                let mut egraph = EGraph::new(ObsEquiv { pts: pts.to_vec() }).with_explanations_enabled();
                for component in &prod.rhs {
                    if let ProdComponent::LanguageConstruct(lang_construct) = component {
                        // let is_terminal = |lc: &ArithLanguage| match lc {
                        //     ArithLanguage::Num(_) | ArithLanguage::Var(_) => true,
                        //     _ => false,
                        // };
                        let is_terminal = |lc: &BoolLanguage| match lc {
                            BoolLanguage::Const(_) | BoolLanguage::Var(_) => true,
                            _ => false,
                        };
                        handle_language_construct!(lang_construct, egraph, new_expressions, prod, size, pts, is_terminal);
                    }
                }
            }
            pretty_egraph(&new_expressions[&("Start".to_string(), 1)]);
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
                                                // ArithLanguage::Neg(_) => {
                                                //     let mut unary_egraph = new_egraph.clone().with_explanations_enabled();
                                                //     for eclass in new_egraph.classes() {
                                                //         let new_id = unary_egraph.add_expr(&new_egraph.id_to_expr(eclass.id));
                                                //         let expr = ArithLanguage::Neg([new_id]);
                                                //         unary_egraph.add(expr);
                                                //     }
                                                //     merge_egraphs(&unary_egraph, &mut new_expressions.entry((prod.lhs.clone(), size)).or_insert(EGraph::new(ObsEquiv { pts: pts.to_vec() }).with_explanations_enabled()));
                                                // }
                                                // ArithLanguage::Add(_) => handle_binary_op!(ArithLanguage::Add, left_mapping, right_mapping, new_egraph, new_expressions, prod, size, pts),
                                                // ArithLanguage::Sub(_) => handle_binary_op!(ArithLanguage::Sub, left_mapping, right_mapping, new_egraph, new_expressions, prod, size, pts),
                                                // ArithLanguage::Mul(_) => handle_binary_op!(ArithLanguage::Mul, left_mapping, right_mapping, new_egraph, new_expressions, prod, size, pts),
                                                // ArithLanguage::Shl(_) => handle_binary_op!(ArithLanguage::Shl, left_mapping, right_mapping, new_egraph, new_expressions, prod, size, pts),
                                                // ArithLanguage::Or(_) => handle_binary_op!(ArithLanguage::Or, left_mapping, right_mapping, new_egraph, new_expressions, prod, size, pts),
                                                // ArithLanguage::Xor(_) => handle_binary_op!(ArithLanguage::Xor, left_mapping, right_mapping, new_egraph, new_expressions, prod, size, pts),
                                                // ArithLanguage::And(_) => handle_binary_op!(ArithLanguage::And, left_mapping, right_mapping, new_egraph, new_expressions, prod, size, pts),
                                                BoolLanguage::And(_) => handle_binary_op!(BoolLanguage::And, left_mapping, right_mapping, new_egraph, new_expressions, prod, size, pts),
                                                BoolLanguage::Or(_) => handle_binary_op!(BoolLanguage::Or, left_mapping, right_mapping, new_egraph, new_expressions, prod, size, pts),
                                                BoolLanguage::Not(_) => {
                                                    let mut unary_egraph = new_egraph.clone().with_explanations_enabled();
                                                    for eclass in new_egraph.classes() {
                                                        let new_id = unary_egraph.add_expr(&new_egraph.id_to_expr(eclass.id));
                                                        let expr = BoolLanguage::Not([new_id]);
                                                        unary_egraph.add(expr);
                                                    }
                                                    merge_egraphs(&unary_egraph, &mut new_expressions.entry((prod.lhs.clone(), size)).or_insert(EGraph::new(ObsEquiv { pts: pts.to_vec() }).with_explanations_enabled()));
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
            pretty_egraph(&new_expressions[&("Start".to_string(), size)]);
        }
        for (key, egraph) in new_expressions {
            self.cache.entry(key).or_insert(egraph);
        }
    
        self.current_size = size;
    }

    fn enumerate(&mut self, size: usize, pts: &[(HashMap<String, bool>, bool)]) -> Vec<RecExpr<BoolLanguage>> {
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

    fn satisfies_counter_examples(&self, expr: &RecExpr<BoolLanguage>, pts: &[(HashMap<String, bool>, bool)]) -> bool {
        for pt in pts {
            if !self.satisfies_counter_example(expr, pt) {
                return false;
            }
        }
        true
    }

    fn satisfies_counter_example(&self, expr: &RecExpr<BoolLanguage>, pt: &(HashMap<String, bool>, bool)) -> bool {
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

    fn synthesize(&self, max_size: usize, pts: &[(HashMap<String, bool>, bool)]) -> Option<RecExpr<BoolLanguage>> {
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

// fn main() {
//     let grammar = Grammar {
//         productions: vec![
//             Production {
//                 lhs: "S".to_string(),
//                 lhs_type: "Op".to_string(),
//                 rhs: vec![
//                     ProdComponent::LhsName("S".to_string()),
//                     ProdComponent::LanguageConstruct(ArithLanguage::Add(Default::default())),
//                     ProdComponent::LhsName("S".to_string()),
//                 ]
//             },
//             Production {
//                 lhs: "S".to_string(),
//                 lhs_type: "Op".to_string(),
//                 rhs: vec![
//                     ProdComponent::LhsName("S".to_string()),
//                     ProdComponent::LanguageConstruct(ArithLanguage::Sub(Default::default())),
//                     ProdComponent::LhsName("S".to_string()),
//                 ]
//             },
//             Production {
//                 lhs: "S".to_string(),
//                 lhs_type: "Op".to_string(),
//                 rhs: vec![
//                     ProdComponent::LhsName("S".to_string()),
//                     ProdComponent::LanguageConstruct(ArithLanguage::Mul(Default::default())),
//                     ProdComponent::LhsName("S".to_string()),
//                 ]
//             },
//             Production {
//                 lhs: "S".to_string(),
//                 lhs_type: "Op".to_string(),
//                 rhs: vec![
//                     ProdComponent::LhsName("S".to_string()),
//                     ProdComponent::LanguageConstruct(ArithLanguage::Shl(Default::default())),
//                     ProdComponent::LhsName("S".to_string()),
//                 ]
//             },
//             Production {
//                 lhs: "S".to_string(),
//                 lhs_type: "Op".to_string(),
//                 rhs: vec![
//                     ProdComponent::LhsName("S".to_string()),
//                     ProdComponent::LanguageConstruct(ArithLanguage::And(Default::default())),
//                     ProdComponent::LhsName("S".to_string()),
//                 ]
//             },
//             Production {
//                 lhs: "S".to_string(),
//                 lhs_type: "Op".to_string(),
//                 rhs: vec![
//                     ProdComponent::LhsName("S".to_string()),
//                     ProdComponent::LanguageConstruct(ArithLanguage::Or(Default::default())),
//                     ProdComponent::LhsName("S".to_string()),
//                 ]
//             },
//             Production {
//                 lhs: "S".to_string(),
//                 lhs_type: "Op".to_string(),
//                 rhs: vec![
//                     ProdComponent::LhsName("S".to_string()),
//                     ProdComponent::LanguageConstruct(ArithLanguage::Xor(Default::default())),
//                     ProdComponent::LhsName("S".to_string()),
//                 ]
//             },
//             // add unary op
//             Production {
//                 lhs: "S".to_string(),
//                 lhs_type: "Op".to_string(),
//                 rhs: vec![
//                     ProdComponent::LanguageConstruct(ArithLanguage::Neg(Default::default())),
//                     ProdComponent::LhsName("S".to_string()),
//                 ]
//             },
//             Production {
//                 lhs: "S".to_string(),
//                 lhs_type: "Var".to_string(),
//                 rhs: vec![
//                     ProdComponent::LanguageConstruct(ArithLanguage::Var("x".to_string())),
//                     ProdComponent::LanguageConstruct(ArithLanguage::Var("y".to_string())),
//                 ]
//             },
//             Production {
//                 lhs: "S".to_string(),
//                 lhs_type: "Num".to_string(),
//                 rhs: vec![ProdComponent::LanguageConstruct(ArithLanguage::Num(0))]
//             },
//             Production {
//                 lhs: "S".to_string(),
//                 lhs_type: "Num".to_string(),
//                 rhs: vec![ProdComponent::LanguageConstruct(ArithLanguage::Num(1))]
//             },
//             Production {
//                 lhs: "S".to_string(),
//                 lhs_type: "Num".to_string(),
//                 rhs: vec![ProdComponent::LanguageConstruct(ArithLanguage::Num(2))]
//             },
//         ]
//     };

//     let solver = EggSolver::new(grammar);
//     let max_size = 4;

//     let pts = vec![ // x * 2 + y
//         (HashMap::from([("x".to_string(), 1), ("y".to_string(), 2)]), 4),
//         (HashMap::from([("x".to_string(), 3), ("y".to_string(), 4)]), 10),
//         (HashMap::from([("x".to_string(), 4), ("y".to_string(), 1)]), 9),
//     ];

//     if let Some(expr) = solver.synthesize(max_size, &pts) {
//         println!("Synthesized expression: {}", expr.pretty(100));
//     } else {
//         println!("No expression could be synthesized.");
//     }
// }
#[test]
fn test_bool_1() {
    let grammar = Grammar {
        productions: vec![
            Production {
                lhs: "Start".to_string(),
                lhs_type: "Bool".to_string(),
                rhs: vec![
                    ProdComponent::LhsName("Start".to_string()),
                    ProdComponent::LanguageConstruct(BoolLanguage::And(Default::default())),
                    ProdComponent::LhsName("Start".to_string()),
                ]
            },
            Production {
                lhs: "Start".to_string(),
                lhs_type: "Bool".to_string(),
                rhs: vec![
                    ProdComponent::LanguageConstruct(BoolLanguage::Not(Default::default())),
                    ProdComponent::LhsName("Start".to_string()),
                ]
            },
            Production {
                lhs: "Start".to_string(),
                lhs_type: "Bool".to_string(),
                rhs: vec![ProdComponent::LanguageConstruct(BoolLanguage::Var("a".to_string()))]
            },
            Production {
                lhs: "Start".to_string(),
                lhs_type: "Bool".to_string(),
                rhs: vec![ProdComponent::LanguageConstruct(BoolLanguage::Var("b".to_string()))]
            },
        ]
    };

    let custom_fn = |inputs: &HashMap<String, bool>| inputs["a"] | inputs["b"];

    let solver = EggSolver::new(grammar);
    let max_size = 5;
    let mut pts = vec![];

    if let Some(expr) = solver.synthesize(max_size, &pts) {
        println!("Found matching expression: {}", expr.pretty(100));
    } else {
        println!("No matching expression found.");
    }
}

#[test]
fn test_bool_2() {
    let grammar = Grammar {
        productions: vec![
            Production {
                lhs: "Start".to_string(),
                lhs_type: "Bool".to_string(),
                rhs: vec![
                    ProdComponent::LhsName("Start".to_string()),
                    ProdComponent::LanguageConstruct(BoolLanguage::And(Default::default())),
                    ProdComponent::LhsName("Start".to_string()),
                ]
            },
            Production {
                lhs: "Start".to_string(),
                lhs_type: "Bool".to_string(),
                rhs: vec![
                    ProdComponent::LanguageConstruct(BoolLanguage::Not(Default::default())),
                    ProdComponent::LhsName("Start".to_string()),
                ]
            },
            Production {
                lhs: "Start".to_string(),
                lhs_type: "Bool".to_string(),
                rhs: vec![ProdComponent::LanguageConstruct(BoolLanguage::Var("a".to_string()))]
            },
            Production {
                lhs: "Start".to_string(),
                lhs_type: "Bool".to_string(),
                rhs: vec![ProdComponent::LanguageConstruct(BoolLanguage::Var("b".to_string()))]
            },
            Production {
                lhs: "Start".to_string(),
                lhs_type: "Bool".to_string(),
                rhs: vec![ProdComponent::LanguageConstruct(BoolLanguage::Var("c".to_string()))]
            },
        ]
    };

    let custom_fn = |inputs: &HashMap<String, bool>| {
        let a = inputs["a"];
        let b = inputs["b"];
        let c = inputs["c"];
        (a | b) | c
    };

    let solver = EggSolver::new(grammar);
    let max_size = 7;
    let mut pts = vec![];

    if let Some(expr) = solver.synthesize(max_size, &pts) {
        println!("Found matching expression: {}", expr.pretty(100));
    } else {
        println!("No matching expression found.");
    }
}