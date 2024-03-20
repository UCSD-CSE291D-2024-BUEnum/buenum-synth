use std::collections::HashMap;
use std::process::exit;
use std::vec;

use egg::{
    rewrite as rw,
    *
};
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

fn verify(
    expr: &RecExpr<BoolLanguage>,
    var_map: &HashMap<String, z3::ast::Bool>,
    ctx: &Context,
    custom_fn: &dyn Fn(&HashMap<String, bool>) -> bool
) -> Result<(), HashMap<String, bool>> {
    let solver = Solver::new(ctx);

    // Convert the synthesized expression to a Z3 expression
    let expr_str = expr.pretty(100);
    // println!("Synthesized expression: {}", expr_str);
    let z3_expr = Dynamic::from(convert_to_z3_expr(&expr, var_map, ctx));

    // Create a Z3 function declaration for the custom function
    let custom_fn_decl = {
        let mut domain = Vec::new();
        for (var, _) in var_map {
            domain.push(var_map[var].get_sort());
        }
        let domain_refs: Vec<&z3::Sort> = domain.iter().collect();
        z3::FuncDecl::new(ctx, "custom_fn", &domain_refs, &z3::Sort::bool(ctx))
    };

    // Create Z3 variables for the custom function arguments
    let mut custom_fn_args = Vec::new();
    for (var, _) in var_map {
        custom_fn_args.push(var_map[var].clone());
    }
    let custom_fn_args_refs: Vec<&dyn z3::ast::Ast> =
        custom_fn_args.iter().map(|arg| arg as &dyn z3::ast::Ast).collect();

    // Assert that the synthesized expression is not equal to the custom function
    solver.assert(&z3_expr._eq(&custom_fn_decl.apply(&custom_fn_args_refs)).not());

    match solver.check() {
        z3::SatResult::Unsat => Ok(()), // no counter-example found
        z3::SatResult::Unknown => panic!("Unknown z3 solver result"),
        z3::SatResult::Sat => {
            let model = solver.get_model().unwrap();
            let mut counter_example = HashMap::new();
            for (var, _) in var_map {
                let value = model.eval(&var_map[var], true).unwrap().as_bool().unwrap();
                counter_example.insert(var.clone(), value);
            }
            Err(counter_example)
        }
    }
}
fn convert_to_z3_expr<'ctx>(
    expr: &'ctx RecExpr<BoolLanguage>,
    var_map: &'ctx HashMap<String, Bool<'ctx>>,
    ctx: &'ctx Context
) -> Bool<'ctx> {
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
        _ => panic!("Unexpected node type")
    }
}

// define_language! {
//     pub enum ArithLanguage {
//         Num(i32),
//         Var(String),
//         "+" = Add([Id; 2]),
//         "-" = Sub([Id; 2]),
//         "*" = Mul([Id; 2]),
//         "|" = Or([Id; 2]),
//         "&" = And([Id; 2]),
//         "^" = Xor([Id; 2]),
//         "<<" = Shl([Id; 2]),
//         "neg" = Neg([Id; 1]),
//     }
// }

define_language! {
    pub enum BoolLanguage {
        "and" = And([Id; 2]),
        "or" = Or([Id; 2]),
        "not" = Not([Id; 1]),
        Var(String),
        Const(bool),
    }
}

// #[derive(Debug, Clone, PartialEq, Eq, Hash, Display)]
// pub enum SyGuSLanguage {
//     BoolLanguage(BoolLanguage),
//     ArithLanguage(ArithLanguage)
// }

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
    pts: Vec<(HashMap<String, bool>, bool)>
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
            BoolLanguage::Const(n) => *n
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
    // runner: Runner<BoolLanguage, ObsEquiv>,
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
where
    L: std::fmt::Display
{
    println!(
        "  EGraph size: [{}, {}]",
        egraph.number_of_classes(),
        egraph.total_number_of_nodes()
    );
    for class in egraph.classes() {
        println!(
            "    Class {}: {:?} [{}]",
            class.id,
            egraph[class.id].data,
            egraph.id_to_expr(class.id).pretty(80)
        );
        for node in class.nodes.iter() {
            println!("      Node: {:?}", node);
        }
    }
    println!();
}

macro_rules! handle_language_construct {
    ($lang_construct:expr, $egraph:ident, $new_expressions:ident, $prod:ident, $size:ident, $pts:ident, $is_terminal:expr) => {
        if $is_terminal($lang_construct) {
            $egraph.add($lang_construct.clone());
            merge_egraphs(
                &$egraph,
                &mut $new_expressions
                    .entry(($prod.lhs.clone(), $size))
                    .or_insert(EGraph::new(ObsEquiv { pts: $pts.to_vec() }).with_explanations_enabled())
            );
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
                merge_egraphs(
                    &$new_egraph,
                    &mut $new_expressions
                        .entry(($prod.lhs.clone(), $size))
                        .or_insert(EGraph::new(ObsEquiv { pts: $pts.to_vec() }).with_explanations_enabled())
                );
            }
        }
    };
}

impl<'a> Enumerator<'a> {
    fn new(grammar: &'a Grammar) -> Self {
        Enumerator {
            grammar,
            // runner: Runner::default(),
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
                        match lang_construct {
                            BoolLanguage::Const(val) => {
                                // println!("Before update: {:?}", &egraph);
                                egraph.add(lang_construct.clone());
                                // println!("After update: {:?}", &egraph);
                            }
                            BoolLanguage::Var(val) => {
                                // println!("Before update: {:?}", &egraph);
                                egraph.add(lang_construct.clone());
                                // println!("After update: {:?}", &egraph);
                            }
                            _ => {} // all other language constructs will be size > 1
                        }
                        merge_egraphs(
                            &egraph,
                            &mut new_expressions
                                .entry((prod.lhs.clone(), size))
                                .or_insert(EGraph::new(ObsEquiv { pts: pts.to_vec() }).with_explanations_enabled())
                        );
                    }
                }
            }
            println!("Size 1 expressions:");
            pretty_egraph(&new_expressions[&("Start".to_string(), 1)]);
        } else {
            for prod in &self.grammar.productions {
                // if unary op
                // ==1 for terminal,
                // ==2 for unary op,
                // ==3 for binary op
                let prod_len = prod.rhs.len();
                dbg!(prod_len);
                if prod.rhs.len() == 2 {
                    let right_size = size - 1;
                    if let Some(right_egraph) = self.cache.get(&(prod.lhs.clone(), right_size)) {
                        for right_enode in right_egraph.classes() {
                            let mut new_egraph = right_egraph.clone().with_explanations_enabled();
                            let right_mapping: HashMap<_, _> = right_egraph
                                .classes()
                                .map(|c| (c.id, new_egraph.add_expr(&right_egraph.id_to_expr(c.id))))
                                .collect();
                            for component in &prod.rhs {
                                if let ProdComponent::LanguageConstruct(lang_construct) = component {
                                    match lang_construct {
                                        BoolLanguage::Not(_) => {
                                            let mut unary_egraph = new_egraph.clone().with_explanations_enabled();
                                            for eclass in new_egraph.classes() {
                                                let new_id = unary_egraph.add_expr(&new_egraph.id_to_expr(eclass.id));
                                                let expr = BoolLanguage::Not([new_id]);
                                                unary_egraph.add(expr);
                                            }

                                            merge_egraphs(
                                                &unary_egraph,
                                                &mut new_expressions.entry((prod.lhs.clone(), size)).or_insert(
                                                    EGraph::new(ObsEquiv { pts: pts.to_vec() })
                                                        .with_explanations_enabled()
                                                )
                                            );
                                        }
                                        _ => {}
                                    }
                                }
                            }
                        }
                    }
                } else {
                    for left_size in 1..size {
                        let right_size = size - left_size;
                        if let Some(left_egraph) = self.cache.get(&(prod.lhs.clone(), left_size)) {
                            if let Some(right_egraph) = self.cache.get(&(prod.lhs.clone(), right_size)) {
                                for left_enode in left_egraph.classes() {
                                    for right_enode in right_egraph.classes() {
                                        // println!("Merging left and right egraphs:");
                                        let mut new_egraph = left_egraph.clone().with_explanations_enabled();
                                        let left_mapping: HashMap<_, _> = left_egraph
                                            .classes()
                                            .map(|c| (c.id, new_egraph.add_expr(&left_egraph.id_to_expr(c.id))))
                                            .collect();
                                        let right_mapping: HashMap<_, _> = right_egraph
                                            .classes()
                                            .map(|c| (c.id, new_egraph.add_expr(&right_egraph.id_to_expr(c.id))))
                                            .collect();
                                        // println!("Left egraph:");
                                        // pretty_egraph(&left_egraph);
                                        // println!("Right egraph:");
                                        // pretty_egraph(&right_egraph);
                                        // println!("Merging...");

                                        merge_egraphs(right_egraph, &mut new_egraph);

                                        // println!("Merged egraph:");
                                        // pretty_egraph(&new_egraph);

                                        for component in &prod.rhs {
                                            if let ProdComponent::LanguageConstruct(lang_construct) = component {
                                                match lang_construct {
                                                    BoolLanguage::And(_) => {
                                                        for &left_id in left_mapping.keys() {
                                                            for &right_id in right_mapping.keys() {
                                                                let new_left_id = left_mapping[&left_id];
                                                                let new_right_id = right_mapping[&right_id];
                                                                dbg!(new_left_id, new_right_id);
                                                                let expr =
                                                                    BoolLanguage::And([new_left_id, new_right_id]);

                                                                new_egraph.add(expr);

                                                                let prev_egraph = new_expressions
                                                                    .get(&(prod.lhs.clone(), size - 1))
                                                                    .unwrap_or(
                                                                        &EGraph::new(ObsEquiv { pts: pts.to_vec() })
                                                                            .with_explanations_enabled()
                                                                    )
                                                                    .clone();

                                                                merge_egraphs(
                                                                    &new_egraph,
                                                                    &mut new_expressions
                                                                        .entry((prod.lhs.clone(), size))
                                                                        .or_insert(prev_egraph)
                                                                );
                                                            }
                                                        }
                                                    }
                                                    BoolLanguage::Or(_) => {
                                                        for &left_id in left_mapping.keys() {
                                                            for &right_id in right_mapping.keys() {
                                                                let new_left_id = left_mapping[&left_id];
                                                                let new_right_id = right_mapping[&right_id];
                                                                let expr =
                                                                    BoolLanguage::Or([new_left_id, new_right_id]);

                                                                new_egraph.add(expr);

                                                                let prev_egraph = new_expressions
                                                                    .get(&(prod.lhs.clone(), size - 1))
                                                                    .unwrap_or(
                                                                        &EGraph::new(ObsEquiv { pts: pts.to_vec() })
                                                                            .with_explanations_enabled()
                                                                    )
                                                                    .clone();

                                                                merge_egraphs(
                                                                    &new_egraph,
                                                                    &mut new_expressions
                                                                        .entry((prod.lhs.clone(), size))
                                                                        .or_insert(prev_egraph)
                                                                );
                                                            }
                                                        }
                                                    }
                                                    _ => {}
                                                }
                                            } // do one derivation, then collect from cache
                                        }
                                    }
                                }
                            }
                        }
                    }
                }
            }
            println!("Size {} expressions:", size);
            pretty_egraph(&new_expressions[&("Start".to_string(), size)]);
        }
        for (key, egraph) in new_expressions {
            let mut cached_egraph = self.cache.entry(key).or_insert(egraph.clone());
            merge_egraphs(&egraph, &mut cached_egraph);
        }

        for (key, egraph) in &self.cache {
            println!("Cached egraph: {:?}", key);
            pretty_egraph(egraph);
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
            println!("Checking counter-example: {:?}", pt);
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
    grammar: Grammar
}

impl EggSolver {
    fn new(grammar: Grammar) -> Self {
        EggSolver { grammar }
    }

    fn synthesize(
        &self,
        max_size: usize,
        pts: &mut Vec<(HashMap<String, bool>, bool)>,
        var_names: &[String],
        custom_fn: &dyn Fn(&HashMap<String, bool>) -> bool
    ) -> Option<RecExpr<BoolLanguage>> {
        let mut enumerator = Enumerator::new(&self.grammar);
        let cfg = Config::new();
        let ctx = Context::new(&cfg);

        for size in 1..=max_size {
            let exprs = enumerator.enumerate(size, pts);
            // dbg!(&exprs);
            for expr in exprs {
                // println!("Checking expression: {}", expr.pretty(100));
                let var_map: HashMap<String, Bool> = var_names
                    .iter()
                    .map(|var| (var.clone(), Bool::new_const(&ctx, var.as_str())))
                    .collect();
                if let Err(counter_example) = verify(&expr, &var_map, &ctx, custom_fn) {
                    pts.push((counter_example.clone(), custom_fn(&counter_example)));
                } else {
                    return Some(expr);
                }
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

#[cfg(test)]
mod tests {
    use std::vec;

    use egg::{
        rewrite as rw,
        *
    };

    use super::*;

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
                    rhs: vec![ProdComponent::LanguageConstruct(BoolLanguage::Var(
                        "a".to_string()
                    ))]
                },
                Production {
                    lhs: "Start".to_string(),
                    lhs_type: "Bool".to_string(),
                    rhs: vec![ProdComponent::LanguageConstruct(BoolLanguage::Var(
                        "b".to_string()
                    ))]
                },
            ]
        }; // correct: ((Start Bool ((and Start Start) (not Start) a b))))

        let custom_fn = |inputs: &HashMap<String, bool>| inputs["a"] | inputs["b"];

        let var_names = vec!["a".to_string(), "b".to_string()];
        let solver = EggSolver::new(grammar);
        let max_size = 5;
        let mut pts = vec![];

        if let Some(expr) = solver.synthesize(max_size, &mut pts, &var_names, &custom_fn) {
            println!("Found matching expression: {}", expr.pretty(100));
        } else {
            println!("No matching expression found.");
            assert!(false)
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
                    rhs: vec![ProdComponent::LanguageConstruct(BoolLanguage::Var(
                        "a".to_string()
                    ))]
                },
                Production {
                    lhs: "Start".to_string(),
                    lhs_type: "Bool".to_string(),
                    rhs: vec![ProdComponent::LanguageConstruct(BoolLanguage::Var(
                        "b".to_string()
                    ))]
                },
                Production {
                    lhs: "Start".to_string(),
                    lhs_type: "Bool".to_string(),
                    rhs: vec![ProdComponent::LanguageConstruct(BoolLanguage::Var(
                        "c".to_string()
                    ))]
                },
                // Production {
                //     lhs: "Start".to_string(),
                //     lhs_type: "Bool".to_string(),
                //     rhs: vec![ProdComponent::LanguageConstruct(BoolLanguage::Const(
                //         true
                //     ))]
                // },
                // Production {
                //     lhs: "Start".to_string(),
                //     lhs_type: "Bool".to_string(),
                //     rhs: vec![ProdComponent::LanguageConstruct(BoolLanguage::Const(
                //         false
                //     ))]
                // },
            ]
        };

        let custom_fn = |inputs: &HashMap<String, bool>| {
            let a = inputs["a"];
            let b = inputs["b"];
            let c = inputs["c"];
            (a | b) | c
        };

        let var_names = vec!["a".to_string(), "b".to_string(), "c".to_string()];
        let solver = EggSolver::new(grammar);
        let max_size = 7;
        let mut pts = vec![];

        if let Some(expr) = solver.synthesize(max_size, &mut pts, &var_names, &custom_fn) {
            println!("Found matching expression: {}", expr.pretty(100));
        } else {
            println!("No matching expression found.");
            assert!(false);
        }
    }
}
