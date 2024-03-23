use std::collections::HashMap;

pub type FuncName = String;
pub type OptName = String;
pub type OptValue = String;
pub type Symbol = String;
pub type ProdName = String;
pub type ProdSort = Sort;

#[derive(Debug, Clone, PartialEq, Default)]
pub struct SyGuSProg {
    pub set_logic: SetLogic,
    pub define_fun: HashMap<FuncName, FuncBody>,
    pub synth_func: HashMap<FuncName, (SynthFun, GrammarDef)>,
    pub declare_var: HashMap<Symbol, Sort>,
    pub constraints: Vec<Expr>,
    pub set_option: HashMap<OptName, OptValue>,
    pub check_synth: bool,
}

#[derive(Debug, Clone, PartialEq, Default)]
pub enum SetLogic {
    LIA,
    BV,
    #[default]
    Unknown,
}

#[derive(Debug, Clone, PartialEq, Default)]
pub struct FuncBody {
    pub name: FuncName,
    pub params: Vec<(Symbol, Sort)>, // retain parameter order
    pub ret_sort: Sort,
    pub body: Expr,
}

#[derive(Debug, Clone, PartialEq, Default)]
pub struct SynthFun {
    pub name: FuncName,
    pub params: Vec<(Symbol, Sort)>, // retain parameter order
    pub ret_sort: Sort,
}

#[derive(Debug, Clone, PartialEq, Default)]
pub enum Sort {
    Bool,
    Int,
    BitVec(i32), // bit width
    Compound(String, Vec<Sort>),
    String,

    #[default]
    None,
}

#[derive(Debug, Clone, PartialEq, Default)]
pub enum Expr {
    ConstBool(bool),
    ConstInt(i64),
    ConstBitVec(i64),
    ConstString(String),
    Var(Symbol, Sort),
    FuncApply(FuncName, Vec<Expr>),
    Let(Vec<(Symbol, Expr)>, Box<Expr>),

    // Bool
    Not(Box<Expr>),
    And(Box<Expr>, Box<Expr>),
    Or(Box<Expr>, Box<Expr>),
    Xor(Box<Expr>, Box<Expr>),
    Iff(Box<Expr>, Box<Expr>), // if and only if
    Equal(Box<Expr>, Box<Expr>),
    Ite(Box<Expr>, Box<Expr>, Box<Expr>), // if-then-else

    // BitVec
    BvAnd(Box<Expr>, Box<Expr>),
    BvOr(Box<Expr>, Box<Expr>),
    BvXor(Box<Expr>, Box<Expr>),
    BvNot(Box<Expr>),
    BvAdd(Box<Expr>, Box<Expr>),
    BvMul(Box<Expr>, Box<Expr>),
    BvSub(Box<Expr>, Box<Expr>),
    BvUdiv(Box<Expr>, Box<Expr>), // Unsigned division
    BvUrem(Box<Expr>, Box<Expr>), // Unsigned remainder
    BvShl(Box<Expr>, Box<Expr>),  // Logical shift left
    BvLshr(Box<Expr>, Box<Expr>), // Logical shift right
    BvNeg(Box<Expr>),             // Negation
    BvUlt(Box<Expr>, Box<Expr>),  // Unsigned less than
    BvConst(i64, i32),            // param1: value, param2: bit width

    #[default]
    UnknownExpr,
}

impl GExpr {
    pub fn to_expr(&self) -> Expr {
        match self {
            GExpr::ConstBool(b) => Expr::ConstBool(*b),
            GExpr::ConstInt(i) => Expr::ConstInt(*i),
            GExpr::ConstBitVec(u) => Expr::ConstBitVec(*u),
            GExpr::ConstString(s) => Expr::ConstString(s.clone()),
            GExpr::Var(sym, sort) => Expr::Var(sym.clone(), sort.clone()),
            GExpr::GFuncApply(prod_name, gexprs) => {
                let exprs = gexprs.iter().map(|gexpr| gexpr.to_expr()).collect();
                Expr::FuncApply(prod_name.clone(), exprs)
            }
            GExpr::FuncApply(func_name, gexprs) => {
                let exprs = gexprs.iter().map(|gexpr| gexpr.to_expr()).collect();
                Expr::FuncApply(func_name.clone(), exprs)
            }
            GExpr::Let(bindings, gexpr) => {
                let binds = bindings
                    .iter()
                    .map(|(sym, gexpr)| (sym.clone(), gexpr.to_expr()))
                    .collect();
                Expr::Let(binds, Box::new(gexpr.to_expr()))
            }
            GExpr::Not(gexpr) => Expr::Not(Box::new(gexpr.to_expr())),
            GExpr::And(gexpr1, gexpr2) => Expr::And(Box::new(gexpr1.to_expr()), Box::new(gexpr2.to_expr())),
            GExpr::Or(gexpr1, gexpr2) => Expr::Or(Box::new(gexpr1.to_expr()), Box::new(gexpr2.to_expr())),
            GExpr::Xor(gexpr1, gexpr2) => Expr::Xor(Box::new(gexpr1.to_expr()), Box::new(gexpr2.to_expr())),
            GExpr::Iff(gexpr1, gexpr2) => Expr::Iff(Box::new(gexpr1.to_expr()), Box::new(gexpr2.to_expr())),
            GExpr::Equal(gexpr1, gexpr2) => Expr::Equal(Box::new(gexpr1.to_expr()), Box::new(gexpr2.to_expr())),
            GExpr::Ite(gexpr1, gexpr2, gexpr3) => Expr::Ite(
                Box::new(gexpr1.to_expr()),
                Box::new(gexpr2.to_expr()),
                Box::new(gexpr3.to_expr()),
            ),
            GExpr::BvAnd(gexpr1, gexpr2) => Expr::BvAnd(Box::new(gexpr1.to_expr()), Box::new(gexpr2.to_expr())),
            GExpr::BvOr(gexpr1, gexpr2) => Expr::BvOr(Box::new(gexpr1.to_expr()), Box::new(gexpr2.to_expr())),
            GExpr::BvXor(gexpr1, gexpr2) => Expr::BvXor(Box::new(gexpr1.to_expr()), Box::new(gexpr2.to_expr())),
            GExpr::BvNot(gexpr) => Expr::BvNot(Box::new(gexpr.to_expr())),
            GExpr::BvAdd(gexpr1, gexpr2) => Expr::BvAdd(Box::new(gexpr1.to_expr()), Box::new(gexpr2.to_expr())),
            GExpr::BvMul(gexpr1, gexpr2) => Expr::BvMul(Box::new(gexpr1.to_expr()), Box::new(gexpr2.to_expr())),
            GExpr::BvSub(gexpr1, gexpr2) => Expr::BvSub(Box::new(gexpr1.to_expr()), Box::new(gexpr2.to_expr())),
            GExpr::BvUdiv(gexpr1, gexpr2) => Expr::BvUdiv(Box::new(gexpr1.to_expr()), Box::new(gexpr2.to_expr())),
            GExpr::BvUrem(gexpr1, gexpr2) => Expr::BvUrem(Box::new(gexpr1.to_expr()), Box::new(gexpr2.to_expr())),
            GExpr::BvShl(gexpr1, gexpr2) => Expr::BvShl(Box::new(gexpr1.to_expr()), Box::new(gexpr2.to_expr())),
            GExpr::BvLshr(gexpr1, gexpr2) => Expr::BvLshr(Box::new(gexpr1.to_expr()), Box::new(gexpr2.to_expr())),
            GExpr::BvNeg(gexpr) => Expr::BvNeg(Box::new(gexpr.to_expr())),
            GExpr::BvUlt(gexpr1, gexpr2) => Expr::BvUlt(Box::new(gexpr1.to_expr()), Box::new(gexpr2.to_expr())),
            GExpr::BvConst(i, j) => Expr::BvConst(*i, *j),
            GExpr::UnknownGExpr => Expr::UnknownExpr,
        }
    }
}

#[derive(Debug, Clone, PartialEq, Default)]
pub struct GrammarDef {
    pub non_terminals: Vec<Production>,
}

#[derive(Debug, Clone, PartialEq, Default)]
pub struct Production {
    pub lhs: ProdName,
    pub lhs_sort: ProdSort,
    pub rhs: Vec<GTerm>,
}

#[derive(Debug, Clone, PartialEq, Default)]
pub enum GTerm {
    Constant(Sort),
    Variable(Sort),
    BfTerm(GExpr),

    #[default]
    None,
}

#[derive(Debug, Clone, PartialEq, Default)]
pub enum GExpr {
    ConstBool(bool),
    ConstInt(i64),
    ConstBitVec(i64),
    ConstString(String),
    Var(Symbol, Sort),
    GFuncApply(ProdName, Vec<GExpr>),
    FuncApply(FuncName, Vec<GExpr>),
    Let(Vec<(Symbol, GExpr)>, Box<GExpr>),

    // Bool
    Not(Box<GExpr>),
    And(Box<GExpr>, Box<GExpr>),
    Or(Box<GExpr>, Box<GExpr>),
    Xor(Box<GExpr>, Box<GExpr>),
    Iff(Box<GExpr>, Box<GExpr>), // if and only if
    Equal(Box<GExpr>, Box<GExpr>),
    Ite(Box<GExpr>, Box<GExpr>, Box<GExpr>), // if-then-else

    // BitVec
    BvAnd(Box<GExpr>, Box<GExpr>),
    BvOr(Box<GExpr>, Box<GExpr>),
    BvXor(Box<GExpr>, Box<GExpr>),
    BvNot(Box<GExpr>),
    BvAdd(Box<GExpr>, Box<GExpr>),
    BvMul(Box<GExpr>, Box<GExpr>),
    BvSub(Box<GExpr>, Box<GExpr>),
    BvUdiv(Box<GExpr>, Box<GExpr>), // Unsigned division
    BvUrem(Box<GExpr>, Box<GExpr>), // Unsigned remainder
    BvShl(Box<GExpr>, Box<GExpr>),  // Logical shift left
    BvLshr(Box<GExpr>, Box<GExpr>), // Logical shift right
    BvNeg(Box<GExpr>),              // Negation
    BvUlt(Box<GExpr>, Box<GExpr>),  // Unsigned less than
    BvConst(i64, i32),              // param1: value, param2: bit width

    #[default]
    UnknownGExpr,
}
