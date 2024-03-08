use std::{collections::HashMap, str::FromStr};

pub type FuncName = String;
pub type OptName = String;
pub type OptValue = String;
pub type Symbol = String;
pub type ProdName = String;
pub type ProdSort = Sort;

#[derive(Debug, Clone, PartialEq)]
pub struct SyGuSProg {
    pub set_logic: SetLogic,
    pub define_fun: HashMap<FuncName, FuncBody>,
    pub declare_var: HashMap<Symbol, Sort>,
    pub synthe_func: HashMap<FuncName, (SynthFun, GrammarDef)>,
    pub constraints: Vec<Expr>,
    pub set_option: HashMap<OptName, OptValue>,
}

impl SyGuSProg {
    pub fn new() -> SyGuSProg {
        SyGuSProg {
            set_logic: SetLogic::Unknown,
            define_fun: HashMap::new(),
            declare_var: HashMap::new(),
            synthe_func: HashMap::new(),
            constraints: Vec::new(),
            set_option: HashMap::new(),
        }
    }
}

#[derive(Debug, Clone, PartialEq)]
pub enum SetLogic {
    LIA,
    BV,

    Unknown,
}

#[derive(Debug, Clone, PartialEq)]
pub struct FuncBody {
    pub name: FuncName,
    pub params: Vec<(Symbol, Sort)>, // retain parameter order
    pub return_type: Sort,
    pub body: Expr,
}

#[derive(Debug, Clone, PartialEq)]
pub struct SynthFun {
    pub name: FuncName,
    pub params: Vec<(Symbol, Sort)>, // retain parameter order
    pub return_type: Sort,
}

#[derive(Debug, Clone, PartialEq)]
pub enum Sort {
    Bool,
    Int,
    BitVec(i32), // bit width
    String,

    None,
}

#[derive(Debug, Clone, PartialEq)]
pub enum Expr {
    ConstBool(bool),
    ConstInt(i64),
    ConstBitVec(u64),
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
}

#[derive(Debug, Clone, PartialEq)]
pub struct GrammarDef {
    pub non_terminals: Vec<Production>,
}
impl GrammarDef {
    pub fn new() -> GrammarDef {
        GrammarDef {
            non_terminals: Vec::new(),
        }
    }
}
#[derive(Debug, Clone, PartialEq)]
pub struct Production {
    pub lhs: ProdName,
    pub lhs_sort: ProdSort,
    pub rhs: Vec<GTerm>,
}
impl Production {
    pub fn new() -> Production {
        Production {
            lhs: "".to_string(),
            lhs_sort: Sort::None,
            rhs: Vec::new(),
        }
    }
}

#[derive(Debug, Clone, PartialEq)]
pub enum GTerm {
    Constant(Sort),
    Variable(Sort),
    BfTerm(GExpr),

    None,
}

#[derive(Debug, Clone, PartialEq)]
pub enum GExpr {
    ConstBool(bool),
    ConstInt(i64),
    ConstBitVec(u64),
    Var(Symbol, Sort),
    FuncApply(FuncName, Vec<GExpr>),
    Let(Vec<(Symbol, GExpr)>, Box<GExpr>),

    // Bool
    Not(Box<GExpr>),
    And(Box<GExpr>, Box<GExpr>),
    Or(Box<GExpr>, Box<GExpr>),
    Xor(Box<GExpr>, Box<GExpr>),
    Iff(Box<GExpr>, Box<GExpr>), // if and only if
    Equal(Box<GExpr>, Box<GExpr>),

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
}
