use std::collections::HashMap;

type FuncName = String;
type OptName = String;
type OptValue = String;
type Symbol = String;
type ProdName = String;
type ProdType = Sort;

#[derive(Debug)]
pub struct SyGuSProg {
    pub set_logic: SetLogic,
    pub define_fun: HashMap<FuncName, FuncBody>,
    pub declare_var: HashMap<Symbol, Sort>,
    pub synthe_func: HashMap<FuncName, GrammarDef>,
    pub set_option: HashMap<OptName, OptValue>,
}

#[derive(Debug)]
pub enum SetLogic {
    LIA,
    BV,

    Unknown,
}

#[derive(Debug)]
pub struct FuncBody {
    pub name: FuncName,
    pub params: Vec<(Symbol, Sort)>, // retain parameter order
    pub return_type: Sort,
    pub body: Expr,
}

#[derive(Debug)]
pub enum Sort {
    Bool,
    Int,
    BitVec(i32), // bit width

    None,
}

#[derive(Debug)]
pub enum Expr {
    ConstBool(bool),
    ConstInt(i64),
    ConstBitVec(u64),
    Var(Symbol),
    FuncApply(FuncName, Vec<Expr>),
    Let(Vec<(Symbol, Expr)>, Box<Expr>),

    // Bool
    Not(Box<Expr>),
    And(Box<Expr>, Box<Expr>),
    Or(Box<Expr>, Box<Expr>),
    Xor(Box<Expr>, Box<Expr>),
    Iff(Box<Expr>, Box<Expr>), // if and only if

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

#[derive(Debug)]
pub struct GrammarDef {
    pub non_terminals: Vec<Production>,
}
#[derive(Debug)]
pub struct Production {
    pub lhs: ProdName,
    pub lhs_type: ProdType,
    pub rhs: Vec<GTerm>,
}

#[derive(Debug)]
pub enum GTerm {
    Constant(Sort),
    Variable(Sort),
    FuncApply(Symbol, Vec<GTerm>),
}
