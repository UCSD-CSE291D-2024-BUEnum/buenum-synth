use super::ast::*;
use std::ops::*;

impl Expr {
    pub fn eval(&self, env: &EvalEnv) -> Value {
        match self {
            Expr::ConstBool(b) => Value::Bool(*b),
            Expr::ConstInt(i) => Value::Int(*i),
            Expr::ConstBitVec(bv) => Value::BitVec(*bv),
            Expr::ConstString(s) => Value::String(s.clone()),
            Expr::Var(name, _) => env.get_var(name),
            Expr::FuncApply(func_name, args) => {
                let args_val: Vec<Value> = args.iter().map(|arg| arg.eval(env)).collect();
                env.apply_func(func_name, &args_val)
            }
            Expr::Let(bindings, body) => {
                let mut new_env = env.clone();
                for (var, expr) in bindings {
                    let val = expr.eval(env);
                    new_env.set_var(var.clone(), val);
                }
                body.eval(&mut new_env)
            }
            Expr::Not(expr) => expr.eval(env).not(),
            Expr::And(left, right) => left.eval(env).and(right.eval(env)),
            Expr::Or(left, right) => left.eval(env).or(right.eval(env)),
            Expr::Xor(left, right) => left.eval(env).xor(right.eval(env)),
            Expr::Iff(left, right) => left.eval(env).iff(right.eval(env)),
            Expr::Equal(left, right) => Value::Bool(left.eval(env) == right.eval(env)),
            Expr::BvAnd(left, right) => left.eval(env).bitand(right.eval(env)),
            Expr::BvOr(left, right) => left.eval(env).bitor(right.eval(env)),
            Expr::BvXor(left, right) => left.eval(env).bitxor(right.eval(env)),
            Expr::BvNot(expr) => expr.eval(env).not(),
            Expr::BvAdd(left, right) => left.eval(env).add(right.eval(env)),
            Expr::BvMul(left, right) => left.eval(env).mul(right.eval(env)),
            Expr::BvSub(left, right) => left.eval(env).sub(right.eval(env)),
            Expr::BvUdiv(left, right) => left.eval(env).div(right.eval(env)),
            Expr::BvUrem(left, right) => left.eval(env).rem(right.eval(env)),
            Expr::BvShl(left, right) => left.eval(env).shl(right.eval(env)),
            Expr::BvLshr(left, right) => left.eval(env).shr(right.eval(env)),
            Expr::BvNeg(expr) => expr.eval(env).neg(),
            Expr::BvUlt(left, right) => left.eval(env).ult(right.eval(env)),
            Expr::BvConst(val, width) => Value::BitVec((*val as u64) & ((1 << *width) - 1)),
            _ => unimplemented!(),
        }
    }
}

#[derive(Debug, Clone, PartialEq, PartialOrd, Default)]
pub enum Value {
    Bool(bool),
    Int(i64),
    BitVec(u64),
    String(String),

    #[default]
    None,
}

macro_rules! impl_binary_op {
    ($trait:ident, $method:ident, $($t:ident),+) => {
        impl $trait for Value {
            type Output = Value;

            fn $method(self, other: Value) -> Value {
                match (self, other) {
                    $(
                        (Value::$t(a), Value::$t(b)) => Value::$t(a.$method(b)),
                    )+
                    _ => panic!(concat!("Invalid operands for '", stringify!($method), "' operation")),
                }
            }
        }
    };
}

impl_binary_op!(BitAnd, bitand, Bool, BitVec);
impl_binary_op!(BitOr, bitor, Bool, BitVec);
impl_binary_op!(BitXor, bitxor, Bool, BitVec);
impl_binary_op!(Add, add, Int, BitVec);
impl_binary_op!(Sub, sub, Int, BitVec);
impl_binary_op!(Mul, mul, Int, BitVec);
impl_binary_op!(Div, div, Int, BitVec);
impl_binary_op!(Rem, rem, Int, BitVec);

macro_rules! impl_unary_op {
    ($trait:ident, $method:ident, $($t:ident),+) => {
        impl $trait for Value {
            type Output = Value;

            fn $method(self) -> Value {
                match self {
                    $(
                        Value::$t(a) => Value::$t(a.$method()),
                    )+
                    _ => panic!(concat!("Invalid operand for '", stringify!($method), "' operation")),
                }
            }
        }
    };
}

impl_unary_op!(Not, not, Bool, BitVec);

impl Value {
    fn neg(self) -> Value {
        match self {
            Value::Int(a) => Value::Int(-a),
            Value::BitVec(a) => Value::BitVec(-(a as i64) as u64),
            _ => panic!("Invalid operand for 'neg' operation"),
        }
    }
    fn and(self, other: Value) -> Value {
        match (self, other) {
            (Value::Bool(a), Value::Bool(b)) => Value::Bool(a && b),
            _ => panic!("Invalid operands for 'and' operation"),
        }
    }
    fn or(self, other: Value) -> Value {
        match (self, other) {
            (Value::Bool(a), Value::Bool(b)) => Value::Bool(a || b),
            _ => panic!("Invalid operands for 'or' operation"),
        }
    }
    fn xor(self, other: Value) -> Value {
        match (self, other) {
            (Value::Bool(a), Value::Bool(b)) => Value::Bool(a ^ b),
            _ => panic!("Invalid operands for 'xor' operation"),
        }
    }
    fn shl(self, other: Value) -> Value {
        match (self, other) {
            (Value::BitVec(a), Value::BitVec(b)) => Value::BitVec(a << b),
            _ => panic!("Invalid operands for 'shl' operation"),
        }
    }

    fn shr(self, other: Value) -> Value {
        match (self, other) {
            (Value::BitVec(a), Value::BitVec(b)) => Value::BitVec(a >> b),
            _ => panic!("Invalid operands for 'shr' operation"),
        }
    }

    fn iff(self, other: Value) -> Value {
        match (self, other) {
            (Value::Bool(a), Value::Bool(b)) => Value::Bool(a == b),
            _ => panic!("Invalid operands for 'iff' operation"),
        }
    }

    fn ult(self, other: Value) -> Value {
        match (self, other) {
            (Value::Int(a), Value::Int(b)) => Value::Bool(a < b),
            (Value::BitVec(a), Value::BitVec(b)) => Value::Bool(a < b),
            _ => panic!("Invalid operands for 'ult' operation"),
        }
    }
}

#[derive(Debug, Clone, PartialEq, Default)]
pub struct EvalEnv {
    vars: Vec<(String, Value)>,
    funcs: Vec<(FuncName, FuncBody)>,
}

impl EvalEnv {
    fn get_var(&self, name: &str) -> Value {
        self.vars
            .iter()
            .find(|(var, _)| var == name)
            .map(|(_, val)| val.clone())
            .unwrap_or(Value::None)
    }

    fn set_var(&mut self, name: String, val: Value) {
        if let Some((_, v)) = self.vars.iter_mut().find(|(var, _)| var == &name) {
            *v = val;
        } else {
            self.vars.push((name, val));
        }
    }

    fn apply_func(&self, func_name: &str, args: &[Value]) -> Value {
        if let Some((_, func_body)) = self.funcs.iter().find(|(name, _)| name == func_name) {
            let mut func_env = self.clone();
            for ((param_name, _), arg) in func_body.params.iter().zip(args.iter()) {
                func_env.set_var(param_name.clone(), arg.clone());
            }
            func_body.body.eval(&mut func_env)
        } else {
            Value::None
        }
    }
}
