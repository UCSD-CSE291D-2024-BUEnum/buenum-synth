pub mod ast;
pub mod eval;
pub mod visitor;

use pest::error::Error;
use pest::Parser;
use pest_derive::Parser;

use self::visitor::SyGuSVisitor;
use crate::parser::visitor::Visitor;

#[derive(Parser)]
#[grammar = "parser/grammar.pest"] // relative to project `src`
pub struct SyGuSParser;

pub fn parse(input: &str) -> Result<ast::SyGuSProg, Error<Rule>> {
    let mut pairs = SyGuSParser::parse(Rule::main, input)?;
    let pair = pairs.next().unwrap();
    let mut sygus_visitor = SyGuSVisitor::default();
    let prog = sygus_visitor.visit_main(pair)?;
    Ok(sygus_visitor.sygus_prog)
}
