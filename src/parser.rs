mod ast;
mod visitor;

use pest::Parser;
use pest_derive::Parser;
use pest::error::Error;

#[derive(Parser)]
#[grammar = "parser/grammar.pest"] // relative to project `src`
pub struct SyGuSParser;

pub fn parse(input: &str) -> Result<ast::SyGuS, Error<Rule>> {
    let mut pairs = SyGuSParser::parse(Rule::main, input)?;
    let pair = pairs.next().unwrap();
    visitor::visit_main(pair)
}
