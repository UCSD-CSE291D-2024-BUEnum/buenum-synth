pub mod baseline_solver;
pub mod egg_solver;

trait Solver {
    type Input;
    type Output;

    fn solve(input: Self::Input) -> Self::Output;

    fn parse(input: &str) -> Self::Input;

    fn print(output: Self::Output) -> String;

    // fn solve_constraint(input: Self::Input) -> Result<_>;

    // fn enumerate_programs(input: Self::Input) -> Result<_>;
}
