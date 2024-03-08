pub mod baseline_solver;
pub mod egg_solver;

trait Solver {
    type SolverInput; // input is a parsed program (SyGuSProg)
    type SolverOutput; // returns a actual program (Expr)

    fn parse(input: &str) -> Self::SolverInput;

    fn solve(input: Self::SolverInput) -> Self::SolverOutput;

    fn print(output: Self::SolverOutput) -> String;

    // fn solve_constraint(input: Self::Input) -> Result<_>;

    // fn enumerate_programs(input: Self::Input) -> Result<_>;
}
