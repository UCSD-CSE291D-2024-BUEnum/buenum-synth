# Bottom-up Enumerative Synthesizer using Egg

## Warning: `z3` dependencies

If you fail to run `cargo build` because of `z3-sys` related errors, you may want to try the following:

1. Install `clang`:

    ```bash
    sudo apt install clang libclang-dev  # Debian/Ubuntu
    # or
    brew install llvm                    # MacOS
    ```

    After installing `clang`, you should set the `LIBCLANG_PATH` environment variable to the path where the libclang library is located. You can do this by adding the following line to your shell configuration file (e.g., `~/.bashrc` or `~/.zshrc`):

    ```bash
    export LIBCLANG_PATH=/usr/lib/llvm-<version>/lib # Debian/Ubuntu
    # or
    export LIBCLANG_PATH=$(brew --prefix llvm)/lib   # MacOS
    ```

2. Install `z3`:

    You may need to install `z3` on your system first before allow the `z3` dependency to be built.

    ```bash
    sudo apt install z3  # Debian/Ubuntu
    # or
    brew install z3      # MacOS
    ```

## Run tests

```bash
cargo test
```
