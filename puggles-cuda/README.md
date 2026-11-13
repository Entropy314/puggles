# puggles-cuda

A CUDA backend for puggles. Its GPU kernel is written in Rust using [Rust-CUDA](https://github.com/Rust-GPU/Rust-CUDA). `CudaEvaluator` implements puggles' `BatchEvaluator` trait and computes in f64.

> **Status:** written against Rust-CUDA commit `3a82c35`, but **never compiled or run**. It was written on an Apple M3, and Rust-CUDA doesn't support macOS. The first build will probably need small API fixes.

## Requirements
- Linux or Windows
- an NVIDIA GPU with compute capability 5.0 or newer
- CUDA 12+, and an NVIDIA driver that matches it
- LLVM 7.x (or 21 with the `llvm21` feature)
- the pinned nightly toolchain in `rust-toolchain.toml`, which `rustup` installs automatically

Rust-CUDA's prebuilt Docker images contain all of these.

## Why it's a separate crate
Rust-CUDA's maintained crates are only published as git dependencies, and crates.io refuses crates that use git dependencies. They also need nightly Rust. Keeping this crate outside puggles means puggles stays publishable and builds on stable Rust.

## Build and run
```bash
cd puggles-cuda
cargo build --release --example bench_cuda
./target/release/examples/bench_cuda ../sandbox/data/real_returns_1gb.csv multi puggles-cuda 2000 1
```
`sandbox/bench_datasets.py` adds a `puggles-cuda` row automatically when this binary exists.
