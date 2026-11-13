#!/usr/bin/env bash
# GPU VM (g2-standard-8, 1x NVIDIA L4, Deep Learning VM image with the NVIDIA driver):
# 1) build puggles-cuda inside Rust-CUDA's prebuilt container (CUDA 12.8 + LLVM 7);
# 2) build the standard workers on the host (wgpu GPU via the NVIDIA Vulkan driver, CubeCL);
# 3) run the full suite including the GPU, CubeCL, CUDA and torch-CUDA rows.
# Logs: ~/setup.log, ~/cuda_build.log, ~/bench.log; done marker ~/DONE.
# Idempotent, so it can simply be rerun after a Spot preemption: finished steps are skipped
# and the benchmark resumes from its saved results.
set -euxo pipefail
exec > >(tee -a ~/setup.log) 2>&1
nvidia-smi
sudo apt-get update -q
sudo DEBIAN_FRONTEND=noninteractive apt-get install -yq build-essential pkg-config libssl-dev time curl libvulkan1 docker.io
# The image ships the NVIDIA container toolkit but not Docker; wire the two together.
sudo nvidia-ctk runtime configure --runtime=docker
sudo systemctl restart docker
curl -sSf https://sh.rustup.rs | sh -s -- -y --default-toolchain 1.94.0 --profile minimal
. "$HOME/.cargo/env"
curl -LsSf https://astral.sh/uv/install.sh | sh
export PATH="$HOME/.local/bin:$PATH"
[ -d ~/puggles ] || tar xzf ~/puggles.tgz -C ~

# 1) Rust-CUDA build in its container (needs the pinned nightly + LLVM 7 the image provides).
[ -f ~/puggles/puggles-cuda/target/release/examples/bench_cuda ] || sudo docker run --rm --gpus all -v "$HOME/puggles":/workspace -w /workspace/puggles-cuda \
  -e CARGO_HOME=/cargo -v rust-cuda-cargo:/cargo -e RUSTUP_HOME=/rustup -v rust-cuda-rustup:/rustup \
  ghcr.io/rust-gpu/rust-cuda-ubuntu24-cuda12:main \
  bash -lc 'rustup show && cargo build --release --example bench_cuda' > ~/cuda_build.log 2>&1 \
  || echo "puggles-cuda build FAILED (see ~/cuda_build.log)"
sudo chown -R "$USER" "$HOME/puggles"

# 2) Host builds and Python env.
cd ~/puggles/sandbox
uv venv .venv --python 3.14 --allow-existing
uv pip install --python .venv/bin/python pymoo deap platypus-opt numpy polars psutil maturin numba jax pytest torch
( cd ../python && VIRTUAL_ENV="$HOME/puggles/sandbox/.venv" ../sandbox/.venv/bin/maturin develop --release --uv )
.venv/bin/python -I scripts/build_real_datasets.py data/real_prices.csv
md5sum data/real_returns_*.csv
cargo build --release --features gpu,cubecl --example bench_datasets --example bench_cubecl
[ -f ~/tests.log ] || ( cd .. && cargo test --release --features gpu && cargo test --release --features gpu -- --ignored gpu_evaluator ) > ~/tests.log 2>&1 \
  || echo "rust tests failed" >> ~/tests.log

# 3) Everything (the driver marks puggles-cuda "not built" if step 1 failed).
.venv/bin/python bench_datasets.py --out ~/results_gcp_g2.json --resume 1 >> ~/bench.log 2>&1
touch ~/DONE
