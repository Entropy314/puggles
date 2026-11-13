#!/usr/bin/env bash
# Setup + full CPU benchmark on a fresh Ubuntu 24.04 VM. Logs: ~/setup.log, ~/bench.log; done marker ~/DONE.
set -euxo pipefail
exec > >(tee -a ~/setup.log) 2>&1
sudo apt-get update -q
sudo DEBIAN_FRONTEND=noninteractive apt-get install -yq build-essential pkg-config libssl-dev time curl
curl -sSf https://sh.rustup.rs | sh -s -- -y --default-toolchain 1.94.0 --profile minimal
. "$HOME/.cargo/env"
curl -LsSf https://astral.sh/uv/install.sh | sh
export PATH="$HOME/.local/bin:$PATH"
tar xzf ~/puggles.tgz -C ~
cd ~/puggles/sandbox
uv venv .venv --python 3.14
uv pip install --python .venv/bin/python pymoo deap platypus-opt numpy polars psutil maturin numba jax pytest
uv pip install --python .venv/bin/python torch --index-url https://download.pytorch.org/whl/cpu
( cd ../python && VIRTUAL_ENV="$HOME/puggles/sandbox/.venv" ../sandbox/.venv/bin/maturin develop --release --uv )
.venv/bin/python -I scripts/build_real_datasets.py data/real_prices.csv
md5sum data/real_returns_*.csv
cargo build --release --features gpu --example bench_datasets
( cd .. && cargo test --release ) > ~/tests.log 2>&1 || echo "rust tests failed" >> ~/tests.log
.venv/bin/python -m pytest ../python/tests -q > ~/pytest.log 2>&1 || echo "pytest failed" >> ~/pytest.log
.venv/bin/python bench_datasets.py \
  --libs puggles-seq,puggles-par,puggles-rowscan,puggles-auto,genevo,ga,puggles-py,puggles-py-batch,puggles-py-numba,puggles-py-cfunc,puggles-py-torch,puggles-py-jax,pymoo,pymoo-numba,platypus,deap \
  --out ~/results_gcp_c3.json > ~/bench.log 2>&1
touch ~/DONE
