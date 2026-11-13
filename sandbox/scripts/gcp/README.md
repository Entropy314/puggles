# Running the benchmark on GCP

`run_cpu.sh` and `run_gpu.sh` set up a fresh Ubuntu 24.04 VM (toolchains, Python env, bindings, datasets) and run the suite. Upload the repo as `~/puggles.tgz` (excluding `target/`, `.venv`, `.git`, and `sandbox/data/real_returns_*`; keep `sandbox/data/real_prices.csv`, which regenerates byte-identical datasets) next to the script.

```bash
# CPU machine
gcloud compute instances create puggles-bench-c3 --zone us-central1-a --machine-type c3-standard-8 \
  --image-family ubuntu-2404-lts-amd64 --image-project ubuntu-os-cloud --boot-disk-size 60GB
# GPU machine (Deep Learning image has the NVIDIA driver; Spot is much cheaper but can be preempted)
gcloud compute instances create puggles-bench-gpu --zone us-central1-a --machine-type g2-standard-8 \
  --accelerator type=nvidia-l4,count=1 --maintenance-policy TERMINATE --provisioning-model=SPOT \
  --image-family common-cu129-ubuntu-2404-nvidia-580 --image-project deeplearning-platform-release \
  --boot-disk-size 150GB --metadata install-nvidia-driver=True
gcloud compute scp puggles.tgz run_gpu.sh VM:~ && gcloud compute ssh VM --command 'nohup bash ~/run_gpu.sh &'
```

Notes from the November 2026 runs:
- The GPU scripts are resumable: rerun after a Spot preemption and the driver continues from `--resume 1`.
- `run_gpu.sh` builds `puggles-cuda` inside Rust-CUDA's `ghcr.io/rust-gpu/rust-cuda-ubuntu24-cuda12` image.
- On that image the NVIDIA Vulkan driver would not initialize (even with `libnvidia-gl-580-server`), so wgpu saw only `llvmpipe`. The benchmark now reports "no GPU" in that case; CUDA and PyTorch are unaffected.
- Delete the VMs when done: `gcloud compute instances delete NAME --zone ZONE`.
