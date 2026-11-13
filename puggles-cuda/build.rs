// Compiles the `kernels` crate to PTX with rustc_codegen_nvvm; src/lib.rs embeds it.
fn main() {
    println!("cargo::rerun-if-changed=kernels");
    let out = std::path::PathBuf::from(std::env::var("OUT_DIR").unwrap());
    let manifest = std::path::PathBuf::from(std::env::var("CARGO_MANIFEST_DIR").unwrap());
    cuda_builder::CudaBuilder::new(manifest.join("kernels"))
        .copy_to(out.join("kernels.ptx"))
        .build()
        .unwrap();
}
