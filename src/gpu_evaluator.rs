/// GPU-accelerated batch evaluation of objective functions via wgpu compute shaders.
///
/// # WGSL shader interface
///
/// The user-supplied shader must bind three resources:
///
/// ```wgsl
/// // Flat input buffer: solutions[pop_size * solution_length] (f32, row-major)
/// @group(0) @binding(0) var<storage, read> solutions: array<f32>;
///
/// // Flat output buffer: objectives[pop_size * num_objectives] (f32, row-major)
/// @group(0) @binding(1) var<storage, read_write> objectives: array<f32>;
///
/// struct Params {
///     solution_length: u32,
///     num_objectives:  u32,
///     pop_size:        u32,
/// }
/// @group(0) @binding(2) var<uniform> params: Params;
///
/// // Optional read-only auxiliary data (e.g. a flattened covariance matrix),
/// // uploaded via `new_blocking_with_data`. A 1-element placeholder when unused.
/// @group(0) @binding(3) var<storage, read> data: array<f32>;
///
/// @compute @workgroup_size(64)
/// fn main(@builtin(global_invocation_id) id: vec3<u32>) {
///     let sol_idx = id.x;
///     if sol_idx >= params.pop_size { return; }
///     let in_off  = sol_idx * params.solution_length;
///     let out_off = sol_idx * params.num_objectives;
///     // ... your objective computation here ...
/// }
/// ```

use wgpu::util::DeviceExt;

pub struct GpuEvaluator {
    device: wgpu::Device,
    queue: wgpu::Queue,
    pipeline: wgpu::ComputePipeline,
    bind_group_layout: wgpu::BindGroupLayout,
    /// Read-only auxiliary data bound at `@group(0) @binding(3)` (e.g. a covariance matrix),
    /// so data-driven objectives can run on the GPU. Empty when the objective is self-contained.
    data_buffer: wgpu::Buffer,
    pub solution_length: usize,
    pub num_objectives: usize,
}

/// Cast any `&[T]` to `&[u8]`.
/// Safety: T must be Pod (plain data, no padding/uninit).
unsafe fn as_bytes<T>(data: &[T]) -> &[u8] {
    std::slice::from_raw_parts(
        data.as_ptr() as *const u8,
        data.len() * std::mem::size_of::<T>(),
    )
}

/// Cast `&[u8]` (aligned to f32) to `&[f32]`.
unsafe fn as_f32_slice(data: &[u8]) -> &[f32] {
    std::slice::from_raw_parts(
        data.as_ptr() as *const f32,
        data.len() / std::mem::size_of::<f32>(),
    )
}

impl GpuEvaluator {
    /// Initialise the GPU evaluator synchronously (blocks on async wgpu init).
    /// Panics if no compatible GPU adapter is found.
    pub fn new_blocking(
        shader_wgsl: &str,
        solution_length: usize,
        num_objectives: usize,
    ) -> Self {
        pollster::block_on(Self::new_async(shader_wgsl, solution_length, num_objectives, &[]))
    }

    /// Like `new_blocking`, but uploads read-only `data` (f32) bound at `@group(0) @binding(3)`.
    /// Use for data-driven objectives — bake nothing into the shader, read `data` instead.
    pub fn new_blocking_with_data(
        shader_wgsl: &str,
        solution_length: usize,
        num_objectives: usize,
        data: &[f32],
    ) -> Self {
        pollster::block_on(Self::new_async(shader_wgsl, solution_length, num_objectives, data))
    }

    async fn new_async(
        shader_wgsl: &str,
        solution_length: usize,
        num_objectives: usize,
        data: &[f32],
    ) -> Self {
        let instance = wgpu::Instance::default();

        let adapter = instance
            .request_adapter(&wgpu::RequestAdapterOptions {
                power_preference: wgpu::PowerPreference::HighPerformance,
                compatible_surface: None,
                force_fallback_adapter: false,
            })
            .await
            .expect("No GPU adapter found. Ensure a wgpu-compatible GPU is available.");

        let info = adapter.get_info();
        eprintln!("rustypus: GPU = {} ({:?}, {:?})", info.name, info.device_type, info.backend);

        let (device, queue) = adapter
            .request_device(&wgpu::DeviceDescriptor::default(), None)
            .await
            .expect("Failed to create wgpu device");

        let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("rustypus_objective_shader"),
            source: wgpu::ShaderSource::Wgsl(shader_wgsl.into()),
        });

        let bind_group_layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("rustypus_bgl"),
            entries: &[
                // binding 0: solutions input (read-only storage)
                wgpu::BindGroupLayoutEntry {
                    binding: 0,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Storage { read_only: true },
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
                // binding 1: objectives output (read-write storage)
                wgpu::BindGroupLayoutEntry {
                    binding: 1,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Storage { read_only: false },
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
                // binding 2: params uniform
                wgpu::BindGroupLayoutEntry {
                    binding: 2,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Uniform,
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
                // binding 3: read-only auxiliary data (optional; unused by self-contained shaders)
                wgpu::BindGroupLayoutEntry {
                    binding: 3,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Storage { read_only: true },
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
            ],
        });

        // Upload the auxiliary data (or a 1-element placeholder — zero-sized buffers are invalid).
        let data_src: Vec<f32> = if data.is_empty() { vec![0.0] } else { data.to_vec() };
        let data_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("rustypus_data"),
            contents: unsafe { as_bytes(&data_src) },
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
        });

        let pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("rustypus_pipeline_layout"),
            bind_group_layouts: &[&bind_group_layout],
            push_constant_ranges: &[],
        });

        let pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some("rustypus_compute_pipeline"),
            layout: Some(&pipeline_layout),
            module: &shader,
            entry_point: "main",
            compilation_options: wgpu::PipelineCompilationOptions::default(),
            cache: None,
        });

        GpuEvaluator {
            device,
            queue,
            pipeline,
            bind_group_layout,
            data_buffer,
            solution_length,
            num_objectives,
        }
    }

    /// Evaluate a batch of solutions on the GPU.
    ///
    /// Returns one `Vec<f64>` of objective values per input solution,
    /// in the same order as `solutions`.
    pub fn evaluate_batch(&self, solutions: &[Vec<f64>]) -> Vec<Vec<f64>> {
        let pop_size = solutions.len();
        if pop_size == 0 {
            return Vec::new();
        }

        // Pack solutions into a flat f32 buffer (GPU works in f32)
        let input_data: Vec<f32> = solutions
            .iter()
            .flat_map(|s| s.iter().map(|&v| v as f32))
            .collect();

        let output_size = (pop_size * self.num_objectives * std::mem::size_of::<f32>()) as u64;

        let solution_buf = self.device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("solutions"),
            contents: unsafe { as_bytes(&input_data) },
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
        });

        let objectives_buf = self.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("objectives"),
            size: output_size,
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
            mapped_at_creation: false,
        });

        // Params uniform: [solution_length, num_objectives, pop_size] as u32
        let params_data: [u32; 3] = [
            self.solution_length as u32,
            self.num_objectives as u32,
            pop_size as u32,
        ];
        let params_buf = self.device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("params"),
            contents: unsafe { as_bytes(&params_data) },
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
        });

        let staging_buf = self.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("staging"),
            size: output_size,
            usage: wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::MAP_READ,
            mapped_at_creation: false,
        });

        let bind_group = self.device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("rustypus_bg"),
            layout: &self.bind_group_layout,
            entries: &[
                wgpu::BindGroupEntry { binding: 0, resource: solution_buf.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 1, resource: objectives_buf.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 2, resource: params_buf.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 3, resource: self.data_buffer.as_entire_binding() },
            ],
        });

        let mut encoder = self.device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
            label: Some("rustypus_encoder"),
        });

        {
            let mut cpass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("rustypus_cpass"),
                timestamp_writes: None,
            });
            cpass.set_pipeline(&self.pipeline);
            cpass.set_bind_group(0, &bind_group, &[]);
            // Workgroup size 64: dispatch enough groups to cover pop_size
            let workgroups = (pop_size as u32 + 63) / 64;
            cpass.dispatch_workgroups(workgroups, 1, 1);
        }

        encoder.copy_buffer_to_buffer(&objectives_buf, 0, &staging_buf, 0, output_size);
        self.queue.submit(std::iter::once(encoder.finish()));

        // Read back results synchronously
        let slice = staging_buf.slice(..);
        let (sender, receiver) = std::sync::mpsc::channel();
        slice.map_async(wgpu::MapMode::Read, move |v| sender.send(v).unwrap());
        self.device.poll(wgpu::Maintain::Wait);
        receiver.recv().unwrap().expect("GPU buffer map failed");

        let results = {
            let raw = slice.get_mapped_range();
            let floats: &[f32] = unsafe { as_f32_slice(&raw) };
            floats
                .chunks_exact(self.num_objectives)
                .map(|chunk| chunk.iter().map(|&v| v as f64).collect())
                .collect()
        };

        staging_buf.unmap();
        results
    }
}
