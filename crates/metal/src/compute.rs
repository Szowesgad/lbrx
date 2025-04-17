use crate::context::{MetalContext, Result};
use metal;
use std::sync::Arc;

/// Matrix multiplication kernel for Metal
pub struct MetalMatMul {
    context: Arc<MetalContext>,
    pipeline_f32: metal::ComputePipelineState,
    pipeline_f16: metal::ComputePipelineState,
}

impl MetalMatMul {
    /// Create a new MatMul kernel, compiling f32 and f16 pipelines
    pub fn new(context: Arc<MetalContext>) -> Result<Self> {
        let f32 = context.compile_kernel("matmul_f32")?.clone();
        let f16 = context.compile_kernel("matmul_f16")?.clone();
        Ok(Self {
            context,
            pipeline_f32: f32,
            pipeline_f16: f16,
        })
    }

    /// Compute C = A * B for f32 data
    pub fn compute_f32(
        &self,
        a: &metal::Buffer,
        b: &metal::Buffer,
        c: &metal::Buffer,
        m: u32,
        n: u32,
        k: u32,
    ) -> Result<()> {
        let pipeline = &self.pipeline_f32;
        self.context.execute(|cmd_buf| {
            let encoder = cmd_buf.new_compute_command_encoder();
            encoder.set_compute_pipeline_state(pipeline);
            encoder.set_buffer(0, Some(a), 0);
            encoder.set_buffer(1, Some(b), 0);
            encoder.set_buffer(2, Some(c), 0);
            // Set dimensions
            let params = [m, n, k];
            encoder.set_bytes(
                3,
                std::mem::size_of_val(&params) as u64,
                params.as_ptr() as *const _,
            );
            // Compute threadgroups
            let width = pipeline.thread_execution_width();
            let threadgroup_size = metal::MTLSize::new(width, 1, 1);
            let grid = metal::MTLSize::new(
                ((n + width - 1) / width) as u64,
                m as u64,
                1,
            );
            encoder.dispatch_threadgroups(grid, threadgroup_size);
            encoder.end_encoding();
            Ok(())
        })
    }
}