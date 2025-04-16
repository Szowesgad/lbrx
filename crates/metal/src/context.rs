use metal;
use std::collections::HashMap;
use thiserror::Error;

/// Errors related to Metal context and pipeline creation
#[derive(Error, Debug)]
pub enum MetalError {
    #[error("no Metal device found")]
    NoDevice,

    #[error("default Metal library not found")]
    DefaultLibraryNotFound,

    #[error("function `{0}` not found: {1}")]
    FunctionNotFound(String, String),

    #[error("failed to create pipeline `{0}`: {1}")]
    PipelineCreationError(String, String),
}

/// Result type for Metal operations
pub type Result<T> = std::result::Result<T, MetalError>;

/// Context for Metal execution: device, command queue, library, and pipelines
pub struct MetalContext {
    pub device: metal::Device,
    pub command_queue: metal::CommandQueue,
    pub library: metal::Library,
    kernels: HashMap<String, metal::ComputePipelineState>,
}

impl MetalContext {
    /// Create a new Metal context, detecting device and loading default library
    pub fn new() -> Result<Self> {
        let device = metal::Device::system_default().ok_or(MetalError::NoDevice)?;
        let command_queue = device.new_command_queue();
        let library = device
            .new_default_library()
            .ok_or(MetalError::DefaultLibraryNotFound)?;

        Ok(Self {
            device,
            command_queue,
            library,
            kernels: HashMap::new(),
        })
    }

    /// Compile or retrieve a compute pipeline for the given function name
    pub fn compile_kernel(&mut self, name: &str) -> Result<&metal::ComputePipelineState> {
        if !self.kernels.contains_key(name) {
            let function = self
                .library
                .get_function(name, None)
                .map_err(|e| MetalError::FunctionNotFound(name.to_string(), e.to_string()))?;

            let pipeline = self
                .device
                .new_compute_pipeline_state_with_function(&function)
                .map_err(|e| MetalError::PipelineCreationError(name.to_string(), e.to_string()))?;

            self.kernels.insert(name.to_string(), pipeline);
        }
        Ok(&self.kernels[name])
    }

    /// Precompile default kernels (to be implemented)
    pub fn precompile_kernels(&mut self) -> Result<()> {
        // TODO: add kernel names, e.g., "matmul_f32", "elementwise_add_f32"
        Ok(())
    }
}