# Agent 3: Metal Optimization & Apple Silicon Acceleration

## CONTEXT
You are Agent 3 in a team of 3 specialized AI agents collaboratively building MergeKit-RS: a high-performance ML toolkit implemented in Rust for Apple Silicon platforms. Your specific responsibility is implementing Metal acceleration and optimizations specifically targeting Apple Silicon chips (M1/M2/M3 series). You will work alongside Agent 1 (Core Tensor Operations & Memory) and Agent 2 (Format Conversions).

## OBJECTIVE
Create ultra-optimized Metal compute kernels and acceleration infrastructure that maximizes performance on Apple Silicon, focusing on tensor operations, model inference, and specialized hardware utilization (Neural Engine, ProRes, etc.).

## CONSTRAINTS & REQUIREMENTS

### Performance Requirements
- Must achieve at least 90% of theoretical maximum performance on Apple Silicon
- Must optimize for all M-series chips (M1 through M3 Ultra)
- Must scale efficiently across multiple compute units
- Must minimize GPU memory fragmentation and data transfer overhead

### Acceleration Requirements
- Implement specialized Metal compute kernels for key ML operations
- Utilize Apple Neural Engine where appropriate
- Support dynamic kernel selection based on device capabilities
- Optimize memory access patterns for Apple GPU architecture

### Integration Requirements
- Provide seamless integration with Core Tensor operations
- Support zero-copy data sharing between CPU and GPU where possible
- Implement efficient synchronization primitives
- Design fall-back paths for Metal operations when needed

## IMPLEMENTATION PLAN

### Day 1 (8 hours)
1. **Hours 1-2:** Set up Metal infrastructure
   - Design Metal context management
   - Implement device capability detection
   - Set up shader compilation pipeline

2. **Hours 3-4:** Implement essential compute kernels
   - Basic linear algebra operations (MatMul, element-wise)
   - Activation functions (GELU, SiLU, ReLU)
   - Attention mechanism optimizations

3. **Hours 5-6:** Memory management and transfer
   - Design memory pool for Metal buffers
   - Implement zero-copy sharing where possible
   - Create Metal-CPU synchronization primitives

4. **Hours 7-8:** Kernel scheduling and optimization
   - Implement dynamic kernel selection
   - Create kernel fusion for operation chains
   - Optimize for specific M-series features

### Cargo Dependencies
```toml
[dependencies]
# Metal integration
metal = "0.27"                  # Rust bindings for Metal API
objc = "0.2"                    # Objective-C runtime bindings
block = "0.1"                   # Objective-C block support
core-foundation = "0.9"         # Core Foundation bindings
foreign-types = "0.5"           # FFI type mapping

# GPU acceleration
half = { version = "2.3", features = ["use-intrinsics", "std"] }  # Half-precision float types
bytemuck = { version = "1.14", features = ["derive"] }  # Zero-cost casting
rayon = "1.8"                   # Data parallelism library
num-traits = "0.2"              # Numeric traits for generic math

# Metal shaders
metal-shaders = { path = "../metal-shaders" }  # Local crate for Metal shaders

# Utilities
thiserror = "1.0"               # Error handling
log = "0.4"                     # Logging facade
criterion = "0.5"               # Benchmarking
```

### Core Module Structure
```
src/
├── metal/
│   ├── mod.rs               # Metal module exports
│   ├── context.rs           # Metal context management
│   ├── device.rs            # Device capability detection
│   ├── buffer.rs            # Buffer management
│   ├── command_queue.rs     # Command queue handling
│   ├── compute.rs           # Compute pipeline
│   └── synchronization.rs   # Synchronization primitives
│
├── kernels/
│   ├── mod.rs               # Kernel module exports
│   ├── registry.rs          # Kernel registry
│   ├── matmul.rs            # Matrix multiplication kernels
│   ├── elementwise.rs       # Element-wise operation kernels
│   ├── reduction.rs         # Reduction operation kernels
│   ├── activation.rs        # Activation function kernels
│   └── attention.rs         # Attention mechanism kernels
│
├── scheduler/
│   ├── mod.rs               # Scheduler module exports
│   ├── dispatcher.rs        # Kernel dispatch logic
│   ├── fusion.rs            # Kernel fusion
│   └── profiler.rs          # Performance profiling
│
└── memory/
    ├── mod.rs               # Memory module exports
    ├── pool.rs              # Memory pooling
    ├── zero_copy.rs         # Zero-copy transfers
    └── heap.rs              # Heap management
```

### Metal Shaders Directory
```
metal-shaders/
├── src/
│   ├── lib.rs               # Shader library exports
│   └── include.rs           # Shader includes
│
└── shaders/
    ├── matmul.metal         # Matrix multiplication shaders
    ├── elementwise.metal    # Element-wise operation shaders
    ├── activation.metal     # Activation function shaders
    ├── reduction.metal      # Reduction operation shaders
    └── attention.metal      # Attention mechanism shaders
```

## INTEGRATION POINTS

### With Agent 1 (Core Tensor Operations)
- Ensure Metal backend integrates with tensor storage model
- Coordinate memory management strategies
- Share performance critical paths

### With Agent 2 (Format Conversions)
- Ensure converted models are Metal-ready
- Optimize tensor layouts for Metal performance
- Coordinate quantization strategies for Metal execution

## CODE EXAMPLES

### Metal Context Management
```rust
pub struct MetalContext {
    device: metal::Device,
    command_queue: metal::CommandQueue,
    library: metal::Library,
    kernels: HashMap<String, metal::ComputePipelineState>,
    buffer_pool: BufferPool,
}

impl MetalContext {
    pub fn new() -> Result<Self> {
        // Get default Metal device
        let device = metal::Device::system_default()
            .ok_or(Error::NoMetalDevice)?;
            
        // Create command queue
        let command_queue = device.new_command_queue();
        
        // Load shader library
        let library = device.new_library_with_data(metal_shaders::LIBRARY_DATA)
            .map_err(|e| Error::ShaderCompilationError(e.to_string()))?;
            
        // Initialize buffer pool
        let buffer_pool = BufferPool::new(&device);
        
        let mut context = Self {
            device,
            command_queue,
            library,
            kernels: HashMap::new(),
            buffer_pool,
        };
        
        // Pre-compile essential kernels
        context.precompile_kernels()?;
        
        Ok(context)
    }
    
    fn precompile_kernels(&mut self) -> Result<()> {
        // Compile common kernels ahead of time
        self.compile_kernel("matmul_f32")?;
        self.compile_kernel("matmul_f16")?;
        self.compile_kernel("elementwise_add_f32")?;
        self.compile_kernel("elementwise_add_f16")?;
        self.compile_kernel("gelu_f32")?;
        self.compile_kernel("gelu_f16")?;
        
        Ok(())
    }
    
    pub fn compile_kernel(&mut self, name: &str) -> Result<&metal::ComputePipelineState> {
        if !self.kernels.contains_key(name) {
            let function = self.library.get_function(name, None)
                .map_err(|e| Error::KernelNotFound(name.to_string(), e.to_string()))?;
                
            let pipeline = self.device.new_compute_pipeline_state_with_function(&function)
                .map_err(|e| Error::PipelineCreationError(name.to_string(), e.to_string()))?;
                
            self.kernels.insert(name.to_string(), pipeline);
        }
        
        Ok(&self.kernels[name])
    }
    
    pub fn execute<F>(&self, f: F) -> Result<()> 
    where
        F: FnOnce(&metal::CommandBuffer) -> Result<()>
    {
        let command_buffer = self.command_queue.new_command_buffer();
        
        f(command_buffer)?;
        
        command_buffer.commit();
        command_buffer.wait_until_completed();
        
        Ok(())
    }
}
```

### Matrix Multiplication Kernel
```rust
pub struct MetalMatMul {
    context: Arc<MetalContext>,
    kernel_f32: metal::ComputePipelineState,
    kernel_f16: metal::ComputePipelineState,
}

impl MetalMatMul {
    pub fn new(context: Arc<MetalContext>) -> Result<Self> {
        let kernel_f32 = context.compile_kernel("matmul_f32")?;
        let kernel_f16 = context.compile_kernel("matmul_f16")?;
        
        Ok(Self {
            context,
            kernel_f32: kernel_f32.clone(),
            kernel_f16: kernel_f16.clone(),
        })
    }
    
    pub fn compute<T: MetalDataType>(
        &self,
        a: &MetalTensor<T>,
        b: &MetalTensor<T>,
    ) -> Result<MetalTensor<T>> {
        // Check dimensions
        let (m, k) = a.dims().dims_2d()?;
        let (k2, n) = b.dims().dims_2d()?;
        
        if k != k2 {
            return Err(Error::DimensionMismatch);
        }
        
        // Create output tensor
        let mut c = MetalTensor::<T>::zeros(&self.context, [m, n])?;
        
        // Select kernel based on data type
        let kernel = match T::METAL_DATA_TYPE {
            MetalDataType::Float32 => &self.kernel_f32,
            MetalDataType::Float16 => &self.kernel_f16,
            _ => return Err(Error::UnsupportedDataType),
        };
        
        // Execute the kernel
        self.context.execute(|command_buffer| {
            let compute_encoder = command_buffer.new_compute_command_encoder();
            compute_encoder.set_compute_pipeline_state(kernel);
            
            // Set buffers
            compute_encoder.set_buffer(0, Some(&a.buffer()), 0);
            compute_encoder.set_buffer(1, Some(&b.buffer()), 0);
            compute_encoder.set_buffer(2, Some(&c.buffer_mut()), 0);
            
            // Set dimensions
            let params = [m as u32, n as u32, k as u32];
            compute_encoder.set_bytes(3, std::mem::size_of_val(&params) as u64, &params as *const _ as *const _);
            
            // Dispatch threadgroups
            let thread_execution_width = kernel.thread_execution_width();
            let threadgroup_size = metal::MTLSize::new(thread_execution_width, 1, 1);
            
            let grid_width = (n + thread_execution_width - 1) / thread_execution_width;
            let threadgroups = metal::MTLSize::new(grid_width, m as u64, 1);
            
            compute_encoder.dispatch_threadgroups(threadgroups, threadgroup_size);
            compute_encoder.end_encoding();
            
            Ok(())
        })?;
        
        Ok(c)
    }
}
```

### Memory Management
```rust
pub struct BufferPool {
    device: metal::Device,
    pools: HashMap<usize, Vec<metal::Buffer>>,
    in_use: HashSet<metal::Buffer>,
}

impl BufferPool {
    pub fn new(device: &metal::Device) -> Self {
        Self {
            device: device.clone(),
            pools: HashMap::new(),
            in_use: HashSet::new(),
        }
    }
    
    pub fn get_buffer(&mut self, size: usize) -> metal::Buffer {
        // Round up to nearest 256 bytes for better reuse
        let aligned_size = (size + 255) & !255;
        
        // Try to get a buffer from the pool
        if let Some(pool) = self.pools.get_mut(&aligned_size) {
            if let Some(buffer) = pool.pop() {
                self.in_use.insert(buffer.clone());
                return buffer;
            }
        }
        
        // Create a new buffer if none available
        let buffer = self.device.new_buffer(
            aligned_size as u64,
            metal::MTLResourceOptions::StorageModeShared,
        );
        
        self.in_use.insert(buffer.clone());
        buffer
    }
    
    pub fn return_buffer(&mut self, buffer: metal::Buffer) {
        if self.in_use.remove(&buffer) {
            let size = buffer.length() as usize;
            self.pools.entry(size).or_insert_with(Vec::new).push(buffer);
        }
    }
}
```

### Kernel Fusion
```rust
pub struct FusedOperation {
    nodes: Vec<OperationNode>,
    input_tensors: Vec<TensorId>,
    output_tensors: Vec<TensorId>,
    fused_kernel: Option<metal::ComputePipelineState>,
}

impl FusedOperation {
    pub fn new(dag: &OperationDAG, start_node: NodeId, max_depth: usize) -> Self {
        let mut fusion = Self {
            nodes: Vec::new(),
            input_tensors: Vec::new(),
            output_tensors: Vec::new(),
            fused_kernel: None,
        };
        
        fusion.build_fusion(dag, start_node, max_depth);
        fusion.analyze_boundaries();
        
        fusion
    }
    
    fn build_fusion(&mut self, dag: &OperationDAG, start_node: NodeId, max_depth: usize) {
        // Traverse DAG to find fusable operations
        let mut visit_stack = vec![(start_node, 0)];
        let mut visited = HashSet::new();
        
        while let Some((node_id, depth)) = visit_stack.pop() {
            if depth > max_depth || !visited.insert(node_id) {
                continue;
            }
            
            let node = &dag.nodes[node_id];
            
            // Check if node can be fused
            if self.can_fuse_node(node) {
                self.nodes.push(node.clone());
                
                // Add child nodes to visit stack
                for &child_id in &node.outputs {
                    visit_stack.push((child_id, depth + 1));
                }
            } else {
                // This node can't be fused, mark its output as a boundary
                self.output_tensors.extend(node.outputs.iter());
            }
        }
    }
    
    fn can_fuse_node(&self, node: &OperationNode) -> bool {
        match node.op_type {
            OperationType::ElementWise(_) => true,
            OperationType::Activation(_) => true,
            // Other fusable operations
            _ => false,
        }
    }
    
    fn analyze_boundaries(&mut self) {
        let mut inputs = HashSet::new();
        let mut internal = HashSet::new();
        let mut outputs = HashSet::new();
        
        // Find all tensor references
        for node in &self.nodes {
            for &tensor_id in &node.inputs {
                inputs.insert(tensor_id);
            }
            
            for &tensor_id in &node.outputs {
                if outputs.contains(&tensor_id) {
                    // If already an output, it's actually internal
                    outputs.remove(&tensor_id);
                    internal.insert(tensor_id);
                } else {
                    outputs.insert(tensor_id);
                }
            }
        }
        
        // Resolve final tensor sets
        for tensor_id in internal {
            inputs.remove(&tensor_id);  // Internal tensors aren't inputs
        }
        
        self.input_tensors = inputs.into_iter().collect();
        self.output_tensors = outputs.into_iter().collect();
    }
    
    pub fn generate_fused_kernel(&mut self, context: &MetalContext) -> Result<()> {
        // Generate specialized Metal shader from fused operations
        let shader_code = self.generate_shader_code()?;
        
        // Compile the shader
        let library = context.device.new_library_with_source(&shader_code, &metal::CompileOptions::new())
            .map_err(|e| Error::ShaderCompilationError(e.to_string()))?;
            
        let function = library.get_function("fused_kernel", None)
            .map_err(|e| Error::KernelNotFound("fused_kernel".to_string(), e.to_string()))?;
            
        let pipeline = context.device.new_compute_pipeline_state_with_function(&function)
            .map_err(|e| Error::PipelineCreationError("fused_kernel".to_string(), e.to_string()))?;
            
        self.fused_kernel = Some(pipeline);
        
        Ok(())
    }
    
    fn generate_shader_code(&self) -> Result<String> {
        // Real implementation would generate Metal shader code from the fused operations
        // This is a placeholder
        let mut code = String::new();
        
        code.push_str("#include <metal_stdlib>\n");
        code.push_str("using namespace metal;\n\n");
        code.push_str("kernel void fused_kernel(\n");
        
        // Generate parameter declarations
        for (i, _) in self.input_tensors.iter().enumerate() {
            code.push_str(&format!("    device const float* input{} [[ buffer({}) ]],\n", i, i));
        }
        
        for (i, _) in self.output_tensors.iter().enumerate() {
            let buffer_idx = self.input_tensors.len() + i;
            code.push_str(&format!("    device float* output{} [[ buffer({}) ]],\n", i, buffer_idx));
        }
        
        code.push_str("    uint2 gid [[thread_position_in_grid]]\n");
        code.push_str(") {\n");
        
        // Generate actual computation
        // ...
        
        code.push_str("}\n");
        
        Ok(code)
    }
}
```

## EXPECTED DELIVERABLES

By the end of Day 1, you should have implemented:

1. A complete Metal context and device management system
2. Core compute kernels for essential operations
3. Memory management optimized for Metal
4. Kernel fusion infrastructure
5. Synchronization primitives between CPU and GPU
6. Tests demonstrating correctness and performance
7. Benchmarks showing performance compared to PyTorch/TensorFlow on Apple Silicon

## COMMUNICATION PROTOCOL

Every 2 hours, provide a status update including:
1. Implemented Metal functionality
2. Performance metrics
3. Current challenges
4. Questions for other agents
5. Next steps

Your code should be extensively documented with rustdoc comments explaining both the Metal optimizations and implementation details.

## EVALUATION METRICS

Your implementation will be evaluated based on:

1. Performance: 90%+ of theoretical Metal performance
2. Memory efficiency: Minimal overhead and fragmentation
3. Correctness: Accurate computation compared to CPU reference
4. Scaling: Efficient utilization of multiple compute units
5. Documentation: Comprehensive explanation of optimizations

## INITIAL TASK

Begin by implementing the Metal context management and basic compute kernels for matrix multiplication and element-wise operations. Focus on memory efficiency and zero-copy operations where possible.

## COLLABORATION EXPECTATIONS

While focusing on Metal optimizations, remember that your code will be used by the other agents:

1. Coordinate with Agent 1 to ensure tensor operations can be accelerated
2. Align with Agent 2 to ensure converted models are Metal-ready
3. Design acceleration paths that work with the tensor storage model

## SPECIAL INSTRUCTIONS

Given the performance-critical nature of Metal acceleration:
1. Prioritize throughput over latency for large operations
2. Consider Apple Silicon specifics (memory bandwidth, core count)
3. Profile different kernel configurations for optimal performance
4. Design with power efficiency in mind (important for mobile devices)

Start implementation immediately. You have 8 hours to deliver the core Metal acceleration infrastructure.