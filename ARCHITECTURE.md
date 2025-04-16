# LBRX-MLX System Architecture

This document describes the high-level architecture of LBRX-MLX, explaining the core components, their interactions, and the design decisions behind them.

## System Overview

LBRX-MLX is architected as a modular system with clear separation of concerns:

```
┌────────────────┐     ┌────────────────┐     ┌────────────────┐
│  Core Systems  │────▶│ Format Handlers │────▶│  Metal Engine  │
└────────────────┘     └────────────────┘     └────────────────┘
        │                      │                      │
        └──────────────┬──────────────┬──────────────┘
                       ▼              ▼
               ┌────────────────┐    ┌────────────────┐
               │  CLI Interface │    │  C/Python API  │
               └────────────────┘    └────────────────┘
```

The system is designed with the following key architectural principles:

1. **Zero-Copy Everywhere**: Minimize data copying across all operations
2. **Stream-First Design**: All components operate on data streams, not bulk data
3. **Type Safety**: Leverage Rust's type system for compile-time guarantees
4. **Pluggable Components**: Format handlers and compute kernels are modular plugins
5. **Metal-Aware Core**: Core data structures are designed for Metal integration

## Core Components

### Tensor System

The Tensor system is the fundamental building block:

```rust
pub struct Tensor<T: DataType> {
    shape: Shape,
    storage: TensorStorage<T>,
    metadata: TensorMetadata,
}

pub enum TensorStorage<T: DataType> {
    Cpu(CpuStorage<T>),
    Metal(MetalStorage<T>),
    Mmap(MmapStorage<T>),
    View(TensorView<T>),
}
```

Key design decisions:
- **Unified Storage Backend**: Different storage types (CPU, GPU, memory-mapped disk) share a common interface
- **Zero-Copy Views**: Tensor views provide zero-copy access to subregions of tensors
- **Rich Metadata**: Full tensor history, provenance, and quantization info is preserved
- **Type-Parametric**: Generic over element types with specialized implementations for optimized paths

### Memory Management

Memory management is critical for large model operations:

```
┌────────────────────────┐
│ Metal Heap Management  │
├────────────────────────┤
│ Memory Pools           │
├────────────────────────┤
│ Shared Memory Regions  │
├────────────────────────┤
│ Memory-Mapped Storage  │
└────────────────────────┘
```

Key design decisions:
- **Pooled Allocation**: Reuse allocations within size classes to minimize fragmentation
- **Metal Heaps**: Dedicated heaps for persistent GPU storage
- **Cross-Process Sharing**: Support for sharing model weights across processes
- **Transparent Paging**: Automatic movement of cold tensor data to disk
- **Explicit Ownership**: Clear ownership model for all memory resources

### Format Handling

Format handlers convert between file formats and in-memory representations:

```
┌─────────────┐     ┌─────────────┐     ┌─────────────┐
│ SafeTensors │     │    GGUF     │     │     MLX     │
└─────────────┘     └─────────────┘     └─────────────┘
        │                  │                   │
        └──────────┬───────┘                   │
                   ▼                           │
        ┌─────────────────────┐                │
        │ Common Tensor Format│◀───────────────┘
        └─────────────────────┘
                   │
        ┌──────────┴───────────┐
        ▼                      ▼
┌─────────────────┐    ┌─────────────────┐
│ Memory Tensors  │    │ Streaming Parser│
└─────────────────┘    └─────────────────┘
```

Key design decisions:
- **Streaming First**: All format handlers support streaming conversion without loading the entire model
- **Lazy Reading**: On-demand tensor loading to minimize memory overhead
- **Schema Evolution**: Graceful handling of schema versions and extensions
- **Format Detection**: Automatic detection of input formats
- **Bidirectional Conversion**: All supported formats can be both read and written

### Metal Engine

The Metal engine provides hardware acceleration:

```
┌────────────────────────────────┐
│ Metal Compute Command Encoder  │
├────────────────────────────────┤
│ Kernel Registry                │
├────────────────────────────────┤
│ Kernel Fusion                  │
├────────────────────────────────┤
│ Custom Kernels                 │
└────────────────────────────────┘
```

Key design decisions:
- **Static Kernel Compilation**: Pre-compile and cache common kernels
- **Dynamic Fusion**: Fuse multiple operations into single GPU passes
- **Auto-Tuning**: Automatically select optimal kernel parameters for the hardware
- **Async Execution**: Non-blocking command submission with explicit synchronization
- **Resource Management**: Intelligent management of scarce GPU resources

### Task Scheduler

The task scheduler coordinates parallel execution:

```
┌─────────────────┐     ┌─────────────────┐
│ Work Stealing   │     │ Data Parallelism │
└─────────────────┘     └─────────────────┘
        │                       │
        └─────────┬─────────────┘
                  ▼
         ┌─────────────────┐
         │  Thread Pool    │
         └─────────────────┘
                  │
         ┌────────┴────────┐
         ▼                 ▼
┌─────────────────┐ ┌─────────────────┐
│ CPU Executor    │ │ Metal Executor  │
└─────────────────┘ └─────────────────┘
```

Key design decisions:
- **Work Stealing**: Efficient load balancing across threads
- **Bounded Queues**: Prevent OOM conditions during heavy workloads
- **Priority Scheduling**: Critical tasks get execution preference
- **Heterogeneous Execution**: Seamless CPU/GPU task distribution
- **Cache-Aware Scheduling**: Schedule tasks to minimize cache thrashing

## Cross-Cutting Concerns

### Error Handling

```rust
#[derive(Error, Debug)]
pub enum Error {
    #[error("IO error: {0}")]
    Io(#[from] std::io::Error),
    
    #[error("Format error in {format}: {message}")]
    Format { format: String, message: String },
    
    #[error("Metal error: {0}")]
    Metal(#[from] MetalError),
    
    #[error("Out of memory: requested {requested} bytes, available {available} bytes")]
    OutOfMemory { requested: usize, available: usize },
    
    // ...
}
```

Errors are rich, context-aware, and propagate complete information about failures.

### Validation

All external inputs undergo strict validation:
- Tensor shapes must be valid and match declared dimensions
- Memory boundaries are strictly enforced
- File formats must conform to specifications
- Model metadata is validated for consistency

### Monitoring

The system provides comprehensive introspection:
- Detailed performance metrics for all operations
- Memory usage tracking across CPU and GPU
- Operation timing with microsecond precision
- Resource utilization history

## Performance Optimizations

### Tensor Operations

- SIMD-accelerated operations on CPU
- Vectorized access patterns
- Cache-friendly memory layouts
- Metal compute shaders for GPU
- Kernel fusion for common patterns

### Memory Management

- Custom allocators for tensor data
- Zero-copy data sharing between CPU and GPU
- Memory pooling for common allocation sizes
- Pre-allocation of buffers for common operations
- Transparent compression for cold data

### Format Conversion

- Streaming parsers without intermediate copies
- Parallel processing of independent tensors
- Memory mapping for large files
- Progressive output generation
- Direct-to-GPU loading for compatible formats

## Extensibility

The system is designed for extensibility:
- Plugin system for new formats
- Custom kernel registration
- User-defined data types
- Extensible metadata
- Custom operation composition

## Future Considerations

- Multi-device scaling across multiple GPUs
- Distributed processing across networked machines
- Integration with neural engine for specific operations
- Serialization/deserialization of computational graphs
- Just-in-time compilation of custom operations

---

*Last updated: April 16, 2025*