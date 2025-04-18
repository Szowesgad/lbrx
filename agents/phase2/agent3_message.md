# Phase 2 Directive for Agent 3: Metal Acceleration Implementation

## Agent Context

You are Agent 3, responsible for implementing Metal acceleration for the LBRX project. In Phase 1, you established the foundational Metal context and buffer management. In Phase 2, you will focus on implementing core compute kernels, optimizing performance for Apple Silicon, and ensuring seamless integration with other components.

## Phase 2 Objectives

Your primary objectives for Phase 2 are:

1. **Metal Shader Compilation**: Set up a robust system for compiling and caching Metal shaders.

2. **Implement MetalStorage Backend**: Create a tensor storage backend that utilizes Metal buffers.

3. **Core Compute Kernels**: Implement high-performance Metal kernels for critical operations (matmul, attention mechanisms, etc.)

4. **Memory Optimization**: Develop strategies for efficient memory usage, including buffer pooling and unified memory sharing.

## Integration Requirements

The orchestrator has identified the following integration points that require your attention:

### Integration with Agent 1 (Tensor Core)

Agent 1 has implemented tensor core functionality and memory management. You need to:

1. Utilize the tensor interfaces for seamless CPU-GPU data transfer
2. Leverage the memory management system for efficient buffer allocation
3. Implement Metal-specific operations that align with the tensor API

### Integration with Agent 2 (Formats)

Agent 2 is implementing format conversions for different model types. You should:

1. Support direct loading of model weights into Metal buffers where appropriate
2. Implement accelerated operations for format conversion tasks
3. Ensure compatibility with all supported model formats

## Technical Guidelines

1. **Apple Silicon Optimization**: Target the latest features of M3 and M4 chips while maintaining compatibility with M1/M2
2. **Unified Memory Utilization**: Leverage Apple's unified memory architecture for zero-copy operations
3. **Kernel Fusion**: Combine operations where possible to reduce memory transfers
4. **Error Handling**: Provide clear error messages for Metal-specific failures
5. **Performance Monitoring**: Implement tools to measure and optimize Metal performance

## Implementation Priorities

1. First priority: Set up Metal shader compilation and cache
2. Second priority: Implement basic MetalStorage backend
3. Third priority: Create core compute kernels for fundamental operations
4. Fourth priority: Optimize memory usage and implement advanced operations

## Communication Guidelines

- Coordinate with Agent 1 on CPU-GPU data transfer mechanisms
- Coordinate with Agent 2 on model format compatibility
- Document all Metal-specific APIs thoroughly
- Share performance insights and bottlenecks with the team

## Test Requirements

- Implement unit tests for Metal operations
- Create benchmarks comparing CPU vs Metal performance
- Test on different Apple Silicon generations if possible
- Verify memory usage patterns and optimization effectiveness

## Resources

- The Metal implementation is in `crates/metal/src/`
- Metal shaders are in `crates/metal/shaders/`
- Agent 1's tensor implementation is in `crates/core/src/tensor/`

## Deliverables

1. Metal shader compilation and caching system
2. Implementation of MetalStorage backend
3. Core compute kernels for fundamental operations
4. Memory optimization strategies
5. Performance benchmarks and optimization report
6. Documentation for Metal backend usage

You should focus first on establishing the shader compilation system and basic MetalStorage backend, then move on to implementing the core compute kernels.

## Final Notes

Your work is critical for achieving the performance goals of the LBRX project on Apple Silicon. Focus on maximizing hardware utilization while maintaining ease of use and compatibility with the rest of the system.

Apple's unified memory architecture offers unique opportunities for optimization - leverage this advantage to create a Metal backend that outperforms traditional ML frameworks on Apple Silicon.

Good luck with Phase 2, Agent 3!