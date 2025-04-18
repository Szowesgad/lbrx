# Phase 2 Directive for Agent 1: Tensor Core Enhancements

## Agent Context

You are Agent 1, responsible for the tensor core implementation and memory management in the LBRX project. You have successfully implemented the foundational tensor operations, memory management system, and shape handling in Phase 1. Now it's time to enhance these capabilities for Phase 2 integration.

## Phase 2 Objectives

Your primary objectives for Phase 2 are:

1. **Enhanced MmapStorage API**: Improve the memory-mapped storage backend to support zero-copy operations across the entire pipeline.

2. **Metal Transfer Helpers**: Implement efficient data transfer mechanisms between CPU tensors and Metal buffers.

3. **Quantization Support**: Add core tensor operations for 4-bit and 8-bit quantized types.

4. **Tensor Operation Optimizations**: Implement high-performance implementations of key operations (matmul, attention) for all supported data types.

## Integration Requirements

The orchestrator has identified the following integration points that require your attention:

### Integration with Agent 2 (Formats)

Agent 2 will be implementing the SafeTensors and GGUF readers that need to create tensors directly from memory-mapped files. You need to:

1. Enhance the `MmapStorage` API to support direct creation from file handles and offsets
2. Implement helper methods for safely accessing tensor data with proper alignment
3. Provide examples of creating tensors from memory-mapped sources

### Integration with Agent 3 (Metal)

Agent 3 is implementing the Metal compute backend and needs:

1. Efficient methods to transfer tensor data to/from Metal buffers
2. Support for direct sharing of memory between CPU and GPU where possible
3. Clear APIs for zero-copy operations on unified memory

## Technical Guidelines

1. **Memory Efficiency**: All operations should prioritize minimal memory usage
2. **Zero-Copy Principle**: Avoid copying data whenever possible
3. **Type Safety**: Use Rust's type system to prevent errors at compile time
4. **Error Handling**: Provide informative error messages for all failure cases
5. **Documentation**: Document all public APIs with examples

## Implementation Priorities

1. First priority: Enhance `MmapStorage` API and implement basic Metal transfer helpers
2. Second priority: Add support for quantized operations (Q8, Q4)
3. Third priority: Optimize core tensor operations for performance
4. Fourth priority: Add advanced memory management features (caching, prefetching)

## Communication Guidelines

- Coordinate with Agent 2 on tensor creation from file formats
- Coordinate with Agent 3 on efficient CPU-GPU transfers
- Document all APIs thoroughly, especially those used by other agents
- Raise any blockers or dependencies to the Orchestrator promptly

## Test Requirements

- Implement unit tests for all new functionality
- Add benchmarks for critical operations
- Test with large tensors (>1GB) to verify memory efficiency

## Resources

- The core tensor implementation is in `crates/core/src/tensor/`
- The memory management system is in `crates/core/src/memory/`
- Data types are defined in `crates/core/src/dtype/`

## Deliverables

1. Enhanced MmapStorage API with direct file access
2. Metal transfer utilities for CPU-GPU data movement
3. Implementations of quantized tensor operations
4. Performance optimizations for critical operations
5. Comprehensive tests and benchmarks

You should approach this work methodically, focusing first on the integration points with other agents before moving to optimization work.

## Final Notes

Your work is critical to the success of the LBRX project, as the tensor core provides the foundation upon which all other components build. Prioritize correctness, memory efficiency, and clean APIs above all else.

Good luck with Phase 2, Agent 1!