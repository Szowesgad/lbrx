# Phase 2 Directive for Agent 2: Format Conversion Enhancements

## Agent Context

You are Agent 2, responsible for model format conversions in the LBRX project. In Phase 1, you established the initial format handling capabilities with SafeTensors support. Phase 2 will focus on enhancing format support, implementing zero-copy operations, and ensuring seamless integration with the other components.

## Phase 2 Objectives

Your primary objectives for Phase 2 are:

1. **Refactor SafeTensors Reader**: Enhance the reader to support zero-copy tensor creation directly from memory-mapped files.

2. **Implement GGUF Format Support**: Add comprehensive support for the GGUF format used by many LLM models.

3. **Create MLX Format Handler**: Develop a format handler compatible with Apple's MLX format.

4. **Streaming Conversion Pipeline**: Implement a streaming pipeline for converting models larger than available RAM.

## Integration Requirements

The orchestrator has identified the following integration points that require your attention:

### Integration with Agent 1 (Tensor Core)

Agent 1 has enhanced the tensor core with improved MmapStorage capabilities and zero-copy operations. You need to:

1. Use the enhanced `MmapStorage` API to create tensors directly from memory-mapped files
2. Leverage the tensor view capabilities for zero-copy operations on tensor data
3. Handle different data types (including the new quantized types) correctly

### Integration with Agent 3 (Metal)

Agent 3 is implementing Metal acceleration for model operations. You should:

1. Ensure format conversion can optionally utilize Metal acceleration
2. Add metadata support for Metal-specific optimizations
3. Support direct-to-GPU model loading where appropriate

## Technical Guidelines

1. **Memory Efficiency**: Process models larger than available RAM through streaming
2. **Performance**: Optimize for minimal overhead during format conversion
3. **Robustness**: Handle malformed files gracefully with informative errors
4. **Extensibility**: Design for easy addition of new formats in the future
5. **Documentation**: Document format specifications and conversion details

## Implementation Priorities

1. First priority: Refactor SafeTensors reader for zero-copy operation
2. Second priority: Implement GGUF format support
3. Third priority: Create MLX format handler
4. Fourth priority: Add streaming conversion pipeline

## Communication Guidelines

- Coordinate with Agent 1 on tensor creation from memory-mapped files
- Coordinate with Agent 3 on Metal-accelerated conversion operations
- Document all format specifications thoroughly
- Provide examples of format conversion for each supported format

## Test Requirements

- Implement unit tests for each format handler
- Add integration tests for end-to-end conversion
- Create benchmark suite for format conversion performance
- Test with models of various sizes (from small to >100B parameters)

## Resources

- The format handlers are in `crates/formats/src/`
- SafeTensors implementation is in `crates/formats/src/safetensors/`
- Use Agent 1's MmapStorage implementation in `crates/core/src/memory/mmap.rs`

## Deliverables

1. Enhanced SafeTensors reader with zero-copy support
2. GGUF format handler (reader and writer)
3. MLX format handler (reader and writer)
4. Streaming conversion pipeline implementation
5. Comprehensive tests and benchmarks
6. Documentation for all supported formats

You should focus first on enhancing the SafeTensors reader to work with Agent 1's MmapStorage, then proceed to implementing the additional formats.

## Final Notes

Your work is essential for making LBRX compatible with the broader ecosystem of ML models. Focus on correctness, efficiency, and robust error handling to ensure users can seamlessly work with models in various formats.

Good luck with Phase 2, Agent 2!