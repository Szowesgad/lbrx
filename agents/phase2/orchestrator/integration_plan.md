# Phase 2 Integration Plan - LBRX-MLX

## Current Project Status (Orchestrator Assessment)

After successful completion of Phase 1, we have:

1. **Core Tensor Implementation (Agent 1):**
   - Complete memory management subsystem with arena, pool, mmap
   - Tensor abstraction with multiple storage backends
   - Element-wise and matrix operations
   - Data type abstraction including fp32, fp16, int8/4 quantized

2. **Format Support (Agent 2):**
   - Format traits defined in `src/formats/traits.rs`
   - SafeTensors reader partially implemented
   - Basic streaming approach defined

3. **Metal Acceleration (Agent 3):**
   - Metal context management
   - Buffer pooling
   - Basic MatMul kernel implementation

### Key Integration Challenges Identified

Based on team status reports and agent responses, the following integration challenges need resolution:

1. **Zero-copy Tensor Access:**
   - Agent 2 needs clear guidance on using `MmapStorage` for zero-copy
   - Current `RefCell<File>` approach may have concurrency limitations

2. **Metal Integration:**
   - Agent 3 needs shader binary strategy for Metal kernels
   - Metal tensor storage needs proper marshalling to/from CPU

3. **Data Type Compatibility:**
   - Consistent representation of quantized types across code base
   - Handling of fp16/bf16 between CPU/GPU

## Phase 2 Integration Plan

### 1. Core Development Tasks (Next 3 Days)

#### Agent 1 (Tensor Core):
- Implement memory-mapped tensor storage for zero-copy format reading
- Add explicit tensor conversion methods for Metal integration
- Enhance quantized type support based on Agent 2's format requirements
- Write integration tests for Tensor + Format operations

#### Agent 2 (Formats):
- Complete SafeTensors reader using zero-copy approach outlined by Agent 1
- Implement SafeTensors writer with appropriate metadata handling
- Begin GGUF reader implementation using same zero-copy approach
- Add format conversion benchmarks

#### Agent 3 (Metal):
- Implement `.metallib` shader compilation pipeline
- Complete `MetalStorage<T>` implementation with CPU<->GPU transfers
- Extend kernel set (add element-wise, activation functions)
- Create Metal benchmarks for core operations

### 2. Integration Points and Handoffs

#### Zero-Copy Integration (Agent 1 → Agent 2):
- Agent 1 has provided detailed API for MmapStorage in response document
- Agent 2 should implement format readers using this API by Day 2
- Agent 1 will review implementation to ensure zero-copy is working correctly

#### Metal Acceleration (Agent 1 → Agent 3):
- Agent 1 has outlined tensor conversion approach for Metal
- Agent 3 should implement MetalStorage following this design by Day 2
- Both agents must agree on memory layout to ensure efficient operations

#### Format + Metal (Agent 2 → Agent 3):
- Agent 2 must provide clear metadata about tensor types, especially quantization
- Agent 3 should validate that converted formats can be efficiently processed

### 3. Milestone Schedule

**Day 1 (Phase 2 Start):**
- All agents implement core APIs outlined in integration plan
- Initial integration tests for basic functionality
- Daily sync meeting to address any API mismatches

**Day 2 (Component Integration):**
- Agent 1 + Agent 2: Zero-copy reading of model files operational
- Agent 1 + Agent 3: CPU<->GPU tensor conversion working
- Cross-component integration tests

**Day 3 (Full Pipeline):**
- End-to-end flow demonstration:
  - Load model from file (SafeTensors)
  - Process with tensor operations
  - Accelerate with Metal
  - Write to new format (GGUF)

## Coordination Plan

1. **Code Reviews:**
   - All PRs must be reviewed by at least one other agent
   - Orchestrator will review integration points specifically
   - Focus on API consistency and error handling

2. **Daily Check-ins:**
   - Each agent will provide a daily progress update
   - Update should include:
     - Completed tasks
     - Blockers/issues requiring help
     - Integration points ready for testing

3. **Documentation Requirements:**
   - All public APIs must have documentation with examples
   - Integration points must have detailed usage instructions
   - Performance characteristics should be documented

## Immediate Next Steps

1. **Agent 1:**
   - Enhance MmapStorage API based on Agent 2 feedback
   - Implement Metal transfer helpers for Agent 3

2. **Agent 2:**
   - Refactor SafeTensors reader to use MmapStorage
   - Implement zero-copy tensor creation

3. **Agent 3:**
   - Setup Metal shader compilation pipeline
   - Implement basic MetalStorage integration

All agents: Review this integration plan and provide feedback by EOD.

— Orchestrator