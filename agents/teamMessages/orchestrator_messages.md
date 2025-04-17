# Orchestrator Initial Messages for Agents

## Message to Agent 1 (Core Tensor Operations)

```
TECHNICAL ALIGNMENT: Core Tensor Architecture

CONTEXT: You're responsible for designing the fundamental tensor operations for MergeKit-RS. As Orchestrator, I'll be coordinating between you and the other agents.

DIRECTIVE: 
1. Prioritize implementing memory management abstractions first, as they will be critical for Agents 2 and 3
2. Ensure your `Tensor` struct supports all precision formats used in current ML models (particularly fp16, bf16, and int8 quantized)
3. Design memory layout to be compatible with Metal GPU execution (row-major format preferred)

INTEGRATION POINTS:
- With Agent 2: Support for zero-copy or minimal-copy format conversion
- With Agent 3: Memory layout optimization for Metal compute kernels

ACTION REQUIRED: Begin with memory management architecture and provide your first status update within 2 hours.
```

## Message to Agent 2 (Format Conversions)

```
TECHNICAL ALIGNMENT: Format Conversion Strategy

CONTEXT: You're responsible for implementing format conversions between different ML model formats. As Orchestrator, I'll be coordinating between you and the other agents.

DIRECTIVE:
1. Focus first on trait definitions that will be used across all formats
2. Prioritize GGUF and SafeTensors support as these are the most widely used formats
3. Design streaming conversion to support models larger than available RAM
4. Ensure proper metadata preservation across format conversions

INTEGRATION POINTS:
- With Agent 1: Leverage the Tensor structures for efficient representation
- With Agent 3: Ensure converted formats are optimized for Metal computation

ACTION REQUIRED: Begin with format trait definitions and provide your first status update within 2 hours.
```

## Message to Agent 3 (Metal Optimizations)

```
TECHNICAL ALIGNMENT: Metal Compute Architecture

CONTEXT: You're responsible for Metal acceleration and Apple Silicon optimizations. As Orchestrator, I'll be coordinating between you and the other agents.

DIRECTIVE:
1. Start with Metal context management and device capability detection
2. Prioritize matrix multiplication kernels as they are the most compute-intensive operations
3. Design memory management to minimize CPU-GPU transfers
4. Ensure compatibility with M1 through M3 Ultra chips

INTEGRATION POINTS:
- With Agent 1: Coordinate on memory layout and tensor structure
- With Agent 2: Define format requirements for optimal Metal execution

ACTION REQUIRED: Begin with Metal context setup and provide your first status update within 2 hours.
```

## Source Control Update (April 16, 2025, 0845)

```
PRIORITY DIRECTIVE: Git Workflow

CONTEXT: We need to establish a clear git workflow for collaborative development of MergeKit-RS.

DIRECTIVE:
1. Each agent should work in a separate feature branch:
   - Agent 1: feature/agent1-tensor-core (already created)
   - Agent 2: feature/agent2-format-conversions
   - Agent 3: feature/agent3-metal-optimizations

2. Development process:
   - Make regular commits with descriptive messages
   - Create pull requests to master branch when features are ready for integration
   - Include appropriate tests with all implementations

3. Branch management:
   - Keep branches focused on specific functionality
   - Rebase on master when necessary to resolve conflicts
   - Coordinate with Orchestrator before merging major changes

ACTION REQUIRED: Create your feature branch immediately and push initial code structure within the next hour.
```

## Integration Checkpoint Schedule

```
DAILY SCHEDULE: April 16, 2025

0800-0830: Initial alignment check
- Review agent initialization and initial code structure
- Verify API compatibility between components

1200-1230: Memory management checkpoint
- Review Agent 1's memory architecture
- Align memory strategies between all agents
- Confirm Metal buffer management approach

1600-1630: Core operation integration
- Verify basic tensor operations
- Test format conversions with core tensors
- Validate Metal acceleration of key operations

2000-2030: End-of-day progress assessment
- Review implemented functionality
- Address technical conflicts
- Set priorities for next day
```