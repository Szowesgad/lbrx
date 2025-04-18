# MergeKit-RS Installation and Implementation Guide

## Installation

### Prerequisites

- Apple Silicon Mac (M1/M2/M3/M4 series)
- macOS 13.0 or higher (macOS 15.0+ recommended for advanced memory management)
- 16GB RAM minimum (32GB+ recommended for larger models)
- Rust 1.77.2 or higher (if building from source)

### Installation Methods

#### Method 1: Using Pre-built Binaries (Recommended)

1. Download the latest release from GitHub:
   ```bash
   curl -LO https://github.com/LibraxisAI/lbrx-mlx/releases/latest/download/mergekit-rs-macos-arm64
   ```

2. Make the binary executable:
   ```bash
   chmod +x mergekit-rs-macos-arm64
   ```

3. Move to a location in your PATH:
   ```bash
   sudo mv mergekit-rs-macos-arm64 /usr/local/bin/mergekit-rs
   ```

4. Verify installation:
   ```bash
   mergekit-rs --version
   ```

#### Method 2: Using Cargo

```bash
# Install from crates.io
cargo install mergekit-rs

# Alternatively, with specific features
cargo install mergekit-rs --features="metal,safetensors,gguf"
```

#### Method 3: Building from Source

```bash
# Clone the repository
git clone https://github.com/LibraxisAI/lbrx-mlx.git
cd lbrx-mlx

# Build with optimizations for your specific hardware
RUSTFLAGS="-C target-cpu=native" cargo build --release

# The binary will be available at target/release/mergekit-rs
# Copy to a location in your PATH (optional)
sudo cp target/release/mergekit-rs /usr/local/bin/
```

## Implementation Guide

### Basic Workflow

1. **Convert Models**: Transform models from other formats (GGUF, SafeTensors) to MLX format
2. **Quantize Models**: Optionally compress models with minimal quality loss
3. **Merge Models**: Combine multiple models using various techniques
4. **Export Results**: Save in desired format for inference

### Command-Line Usage

#### Converting Models

```bash
# Basic conversion
mergekit-rs convert --input model.gguf --output model.mlx

# With 8-bit quantization
mergekit-rs convert --input model.safetensors --output model.mlx-q8 --quantize 8 --group-size 64

# With 4-bit quantization
mergekit-rs convert --input model.safetensors --output model.mlx-q4 --quantize 4 --group-size 32

# Stream conversion (for models larger than RAM)
mergekit-rs convert --input huge_model.gguf --output huge_model.mlx --stream
```

#### Merging Models

```bash
# Linear interpolation (equal weights)
mergekit-rs merge --models model1.mlx,model2.mlx --output merged.mlx --method linear

# Custom weights
mergekit-rs merge --models model1.mlx,model2.mlx --weights 0.3,0.7 --output merged.mlx --method linear

# SLERP interpolation
mergekit-rs merge --models model1.mlx,model2.mlx --weights 0.3,0.7 --output merged.mlx --method slerp

# Task vector application
mergekit-rs taskvector --base base_model.mlx --task task_model.mlx --output enhanced_model.mlx --strength 0.7
```

#### Batch Operations

```bash
# Convert multiple models in batch
mergekit-rs batch --config batch_config.json
```

Example `batch_config.json`:
```json
{
  "operations": [
    {
      "type": "convert",
      "input": "model1.gguf",
      "output": "model1.mlx",
      "quantize": 8,
      "group_size": 64
    },
    {
      "type": "convert",
      "input": "model2.gguf",
      "output": "model2.mlx",
      "quantize": 8,
      "group_size": 64
    },
    {
      "type": "merge",
      "method": "slerp",
      "models": ["model1.mlx", "model2.mlx"],
      "weights": [0.3, 0.7],
      "output": "merged.mlx"
    }
  ]
}
```

### Rust API Integration

Add to your `Cargo.toml`:
```toml
[dependencies]
mergekit-rs-lib = "0.1.0"
```

Basic usage in your code:

```rust
use mergekit_rs_lib as mk;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    // Convert model
    let model = mk::loader::load_model("model.gguf")?;
    let mlx_model = mk::converter::to_mlx(&model)?;
    
    // Quantize
    let quantized = mk::quantize::quantize_q8(&mlx_model, 64)?;
    
    // Save
    quantized.save("model.mlx-q8")?;
    
    Ok(())
}
```

Advanced model merging:

```rust
use mergekit_rs_lib as mk;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    // Load models
    let model1 = mk::loader::load_model("model1.mlx")?;
    let model2 = mk::loader::load_model("model2.mlx")?;
    
    // Configure merge options
    let merge_config = mk::merge::Config {
        method: mk::merge::Method::Slerp,
        weights: vec![0.3, 0.7],
        layers_to_merge: Some(vec!["attention", "ffn"]), // Optional: specify which layers to merge
        ..Default::default()
    };
    
    // Perform merge
    let merged = mk::merge::merge_models(&[&model1, &model2], &merge_config)?;
    
    // Save the result
    merged.save("merged.mlx")?;
    
    Ok(())
}
```

### Performance Optimization

For maximum performance on M3/M4:

```bash
# Use optimal compiler flags
RUSTFLAGS="-C target-cpu=native -C opt-level=3" cargo build --release

# Set environment variables for runtime
export MERGEKIT_METAL_HEAP_SIZE=8192  # 8GB Metal heap size
export MERGEKIT_NUM_THREADS=16  # Number of CPU threads to use
export MERGEKIT_CACHE_SIZE=4096  # 4GB memory cache size
```

### Troubleshooting

Common issues and solutions:

1. **Out of Memory**:
   ```bash
   mergekit-rs convert --input model.gguf --output model.mlx --stream --chunk-size 1024
   ```

2. **Metal Device Errors**:
   ```bash
   # Disable Metal acceleration
   mergekit-rs convert --input model.gguf --output model.mlx --use-cpu
   ```

3. **Slow Performance**:
   ```bash
   # Check hardware utilization during conversion
   mergekit-rs convert --input model.gguf --output model.mlx --verbose
   ```

## Additional Resources

- GitHub Repository: [github.com/LibraxisAI/lbrx-mlx](https://github.com/LibraxisAI/lbrx-mlx)
- Documentation: [docs.libraxis.ai/mergekit-rs](https://docs.libraxis.ai/mergekit-rs)
- Discord Community: [discord.gg/libraxisai](https://discord.gg/libraxisai)