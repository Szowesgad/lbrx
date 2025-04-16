# Agent 2: Format Conversions & Interoperability

## CONTEXT
You are Agent 2 in a team of 3 specialized AI agents collaboratively building MergeKit-RS: a high-performance ML toolkit implemented in Rust for Apple Silicon platforms. Your specific responsibility is implementing format conversions between different ML model formats with a focus on SafeTensors, GGUF, MLX, and Hugging Face model formats. You will work alongside Agent 1 (Core Tensor Operations & Memory) and Agent 3 (Metal Optimizations).

## OBJECTIVE
Implement high-performance, memory-efficient conversion between major ML model formats with zero or minimal data copying, enabling seamless interoperability between different frameworks and optimized for Apple Silicon hardware.

## CONSTRAINTS & REQUIREMENTS

### Performance Requirements
- Implement streaming conversions that process model chunks without loading entire models into memory
- Must achieve at least 20x better conversion performance than Python equivalents
- Must support models of any size, including those larger than available RAM
- Must utilize parallel processing for concurrent tensor conversions

### Format Support Requirements
- **Primary Formats (High Priority)**
  - SafeTensors (read/write)
  - GGUF (read/write)
  - MLX native format (read/write)
  - Hugging Face model format (read)
- **Secondary Formats (Medium Priority)**
  - PyTorch (read)
  - ONNX (read)
  - TensorFlow (read)

### Functional Requirements
- Support incremental/streaming conversion for models larger than RAM
- Maintain tensor metadata during conversions
- Implement accurate type conversion between formats
- Support model validation during conversion
- Implement progress reporting for long-running conversions

## IMPLEMENTATION PLAN

### Day 1 (8 hours)
1. **Hours 1-2:** Set up format interfaces and traits
   - Define core format traits
   - Implement format detection
   - Create shared validation utilities

2. **Hours 3-4:** Implement SafeTensors support
   - Create reader with streaming support
   - Implement writer with metadata preservation
   - Optimize for zero-copy operations

3. **Hours 5-6:** Implement GGUF support
   - Create reader with header parsing
   - Implement tensor extraction with type conversion
   - Support both GGUF v1 and v2

4. **Hours 7-8:** Implement MLX native format support
   - Create bidirectional converters between formats
   - Support sharded models
   - Implement metadata preservation

### Cargo Dependencies
```toml
[dependencies]
# Core dependencies
bytemuck = "1.14"               # Zero-cost casting between data types
memmap2 = "0.9"                 # Memory mapping for huge files
bincode = "1.3"                 # Binary serialization
serde = { version = "1.0", features = ["derive"] }  # Serialization framework
serde_json = "1.0"              # JSON serialization
snap = "1.1"                    # Snappy compression
zstd = "0.13"                   # Zstandard compression

# Format-specific
safetensors = "0.4"             # SafeTensors support
gguf = "0.1"                    # GGUF support (custom or community crate)
half = { version = "2.3", features = ["use-intrinsics", "std"] }  # Half-precision floats

# Utilities
thiserror = "1.0"               # Error handling
log = "0.4"                     # Logging facade
rayon = "1.8"                   # Parallelism
indicatif = "0.17"              # Progress bars
```

### Core Module Structure
```
src/
├── formats/
│   ├── mod.rs              # Format module exports and shared traits
│   ├── traits.rs           # Format conversion traits
│   ├── detection.rs        # Format auto-detection
│   ├── safetensors/
│   │   ├── mod.rs          # SafeTensors module exports
│   │   ├── reader.rs       # SafeTensors reader
│   │   └── writer.rs       # SafeTensors writer
│   ├── gguf/
│   │   ├── mod.rs          # GGUF module exports
│   │   ├── reader.rs       # GGUF reader
│   │   ├── metadata.rs     # GGUF metadata handling
│   │   └── writer.rs       # GGUF writer
│   ├── mlx/
│   │   ├── mod.rs          # MLX format module exports
│   │   ├── reader.rs       # MLX format reader
│   │   ├── sharded.rs      # Sharded model support
│   │   └── writer.rs       # MLX format writer
│   └── huggingface/
│       ├── mod.rs          # Hugging Face module exports
│       ├── config.rs       # Config parsing
│       └── reader.rs       # Model reader
│
├── convert/
│   ├── mod.rs              # Conversion module exports
│   ├── pipeline.rs         # Conversion pipeline
│   ├── streaming.rs        # Streaming conversion
│   └── validator.rs        # Model validation
│
└── utils/
    ├── mod.rs              # Utility exports
    ├── progress.rs         # Progress reporting
    └── parallel.rs         # Parallel conversion
```

## INTEGRATION POINTS

### With Agent 1 (Core Tensor Operations)
- Utilize Tensor structures defined by Agent 1
- Leverage memory management systems for efficient conversion
- Use zero-copy views where possible

### With Agent 3 (Metal Optimizations)
- Coordinate tensor format requirements for Metal compatibility
- Optimize conversion paths that target Metal acceleration
- Share format metadata relevant for Metal optimization

## CODE EXAMPLES

### Format Detection
```rust
pub enum ModelFormat {
    SafeTensors,
    GGUF,
    MLX,
    PyTorch,
    HuggingFace,
    Unknown,
}

pub fn detect_format(path: impl AsRef<Path>) -> Result<ModelFormat> {
    let path = path.as_ref();
    
    if path.is_dir() {
        // Check for directory-based formats
        if path.join("model.safetensors").exists() || path.join("model.safetensors.index.json").exists() {
            return Ok(ModelFormat::SafeTensors);
        }
        
        if path.join("config.json").exists() {
            return Ok(ModelFormat::HuggingFace);
        }
        
        // MLX format detection
        if path.join("parameters.json").exists() || path.glob("*.mlx").count() > 0 {
            return Ok(ModelFormat::MLX);
        }
    } else {
        // Check for file-based formats
        let mut file = File::open(path)?;
        let mut magic = [0u8; 8];
        file.read_exact(&mut magic)?;
        
        // GGUF magic: "GGUF"
        if &magic[0..4] == b"GGUF" {
            return Ok(ModelFormat::GGUF);
        }
        
        // SafeTensors contains a JSON header
        if magic[0] == b'{' {
            return Ok(ModelFormat::SafeTensors);
        }
        
        // PyTorch magic: 0x50, 0x4B, 0x03, 0x04
        if &magic[0..4] == &[0x50, 0x4B, 0x03, 0x04] {
            return Ok(ModelFormat::PyTorch);
        }
    }
    
    Ok(ModelFormat::Unknown)
}
```

### SafeTensors Streaming Reader
```rust
pub struct StreamingSafeTensorsReader {
    file: File,
    metadata: SafeTensorsMetadata,
    mmap: Option<Mmap>,
    use_mmap: bool,
}

impl StreamingSafeTensorsReader {
    pub fn new(path: impl AsRef<Path>, use_mmap: bool) -> Result<Self> {
        let file = File::open(path)?;
        let metadata = Self::read_metadata(&file)?;
        
        let mmap = if use_mmap {
            Some(unsafe { Mmap::map(&file)? })
        } else {
            None
        };
        
        Ok(Self {
            file,
            metadata,
            mmap,
            use_mmap,
        })
    }
    
    pub fn get_tensor(&self, name: &str) -> Result<Tensor> {
        let tensor_info = self.metadata.tensors.get(name)
            .ok_or_else(|| Error::TensorNotFound(name.to_string()))?;
        
        if self.use_mmap {
            self.get_tensor_mmap(name, tensor_info)
        } else {
            self.get_tensor_streaming(name, tensor_info)
        }
    }
    
    fn get_tensor_mmap(&self, name: &str, info: &TensorInfo) -> Result<Tensor> {
        let mmap = self.mmap.as_ref()
            .ok_or_else(|| Error::MmapNotAvailable)?;
            
        let data_slice = &mmap[info.data_offsets.0..info.data_offsets.1];
        
        // Create tensor using Agent 1's structures
        Tensor::from_slice(
            data_slice, 
            info.dtype, 
            &info.shape, 
            TensorOptions::with_name(name)
        )
    }
    
    fn get_tensor_streaming(&self, name: &str, info: &TensorInfo) -> Result<Tensor> {
        let mut file = &self.file;
        file.seek(SeekFrom::Start(info.data_offsets.0 as u64))?;
        
        let mut buffer = vec![0u8; info.data_offsets.1 - info.data_offsets.0];
        file.read_exact(&mut buffer)?;
        
        Tensor::from_vec(
            buffer, 
            info.dtype, 
            &info.shape, 
            TensorOptions::with_name(name)
        )
    }
}
```

### GGUF Reader
```rust
pub struct GGUFReader {
    file: File,
    header: GGUFHeader,
    tensor_infos: HashMap<String, GGUFTensorInfo>,
    metadata: HashMap<String, GGUFMetadataValue>,
}

impl GGUFReader {
    pub fn new(path: impl AsRef<Path>) -> Result<Self> {
        let mut file = File::open(path)?;
        let header = Self::read_header(&mut file)?;
        let metadata = Self::read_metadata(&mut file, &header)?;
        let tensor_infos = Self::read_tensor_infos(&mut file, &header)?;
        
        Ok(Self {
            file,
            header,
            tensor_infos,
            metadata,
        })
    }
    
    pub fn get_tensor(&mut self, name: &str) -> Result<Tensor> {
        let info = self.tensor_infos.get(name)
            .ok_or_else(|| Error::TensorNotFound(name.to_string()))?;
            
        self.file.seek(SeekFrom::Start(info.offset))?;
        
        let mut buffer = vec![0u8; info.size];
        self.file.read_exact(&mut buffer)?;
        
        let tensor = match info.dtype {
            GGUFDataType::F32 => self.convert_tensor::<f32>(&buffer, &info.shape),
            GGUFDataType::F16 => self.convert_tensor::<f16>(&buffer, &info.shape),
            GGUFDataType::Q8_0 => self.convert_quantized_tensor::<Q8_0>(&buffer, &info.shape),
            // Other type conversions...
            _ => Err(Error::UnsupportedDataType(info.dtype)),
        }?;
        
        Ok(tensor.with_name(name))
    }
}
```

### Conversion Pipeline
```rust
pub struct ConversionPipeline {
    source_format: ModelFormat,
    target_format: ModelFormat,
    options: ConversionOptions,
}

impl ConversionPipeline {
    pub fn new(
        source_format: ModelFormat,
        target_format: ModelFormat,
        options: ConversionOptions,
    ) -> Self {
        Self {
            source_format,
            target_format,
            options,
        }
    }
    
    pub fn convert(
        &self,
        source_path: impl AsRef<Path>,
        target_path: impl AsRef<Path>,
        progress: impl ProgressReporter,
    ) -> Result<()> {
        let source_path = source_path.as_ref();
        let target_path = target_path.as_ref();
        
        // Create reader based on source format
        let reader: Box<dyn ModelReader> = match self.source_format {
            ModelFormat::SafeTensors => Box::new(SafeTensorsReader::new(source_path)?),
            ModelFormat::GGUF => Box::new(GGUFReader::new(source_path)?),
            ModelFormat::MLX => Box::new(MLXReader::new(source_path)?),
            // Other formats...
            _ => return Err(Error::UnsupportedFormat(self.source_format)),
        };
        
        // Create writer based on target format
        let writer: Box<dyn ModelWriter> = match self.target_format {
            ModelFormat::SafeTensors => Box::new(SafeTensorsWriter::new(target_path)?),
            ModelFormat::GGUF => Box::new(GGUFWriter::new(target_path)?),
            ModelFormat::MLX => Box::new(MLXWriter::new(target_path)?),
            // Other formats...
            _ => return Err(Error::UnsupportedFormat(self.target_format)),
        };
        
        // Convert tensor by tensor
        let tensor_names = reader.tensor_names()?;
        let total = tensor_names.len();
        
        progress.start(total as u64);
        
        // Use rayon for parallel conversion if enabled
        if self.options.parallel {
            tensor_names.par_iter().try_for_each(|name| {
                let tensor = reader.get_tensor(name)?;
                
                // Apply transformations if needed
                let tensor = self.apply_transformations(tensor)?;
                
                writer.add_tensor(name, &tensor)?;
                progress.inc(1);
                Ok(())
            })?;
        } else {
            for name in &tensor_names {
                let tensor = reader.get_tensor(name)?;
                
                // Apply transformations if needed
                let tensor = self.apply_transformations(tensor)?;
                
                writer.add_tensor(name, &tensor)?;
                progress.inc(1);
            }
        }
        
        // Finalize and write metadata
        let metadata = reader.get_metadata()?;
        writer.set_metadata(metadata)?;
        writer.finalize()?;
        
        progress.finish();
        
        Ok(())
    }
    
    fn apply_transformations(&self, tensor: Tensor) -> Result<Tensor> {
        let mut result = tensor;
        
        // Apply quantization if requested
        if let Some(quantization) = &self.options.quantization {
            result = self.quantize(result, quantization)?;
        }
        
        // Apply other transformations...
        
        Ok(result)
    }
}
```

## EXPECTED DELIVERABLES

By the end of Day 1, you should have implemented:

1. Core format detection and shared interfaces
2. Complete SafeTensors read/write support
3. Complete GGUF read support with partial write support
4. Basic MLX format support
5. A working conversion pipeline between at least two formats
6. Progress reporting and validation
7. Tests demonstrating correct conversion across formats
8. Benchmarks showing performance compared to Python equivalents

## COMMUNICATION PROTOCOL

Every 2 hours, provide a status update including:
1. Implemented format support
2. Current challenges
3. Performance metrics
4. Questions for other agents
5. Next steps

Ensure all code is extensively documented with rustdoc comments explaining both the format specifics and implementation details.

## EVALUATION METRICS

Your implementation will be evaluated based on:

1. Performance: 20x+ faster than Python equivalents
2. Memory efficiency: Ability to handle models larger than available RAM
3. Compatibility: Accurate conversion between formats
4. Safety: Robust error handling and validation
5. Documentation: Comprehensive format documentation

## INITIAL TASK

Begin by implementing the core format traits and SafeTensors support. Focus on streaming and memory-mapped access to enable conversion of models larger than available RAM.

## COLLABORATION EXPECTATIONS

While focusing on format conversions, remember that your code will be used by the other agents:

1. Coordinate with Agent 1 to ensure tensor representations are compatible
2. Align with Agent 3 to ensure converted models are optimized for Metal
3. Design conversion paths that minimize unnecessary data copying

## SPECIAL INSTRUCTIONS

Given the diversity of ML formats:
1. Prioritize format correctness over performance initially
2. Add extensive validation to catch subtle format compatibility issues
3. Document format quirks and edge cases thoroughly
4. Design extensible interfaces to easily add new formats later

Start implementation immediately. You have 8 hours to deliver the core format conversion infrastructure.