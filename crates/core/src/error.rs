//! Error types for the LBRX-MLX core library.

use thiserror::Error;
use std::path::PathBuf;

/// Errors that can occur in the LBRX-MLX core library.
#[derive(Error, Debug)]
pub enum Error {
    /// I/O error during file operations.
    #[error("I/O error: {0}")]
    Io(#[from] std::io::Error),
    
    /// Shape mismatch between tensors or operations.
    #[error("Shape mismatch: expected {expected} elements, got {actual}")]
    ShapeMismatch {
        expected: usize,
        actual: usize,
    },
    
    /// Dimension mismatch for an operation.
    #[error("Dimension mismatch: {0}")]
    DimensionMismatch(String),
    
    /// Out of memory error during allocation.
    #[error("Out of memory: requested {requested} bytes, available {available} bytes")]
    OutOfMemory {
        requested: usize,
        available: usize,
    },
    
    /// Index out of bounds error.
    #[error("Index out of bounds: index {index} is out of bounds for axis {axis} with size {size}")]
    IndexOutOfBounds {
        index: usize,
        axis: usize,
        size: usize,
    },
    
    /// Invalid shape for an operation.
    #[error("Invalid shape: {0}")]
    InvalidShape(String),
    
    /// Invalid argument to a function.
    #[error("Invalid argument: {0}")]
    InvalidArgument(String),
    
    /// Unsupported operation.
    #[error("Unsupported operation: {0}")]
    Unsupported(String),
    
    /// Feature is disabled in the current build.
    #[error("Feature disabled: {0}")]
    FeatureDisabled(String),
    
    /// Error in Metal operations.
    #[error("Metal error: {0}")]
    Metal(String),
    
    /// Error in memory-mapped operations.
    #[error("Memory map error: {0}")]
    MemoryMap(String),
    
    /// File not found.
    #[error("File not found: {0}")]
    FileNotFound(PathBuf),
    
    /// Format error in a file.
    #[error("Format error: {0}")]
    Format(String),
    
    /// Data type mismatch.
    #[error("Data type mismatch: expected {expected}, got {actual}")]
    DataTypeMismatch {
        expected: String,
        actual: String,
    },
    
    /// Internal error.
    #[error("Internal error: {0}")]
    Internal(String),
}

/// Result type for operations in the LBRX-MLX core library.
pub type Result<T> = std::result::Result<T, Error>;