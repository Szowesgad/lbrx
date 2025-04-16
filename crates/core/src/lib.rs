//! Core tensor operations and memory management for LBRX-MLX.
//!
//! This crate provides the fundamental data structures and algorithms
//! for efficient tensor operations on Apple Silicon hardware.

mod tensor;
mod memory;
mod parallel;
mod error;
mod dtype;
mod shape;

pub use tensor::*;
pub use memory::*;
pub use parallel::*;
pub use error::*;
pub use dtype::*;
pub use shape::*;

/// Library version
pub const VERSION: &str = env!("CARGO_PKG_VERSION");

/// Initializes the core library.
///
/// This function should be called at the start of the application
/// to set up global resources and configuration.
pub fn init() -> Result<(), Error> {
    // Initialize global memory management
    memory::init()?;
    
    // Set up parallel execution
    parallel::init()?;
    
    // Initialize Metal backend if available
    #[cfg(feature = "metal-backend")]
    {
        if let Err(e) = tensor::metal::init() {
            log::warn!("Failed to initialize Metal backend: {}", e);
            // Continue without Metal acceleration
        }
    }
    
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_library_initialization() {
        assert!(init().is_ok());
    }
}