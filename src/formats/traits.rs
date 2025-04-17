use std::path::Path;
use std::collections::HashMap;

// Define custom Result and Error types later
type Result<T> = std::result::Result<T, Box<dyn std::error::Error>>; 

// Define Tensor and Metadata types based on Agent 1's work later
struct Tensor; 
type Metadata = HashMap<String, String>; 

/// Trait for reading model tensors and metadata.
pub trait ModelReader {
    /// Returns the names of all tensors in the model.
    fn tensor_names(&self) -> Result<Vec<String>>;

    /// Retrieves a specific tensor by name.
    fn get_tensor(&self, name: &str) -> Result<Tensor>;

    /// Retrieves the model's metadata.
    fn get_metadata(&self) -> Result<Metadata>;
}

/// Trait for writing model tensors and metadata.
pub trait ModelWriter {
    /// Adds a tensor to the model.
    fn add_tensor(&mut self, name: &str, tensor: &Tensor) -> Result<()>;

    /// Sets the model's metadata.
    fn set_metadata(&mut self, metadata: Metadata) -> Result<()>;

    /// Finalizes the writing process (e.g., closes files, writes headers).
    fn finalize(&mut self) -> Result<()>;
} 