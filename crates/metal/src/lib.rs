//! Metal acceleration module
//!
// Provides context management and pipeline compilation for Metal compute kernels.
pub mod context;
/// Buffer pool for Metal buffers
pub mod buffer;
/// Basic compute kernels (e.g., MatMul)
pub mod compute;