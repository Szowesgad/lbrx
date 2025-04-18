//! Metal acceleration module
//!
// Provides context management and pipeline compilation for Metal compute kernels.
pub mod context;
/// Buffer pool for Metal buffers
pub mod buffer;
/// Basic compute kernels (e.g., MatMul)
pub mod compute;

// -- Tests ---------------------------------------------------------------
#[cfg(test)]
mod tests {
    // Skip Metal-specific tests on non-macOS
    #[cfg(not(target_os = "macos"))]
    #[test]
    fn skip_metal_tests() {
        eprintln!("Skipping Metal backend tests: not macOS");
    }

    #[cfg(target_os = "macos")]
    mod macos {
        use crate::buffer::BufferPool;
        use crate::context::{MetalContext, MetalError};
        use metal::Device;

        #[test]
        fn test_buffer_pool_reuse() {
            let device = Device::system_default().expect("Metal device not found");
            let mut pool = BufferPool::new(&device);
            let size = 1000;
            let buf1 = pool.get_buffer(size);
            let aligned = ((size + 255) & !255) as u64;
            assert_eq!(buf1.length(), aligned, "Buffer length alignment mismatch");
            let ptr1 = buf1.as_ptr();
            pool.return_buffer(buf1);
            let buf2 = pool.get_buffer(size);
            assert_eq!(buf2.length(), aligned, "Buffer length alignment mismatch");
            let ptr2 = buf2.as_ptr();
            assert_eq!(ptr1, ptr2, "BufferPool should reuse the same buffer");
        }

        #[test]
        fn test_compile_kernel_error() {
            let mut ctx = MetalContext::new().expect("Failed to create MetalContext");
            let err = ctx.compile_kernel("nonexistent_kernel").unwrap_err();
            match err {
                MetalError::FunctionNotFound(name, _) => assert_eq!(name, "nonexistent_kernel"),
                _ => panic!("Expected FunctionNotFound error"),
            }
        }

        #[test]
        fn test_context_new() {
            let ctx = MetalContext::new();
            assert!(ctx.is_ok(), "MetalContext::new should succeed");
        }
    }
}
// ------------------------------------------------------------------------