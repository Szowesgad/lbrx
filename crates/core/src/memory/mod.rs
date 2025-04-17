//! Memory management for high-performance tensor operations.
//!
//! This module provides memory management abstractions for efficient tensor
//! storage, including memory pooling, memory-mapped file access, and zero-copy
//! operations where possible.

mod arena;
mod mmap;
mod pool;
mod cache;

use std::sync::Arc;
use std::sync::atomic::{AtomicUsize, Ordering};
use crate::error::{Error, Result};

pub use arena::MemoryArena;
pub use mmap::{MemoryMap, MemoryMapOptions};
pub use pool::{MemoryPool, AllocatorStats};
pub use cache::{MemoryCache, CacheOptions};

/// Global memory pool for tensor allocations
static GLOBAL_MEMORY_POOL: once_cell::sync::Lazy<Arc<MemoryPool>> = 
    once_cell::sync::Lazy::new(|| Arc::new(MemoryPool::new()));

/// Current peak memory usage in bytes
static PEAK_MEMORY_USAGE: AtomicUsize = AtomicUsize::new(0);

/// Current total memory usage in bytes
static CURRENT_MEMORY_USAGE: AtomicUsize = AtomicUsize::new(0);

/// Memory allocation options
#[derive(Debug, Clone)]
pub struct AllocOptions {
    /// Whether to use memory pooling
    pub use_pool: bool,
    
    /// Alignment requirement in bytes (must be a power of 2)
    pub alignment: usize,
    
    /// Name for the allocation (for debugging/profiling)
    pub name: Option<String>,
    
    /// Whether the allocation needs to be pinned (for device transfers)
    pub pinned: bool,
}

impl Default for AllocOptions {
    fn default() -> Self {
        Self {
            use_pool: true,
            alignment: 64, // Cache line size on most architectures
            name: None,
            pinned: false,
        }
    }
}

impl AllocOptions {
    /// Create new allocation options with a name
    pub fn with_name(name: impl Into<String>) -> Self {
        let mut options = Self::default();
        options.name = Some(name.into());
        options
    }
    
    /// Set alignment requirement
    pub fn with_alignment(mut self, alignment: usize) -> Self {
        // Ensure alignment is a power of 2
        assert!(alignment.is_power_of_two());
        self.alignment = alignment;
        self
    }
    
    /// Set whether to use pooling
    pub fn with_pooling(mut self, use_pool: bool) -> Self {
        self.use_pool = use_pool;
        self
    }
    
    /// Set whether memory should be pinned
    pub fn with_pinned(mut self, pinned: bool) -> Self {
        self.pinned = pinned;
        self
    }
}

/// A memory buffer with a specific alignment.
#[derive(Debug)]
pub struct Buffer {
    /// Pointer to the allocated memory
    ptr: *mut u8,
    
    /// Size of the allocation in bytes
    size: usize,
    
    /// Whether this buffer is managed by the pool
    pooled: bool,
    
    /// Allocation options
    options: AllocOptions,
}

// Unsafe impl because we're managing raw pointers, but we ensure safety
unsafe impl Send for Buffer {}
unsafe impl Sync for Buffer {}

impl Buffer {
    /// Allocate a new buffer with the given size and default options
    pub fn new(size: usize) -> Result<Self> {
        Self::with_options(size, AllocOptions::default())
    }
    
    /// Allocate a new buffer with the given size and options
    pub fn with_options(size: usize, options: AllocOptions) -> Result<Self> {
        if size == 0 {
            return Ok(Self {
                ptr: std::ptr::null_mut(),
                size: 0,
                pooled: false,
                options,
            });
        }
        
        let ptr = if options.use_pool {
            // Try to allocate from the pool
            match GLOBAL_MEMORY_POOL.allocate(size, options.alignment, options.pinned) {
                Ok(ptr) => {
                    // Update current memory usage for accounting
                    let old_usage = CURRENT_MEMORY_USAGE.fetch_add(size, Ordering::SeqCst);
                    let new_usage = old_usage + size;
                    
                    // Update peak memory usage if necessary
                    let mut peak = PEAK_MEMORY_USAGE.load(Ordering::SeqCst);
                    while new_usage > peak {
                        match PEAK_MEMORY_USAGE.compare_exchange(
                            peak, new_usage, Ordering::SeqCst, Ordering::SeqCst
                        ) {
                            Ok(_) => break,
                            Err(actual) => peak = actual,
                        }
                    }
                    
                    Ok(ptr)
                }
                Err(e) => Err(Error::OutOfMemory {
                    requested: size,
                    available: GLOBAL_MEMORY_POOL.available_memory(),
                })
            }
        } else {
            // Direct OS allocation
            unsafe {
                let layout = std::alloc::Layout::from_size_align(size, options.alignment)
                    .map_err(|e| Error::InvalidArgument(e.to_string()))?;
                
                let ptr = std::alloc::alloc(layout);
                if ptr.is_null() {
                    Err(Error::OutOfMemory {
                        requested: size,
                        available: 0, // Unknown for direct OS allocation
                    })
                } else {
                    // Update memory usage accounting
                    let old_usage = CURRENT_MEMORY_USAGE.fetch_add(size, Ordering::SeqCst);
                    let new_usage = old_usage + size;
                    
                    // Update peak memory usage if necessary
                    let mut peak = PEAK_MEMORY_USAGE.load(Ordering::SeqCst);
                    while new_usage > peak {
                        match PEAK_MEMORY_USAGE.compare_exchange(
                            peak, new_usage, Ordering::SeqCst, Ordering::SeqCst
                        ) {
                            Ok(_) => break,
                            Err(actual) => peak = actual,
                        }
                    }
                    
                    Ok(ptr)
                }
            }
        }?;
        
        Ok(Self {
            ptr,
            size,
            pooled: options.use_pool,
            options,
        })
    }
    
    /// Get a pointer to the buffer
    pub fn as_ptr(&self) -> *const u8 {
        self.ptr
    }
    
    /// Get a mutable pointer to the buffer
    pub fn as_mut_ptr(&mut self) -> *mut u8 {
        self.ptr
    }
    
    /// Get the size of the buffer in bytes
    pub fn size(&self) -> usize {
        self.size
    }
    
    /// Get a slice to the buffer
    pub fn as_slice<T>(&self) -> &[T] where T: Copy {
        if self.size == 0 {
            return &[];
        }
        
        let elem_size = std::mem::size_of::<T>();
        let num_elements = self.size / elem_size;
        
        unsafe {
            std::slice::from_raw_parts(self.ptr as *const T, num_elements)
        }
    }
    
    /// Get a mutable slice to the buffer
    pub fn as_mut_slice<T>(&mut self) -> &mut [T] where T: Copy {
        if self.size == 0 {
            return &mut [];
        }
        
        let elem_size = std::mem::size_of::<T>();
        let num_elements = self.size / elem_size;
        
        unsafe {
            std::slice::from_raw_parts_mut(self.ptr as *mut T, num_elements)
        }
    }
    
    /// Copy data from another buffer into this buffer
    pub fn copy_from(&mut self, other: &Buffer) -> Result<()> {
        if self.size < other.size {
            return Err(Error::InvalidArgument(format!(
                "Destination buffer too small: {} < {}",
                self.size,
                other.size
            )));
        }
        
        unsafe {
            std::ptr::copy_nonoverlapping(
                other.ptr,
                self.ptr,
                other.size
            );
        }
        
        Ok(())
    }
    
    /// Clear the buffer (set all bytes to zero)
    pub fn clear(&mut self) {
        unsafe {
            std::ptr::write_bytes(self.ptr, 0, self.size);
        }
    }
}

impl Drop for Buffer {
    fn drop(&mut self) {
        if self.ptr.is_null() || self.size == 0 {
            return;
        }
        
        // Update memory usage accounting
        CURRENT_MEMORY_USAGE.fetch_sub(self.size, Ordering::SeqCst);
        
        if self.pooled {
            // Return to the pool
            if let Err(e) = GLOBAL_MEMORY_POOL.free(self.ptr, self.size, self.options.alignment) {
                // Just log the error, can't do much else in a destructor
                eprintln!("Error returning memory to pool: {}", e);
            }
        } else {
            // Free directly
            unsafe {
                let layout = std::alloc::Layout::from_size_align_unchecked(
                    self.size,
                    self.options.alignment
                );
                std::alloc::dealloc(self.ptr, layout);
            }
        }
    }
}

/// Memory region type
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum MemoryRegionType {
    /// System memory (RAM)
    System,
    
    /// Memory-mapped file
    MappedFile,
    
    /// Pinned memory (for device transfers)
    Pinned,
    
    /// Device (GPU) memory
    Device,
}

/// Initialize the memory subsystem
pub fn init() -> Result<()> {
    // Force initialization of the global memory pool
    let _ = GLOBAL_MEMORY_POOL.available_memory();
    
    Ok(())
}

/// Get current memory usage in bytes
pub fn current_memory_usage() -> usize {
    CURRENT_MEMORY_USAGE.load(Ordering::SeqCst)
}

/// Get peak memory usage in bytes
pub fn peak_memory_usage() -> usize {
    PEAK_MEMORY_USAGE.load(Ordering::SeqCst)
}

/// Get memory pool statistics
pub fn memory_stats() -> AllocatorStats {
    GLOBAL_MEMORY_POOL.stats()
}

/// Reset the memory pool (free all pooled allocations)
pub fn reset_pool() -> Result<()> {
    GLOBAL_MEMORY_POOL.reset()
}

/// Set the memory pool capacity
pub fn set_pool_capacity(capacity: usize) {
    GLOBAL_MEMORY_POOL.set_capacity(capacity);
}

#[cfg(test)]
mod tests {
    use super::*;
    
    #[test]
    fn test_buffer_allocation() {
        let size = 1024;
        let buffer = Buffer::new(size).unwrap();
        
        assert_eq!(buffer.size(), size);
        assert!(!buffer.as_ptr().is_null());
    }
    
    #[test]
    fn test_buffer_as_slice() {
        let mut buffer = Buffer::new(16).unwrap();
        
        // Test with u32 (4 bytes each, so 16/4 = 4 elements)
        {
            let slice = buffer.as_mut_slice::<u32>();
            assert_eq!(slice.len(), 4);
            
            // Write some data
            for (i, item) in slice.iter_mut().enumerate() {
                *item = i as u32;
            }
        }
        
        // Read back as u32
        {
            let slice = buffer.as_slice::<u32>();
            assert_eq!(slice.len(), 4);
            
            // Verify data
            for (i, &item) in slice.iter().enumerate() {
                assert_eq!(item, i as u32);
            }
        }
        
        // Read as u8
        {
            let slice = buffer.as_slice::<u8>();
            assert_eq!(slice.len(), 16);
            
            // Little-endian encoding check for the first u32 (0)
            assert_eq!(slice[0], 0);
            assert_eq!(slice[1], 0);
            assert_eq!(slice[2], 0);
            assert_eq!(slice[3], 0);
            
            // Little-endian encoding check for the second u32 (1)
            assert_eq!(slice[4], 1);
            assert_eq!(slice[5], 0);
            assert_eq!(slice[6], 0);
            assert_eq!(slice[7], 0);
        }
    }
    
    #[test]
    fn test_buffer_copy() {
        let size = 128;
        let mut src = Buffer::new(size).unwrap();
        let mut dst = Buffer::new(size).unwrap();
        
        // Fill source buffer with pattern
        for (i, byte) in src.as_mut_slice::<u8>().iter_mut().enumerate() {
            *byte = (i % 256) as u8;
        }
        
        // Clear destination
        dst.clear();
        
        // Copy data
        dst.copy_from(&src).unwrap();
        
        // Verify copy
        let src_slice = src.as_slice::<u8>();
        let dst_slice = dst.as_slice::<u8>();
        
        assert_eq!(src_slice, dst_slice);
    }
    
    #[test]
    fn test_memory_accounting() {
        // Record initial memory usage
        let initial_usage = current_memory_usage();
        let initial_peak = peak_memory_usage();
        
        // Allocate some memory
        let size = 1024 * 1024; // 1 MB
        let buffer = Buffer::new(size).unwrap();
        
        // Check that current usage increased
        let new_usage = current_memory_usage();
        assert!(new_usage >= initial_usage + size);
        
        // Check that peak usage increased
        let new_peak = peak_memory_usage();
        assert!(new_peak >= initial_peak + size);
        
        // Drop the buffer
        drop(buffer);
        
        // Check that current usage decreased
        let final_usage = current_memory_usage();
        assert!(final_usage <= new_usage - size);
        
        // Peak usage should not decrease
        let final_peak = peak_memory_usage();
        assert_eq!(final_peak, new_peak);
    }
}