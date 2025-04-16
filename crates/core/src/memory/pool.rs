//! Memory pooling for efficient tensor allocations.
//!
//! This module provides a memory pool for reusing allocations and reducing
//! memory fragmentation during tensor operations.

use std::collections::HashMap;
use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::Mutex;
use parking_lot::RwLock;
use crate::error::{Error, Result};

/// Default size buckets for the memory pool (in bytes)
const DEFAULT_BUCKETS: &[usize] = &[
    // Small allocations
    16, 32, 64, 128, 256, 512, 1024, 2048, 4096, 8192, 16384, 32768,
    // Medium allocations
    65536, 131072, 262144, 524288, 1048576, 2097152, 4194304, 8388608, 16777216,
    // Large allocations
    33554432, 67108864, 134217728, 268435456, 536870912, 1073741824,
];

/// Default capacity of the memory pool (4GB)
const DEFAULT_CAPACITY: usize = 4 * 1024 * 1024 * 1024;

/// Memory pool statistics
#[derive(Debug, Clone)]
pub struct AllocatorStats {
    /// Total memory allocated (both in-use and cached)
    pub total_allocated: usize,
    
    /// Memory currently in use (not in the pool)
    pub in_use: usize,
    
    /// Memory currently cached in the pool
    pub cached: usize,
    
    /// Number of active allocations
    pub active_allocations: usize,
    
    /// Number of cached allocations
    pub cached_allocations: usize,
    
    /// Pool capacity
    pub capacity: usize,
    
    /// Peak memory usage
    pub peak_usage: usize,
    
    /// Total cumulative allocations
    pub total_allocations: usize,
    
    /// Total cumulative frees
    pub total_frees: usize,
}

/// Memory pool allocation metadata
#[derive(Debug)]
struct AllocationMeta {
    /// Pointer to the allocated memory
    ptr: *mut u8,
    
    /// Size of the allocation in bytes
    size: usize,
    
    /// Alignment requirement in bytes
    alignment: usize,
    
    /// Whether this allocation is pinned
    pinned: bool,
}

/// Memory pool for efficient tensor allocations
pub struct MemoryPool {
    /// Mutex-protected map of size buckets to lists of free blocks
    free_blocks: Mutex<HashMap<(usize, usize, bool), Vec<*mut u8>>>,
    
    /// Map of active allocations for tracking and deallocation
    active_allocs: RwLock<HashMap<*mut u8, AllocationMeta>>,
    
    /// Current cached memory in bytes
    cached_memory: AtomicUsize,
    
    /// Current in-use memory in bytes
    in_use_memory: AtomicUsize,
    
    /// Pool capacity in bytes
    capacity: AtomicUsize,
    
    /// Peak memory usage in bytes
    peak_usage: AtomicUsize,
    
    /// Allocation count statistics
    alloc_count: AtomicUsize,
    
    /// Free count statistics
    free_count: AtomicUsize,
}

impl MemoryPool {
    /// Create a new memory pool with default settings
    pub fn new() -> Self {
        Self {
            free_blocks: Mutex::new(HashMap::new()),
            active_allocs: RwLock::new(HashMap::new()),
            cached_memory: AtomicUsize::new(0),
            in_use_memory: AtomicUsize::new(0),
            capacity: AtomicUsize::new(DEFAULT_CAPACITY),
            peak_usage: AtomicUsize::new(0),
            alloc_count: AtomicUsize::new(0),
            free_count: AtomicUsize::new(0),
        }
    }
    
    /// Set the capacity of the memory pool
    pub fn set_capacity(&self, capacity: usize) {
        self.capacity.store(capacity, Ordering::SeqCst);
        self.trim_if_needed();
    }
    
    /// Get the current capacity of the memory pool
    pub fn capacity(&self) -> usize {
        self.capacity.load(Ordering::SeqCst)
    }
    
    /// Get the amount of cached memory in the pool
    pub fn cached_memory(&self) -> usize {
        self.cached_memory.load(Ordering::SeqCst)
    }
    
    /// Get the amount of memory currently in use
    pub fn in_use_memory(&self) -> usize {
        self.in_use_memory.load(Ordering::SeqCst)
    }
    
    /// Get the amount of memory currently available for allocation
    pub fn available_memory(&self) -> usize {
        let capacity = self.capacity();
        let in_use = self.in_use_memory();
        
        if in_use >= capacity {
            0
        } else {
            capacity - in_use
        }
    }
    
    /// Get the peak memory usage
    pub fn peak_memory(&self) -> usize {
        self.peak_usage.load(Ordering::SeqCst)
    }
    
    /// Round up size to the next power of 2
    fn round_up_size(size: usize) -> usize {
        // Find the next bucket size that fits the requested size
        for &bucket in DEFAULT_BUCKETS {
            if bucket >= size {
                return bucket;
            }
        }
        
        // Round to the next power of 2 for large allocations
        let mut power = 1;
        while power < size {
            power *= 2;
        }
        power
    }
    
    /// Allocate memory from the pool
    pub fn allocate(&self, size: usize, alignment: usize, pinned: bool) -> Result<*mut u8> {
        if size == 0 {
            return Ok(std::ptr::null_mut());
        }
        
        // Increment allocation counter
        self.alloc_count.fetch_add(1, Ordering::SeqCst);
        
        // Round up size to a bucket size for better reuse
        let bucket_size = Self::round_up_size(size);
        
        // Check if we have a cached block of this size and alignment
        let ptr = {
            let mut free_blocks = self.free_blocks.lock().unwrap();
            let key = (bucket_size, alignment, pinned);
            
            free_blocks.get_mut(&key).and_then(|blocks| {
                if !blocks.is_empty() {
                    // Reuse an existing block
                    let ptr = blocks.pop().unwrap();
                    
                    // Update cached memory count
                    self.cached_memory.fetch_sub(bucket_size, Ordering::SeqCst);
                    
                    Some(ptr)
                } else {
                    None
                }
            })
        };
        
        let ptr = if let Some(ptr) = ptr {
            // Found a cached block
            ptr
        } else {
            // No cached block, allocate a new one
            let layout = std::alloc::Layout::from_size_align(bucket_size, alignment)
                .map_err(|e| Error::InvalidArgument(e.to_string()))?;
            
            let ptr = unsafe { std::alloc::alloc(layout) };
            
            if ptr.is_null() {
                // Try to free some cached memory and retry
                self.trim(bucket_size);
                
                let ptr = unsafe { std::alloc::alloc(layout) };
                
                if ptr.is_null() {
                    return Err(Error::OutOfMemory {
                        requested: bucket_size,
                        available: self.available_memory(),
                    });
                }
            }
            
            ptr
        };
        
        // Add to active allocations
        {
            let mut active_allocs = self.active_allocs.write();
            active_allocs.insert(ptr, AllocationMeta {
                ptr,
                size: bucket_size,
                alignment,
                pinned,
            });
        }
        
        // Update in-use memory
        let old_in_use = self.in_use_memory.fetch_add(bucket_size, Ordering::SeqCst);
        let new_in_use = old_in_use + bucket_size;
        
        // Update peak usage if needed
        let mut peak = self.peak_usage.load(Ordering::SeqCst);
        while new_in_use > peak {
            match self.peak_usage.compare_exchange(
                peak, new_in_use, Ordering::SeqCst, Ordering::SeqCst
            ) {
                Ok(_) => break,
                Err(actual) => peak = actual,
            }
        }
        
        Ok(ptr)
    }
    
    /// Return memory to the pool
    pub fn free(&self, ptr: *mut u8, size: usize, alignment: usize) -> Result<()> {
        if ptr.is_null() || size == 0 {
            return Ok(());
        }
        
        // Increment free counter
        self.free_count.fetch_add(1, Ordering::SeqCst);
        
        // Get allocation metadata
        let meta = {
            let mut active_allocs = self.active_allocs.write();
            active_allocs.remove(&ptr).ok_or_else(|| {
                Error::InvalidArgument(format!("Pointer {:p} not found in active allocations", ptr))
            })?
        };
        
        // Update in-use memory
        self.in_use_memory.fetch_sub(meta.size, Ordering::SeqCst);
        
        // Add to free blocks
        let key = (meta.size, meta.alignment, meta.pinned);
        
        {
            let mut free_blocks = self.free_blocks.lock().unwrap();
            
            free_blocks.entry(key).or_insert_with(Vec::new).push(ptr);
            
            // Update cached memory
            self.cached_memory.fetch_add(meta.size, Ordering::SeqCst);
        }
        
        // Check if we need to trim the cache
        self.trim_if_needed();
        
        Ok(())
    }
    
    /// Trim the memory pool to ensure it doesn't exceed capacity
    fn trim_if_needed(&self) {
        let cached = self.cached_memory.load(Ordering::SeqCst);
        let in_use = self.in_use_memory.load(Ordering::SeqCst);
        let capacity = self.capacity.load(Ordering::SeqCst);
        
        if in_use + cached > capacity {
            // Need to trim
            let target = (in_use + cached) - capacity;
            self.trim(target);
        }
    }
    
    /// Trim the memory pool by freeing cached blocks
    fn trim(&self, target_bytes: usize) {
        let mut freed = 0;
        let mut free_blocks = self.free_blocks.lock().unwrap();
        
        // Sort blocks by size (largest first) to free fewer large blocks
        let mut sizes: Vec<(usize, usize, bool)> = free_blocks.keys().cloned().collect();
        sizes.sort_by(|a, b| b.0.cmp(&a.0));
        
        for key @ (size, alignment, _) in sizes {
            if freed >= target_bytes {
                break;
            }
            
            if let Some(blocks) = free_blocks.get_mut(&key) {
                while !blocks.is_empty() && freed < target_bytes {
                    let ptr = blocks.pop().unwrap();
                    
                    // Free the memory
                    unsafe {
                        let layout = std::alloc::Layout::from_size_align_unchecked(size, alignment);
                        std::alloc::dealloc(ptr, layout);
                    }
                    
                    freed += size;
                }
            }
        }
        
        // Update cached memory count
        self.cached_memory.fetch_sub(freed, Ordering::SeqCst);
    }
    
    /// Clear the memory pool and free all cached blocks
    pub fn reset(&self) -> Result<()> {
        let mut free_blocks = self.free_blocks.lock().unwrap();
        let mut freed = 0;
        
        for ((size, alignment, _), blocks) in free_blocks.iter_mut() {
            for &ptr in blocks.iter() {
                // Free the memory
                unsafe {
                    let layout = std::alloc::Layout::from_size_align_unchecked(*size, *alignment);
                    std::alloc::dealloc(ptr, layout);
                }
                
                freed += size;
            }
            
            blocks.clear();
        }
        
        // Update cached memory count
        self.cached_memory.fetch_sub(freed, Ordering::SeqCst);
        
        Ok(())
    }
    
    /// Get pool statistics
    pub fn stats(&self) -> AllocatorStats {
        let cached = self.cached_memory.load(Ordering::SeqCst);
        let in_use = self.in_use_memory.load(Ordering::SeqCst);
        let capacity = self.capacity.load(Ordering::SeqCst);
        let peak = self.peak_usage.load(Ordering::SeqCst);
        let alloc_count = self.alloc_count.load(Ordering::SeqCst);
        let free_count = self.free_count.load(Ordering::SeqCst);
        
        let (cached_allocs, active_allocs) = {
            let free_blocks = self.free_blocks.lock().unwrap();
            let active_allocs = self.active_allocs.read();
            
            let cached_allocs = free_blocks.values().map(|v| v.len()).sum();
            let active_allocs = active_allocs.len();
            
            (cached_allocs, active_allocs)
        };
        
        AllocatorStats {
            total_allocated: in_use + cached,
            in_use,
            cached,
            active_allocations: active_allocs,
            cached_allocations: cached_allocs,
            capacity,
            peak_usage: peak,
            total_allocations: alloc_count,
            total_frees: free_count,
        }
    }
}

impl Drop for MemoryPool {
    fn drop(&mut self) {
        // Free all blocks to prevent memory leaks
        let _ = self.reset();
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    
    #[test]
    fn test_pool_allocate_free() {
        let pool = MemoryPool::new();
        
        // Allocate some memory
        let ptr = pool.allocate(1024, 64, false).unwrap();
        assert!(!ptr.is_null());
        
        // Check that memory is tracked
        let stats = pool.stats();
        assert_eq!(stats.in_use, 1024);
        assert_eq!(stats.active_allocations, 1);
        
        // Free the memory
        pool.free(ptr, 1024, 64).unwrap();
        
        // Check that memory was returned to pool
        let stats = pool.stats();
        assert_eq!(stats.in_use, 0);
        assert_eq!(stats.cached, 1024);
        assert_eq!(stats.active_allocations, 0);
        assert_eq!(stats.cached_allocations, 1);
    }
    
    #[test]
    fn test_pool_reuse() {
        let pool = MemoryPool::new();
        
        // Allocate and free
        let ptr1 = pool.allocate(1024, 64, false).unwrap();
        pool.free(ptr1, 1024, 64).unwrap();
        
        // Allocate again - should reuse
        let ptr2 = pool.allocate(1024, 64, false).unwrap();
        
        // Should get the same pointer
        assert_eq!(ptr1, ptr2);
        
        // Clean up
        pool.free(ptr2, 1024, 64).unwrap();
    }
    
    #[test]
    fn test_pool_size_rounding() {
        let pool = MemoryPool::new();
        
        // Request a non-standard size
        let size = 1000; // Will be rounded up to 1024
        let ptr = pool.allocate(size, 64, false).unwrap();
        
        let stats = pool.stats();
        assert_eq!(stats.in_use, 1024); // Check rounded size
        
        // Clean up
        pool.free(ptr, size, 64).unwrap();
    }
    
    #[test]
    fn test_pool_trim() {
        let pool = MemoryPool::new();
        
        // Set a small capacity
        pool.set_capacity(4096);
        
        // Allocate and free several blocks to fill the cache
        let ptrs = [
            pool.allocate(1024, 64, false).unwrap(),
            pool.allocate(1024, 64, false).unwrap(),
            pool.allocate(1024, 64, false).unwrap(),
            pool.allocate(1024, 64, false).unwrap(),
        ];
        
        // Free all blocks
        for ptr in ptrs.iter() {
            pool.free(*ptr, 1024, 64).unwrap();
        }
        
        // Cache should be full
        let stats = pool.stats();
        assert_eq!(stats.cached, 4096);
        
        // Now allocate a block that would exceed capacity
        let new_ptr = pool.allocate(2048, 64, false).unwrap();
        
        // Cache should have been trimmed
        let stats = pool.stats();
        assert!(stats.cached < 4096);
        assert_eq!(stats.in_use, 2048);
        
        // Clean up
        pool.free(new_ptr, 2048, 64).unwrap();
    }
    
    #[test]
    fn test_pool_stats() {
        let pool = MemoryPool::new();
        
        // Initial stats
        let stats = pool.stats();
        assert_eq!(stats.in_use, 0);
        assert_eq!(stats.cached, 0);
        assert_eq!(stats.active_allocations, 0);
        
        // Allocate some memory
        let ptr1 = pool.allocate(1024, 64, false).unwrap();
        let ptr2 = pool.allocate(2048, 64, false).unwrap();
        
        // Check updated stats
        let stats = pool.stats();
        assert_eq!(stats.in_use, 3072);
        assert_eq!(stats.active_allocations, 2);
        assert_eq!(stats.total_allocations, 2);
        
        // Free one block
        pool.free(ptr1, 1024, 64).unwrap();
        
        // Check stats again
        let stats = pool.stats();
        assert_eq!(stats.in_use, 2048);
        assert_eq!(stats.cached, 1024);
        assert_eq!(stats.active_allocations, 1);
        assert_eq!(stats.cached_allocations, 1);
        assert_eq!(stats.total_frees, 1);
        
        // Clean up
        pool.free(ptr2, 2048, 64).unwrap();
    }
}