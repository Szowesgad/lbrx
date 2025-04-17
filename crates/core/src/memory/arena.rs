//! Memory arena for efficient allocation of many small objects.
//!
//! This module provides a memory arena for efficient allocation of
//! many small objects without the overhead of individual allocations.

use std::alloc::{alloc, dealloc, Layout};
use std::cell::Cell;
use std::ptr::NonNull;
use std::sync::atomic::{AtomicUsize, Ordering};
use parking_lot::Mutex;
use crate::error::{Error, Result};

/// Default block size for the arena (1MB)
const DEFAULT_BLOCK_SIZE: usize = 1024 * 1024;

/// Default alignment for arena allocations
const DEFAULT_ALIGNMENT: usize = 16;

/// A block of memory in the arena
struct MemoryBlock {
    /// Pointer to the block
    ptr: NonNull<u8>,
    
    /// Size of the block in bytes
    size: usize,
    
    /// Current offset within the block
    offset: Cell<usize>,
    
    /// Number of allocations from this block
    allocation_count: Cell<usize>,
}

unsafe impl Send for MemoryBlock {}
unsafe impl Sync for MemoryBlock {}

impl MemoryBlock {
    /// Allocate a new memory block
    fn new(size: usize) -> Result<Self> {
        // Create layout for the allocation
        let layout = Layout::from_size_align(size, DEFAULT_ALIGNMENT)
            .map_err(|e| Error::InvalidArgument(e.to_string()))?;
        
        // Allocate memory
        let ptr = unsafe { alloc(layout) };
        
        if ptr.is_null() {
            return Err(Error::OutOfMemory {
                requested: size,
                available: 0,
            });
        }
        
        Ok(Self {
            ptr: NonNull::new(ptr).unwrap(),
            size,
            offset: Cell::new(0),
            allocation_count: Cell::new(0),
        })
    }
    
    /// Allocate memory from this block
    fn allocate(&self, size: usize, align: usize) -> Option<NonNull<u8>> {
        // Calculate aligned offset
        let offset = self.offset.get();
        let aligned_offset = (offset + align - 1) & !(align - 1);
        
        // Check if we have enough space
        if aligned_offset + size > self.size {
            return None;
        }
        
        // Update offset
        self.offset.set(aligned_offset + size);
        self.allocation_count.set(self.allocation_count.get() + 1);
        
        // Return pointer to the allocated memory
        let ptr = unsafe { self.ptr.as_ptr().add(aligned_offset) };
        Some(NonNull::new(ptr).unwrap())
    }
    
    /// Reset the block
    fn reset(&self) {
        self.offset.set(0);
        self.allocation_count.set(0);
    }
    
    /// Check if the block is empty
    fn is_empty(&self) -> bool {
        self.allocation_count.get() == 0
    }
    
    /// Get the amount of free space in the block
    fn free_space(&self) -> usize {
        self.size - self.offset.get()
    }
    
    /// Get the utilization ratio of the block (0.0 - 1.0)
    fn utilization(&self) -> f32 {
        self.offset.get() as f32 / self.size as f32
    }
}

impl Drop for MemoryBlock {
    fn drop(&mut self) {
        // Free the memory
        unsafe {
            let layout = Layout::from_size_align_unchecked(self.size, DEFAULT_ALIGNMENT);
            dealloc(self.ptr.as_ptr(), layout);
        }
    }
}

/// A handle to memory allocated from an arena
#[derive(Debug)]
pub struct ArenaBox<T> {
    /// Pointer to the allocated object
    ptr: NonNull<T>,
    
    /// Reference to the arena that owns this allocation
    arena: *const MemoryArena,
}

impl<T> ArenaBox<T> {
    /// Create a new arena box
    fn new(ptr: NonNull<T>, arena: &MemoryArena) -> Self {
        Self {
            ptr,
            arena,
        }
    }
    
    /// Get a reference to the contained value
    pub fn get(&self) -> &T {
        unsafe { self.ptr.as_ref() }
    }
    
    /// Get a mutable reference to the contained value
    pub fn get_mut(&mut self) -> &mut T {
        unsafe { self.ptr.as_mut() }
    }
}

impl<T> std::ops::Deref for ArenaBox<T> {
    type Target = T;
    
    fn deref(&self) -> &Self::Target {
        self.get()
    }
}

impl<T> std::ops::DerefMut for ArenaBox<T> {
    fn deref_mut(&mut self) -> &mut Self::Target {
        self.get_mut()
    }
}

unsafe impl<T: Send> Send for ArenaBox<T> {}
unsafe impl<T: Sync> Sync for ArenaBox<T> {}

/// A memory arena for efficient allocation of small objects
pub struct MemoryArena {
    /// Blocks of memory in the arena
    blocks: Mutex<Vec<MemoryBlock>>,
    
    /// Size of each block
    block_size: usize,
    
    /// Total memory allocated by this arena
    total_memory: AtomicUsize,
    
    /// Total number of allocations
    allocation_count: AtomicUsize,
}

impl MemoryArena {
    /// Create a new memory arena with the default block size
    pub fn new() -> Self {
        Self::with_block_size(DEFAULT_BLOCK_SIZE)
    }
    
    /// Create a new memory arena with the specified block size
    pub fn with_block_size(block_size: usize) -> Self {
        Self {
            blocks: Mutex::new(Vec::new()),
            block_size,
            total_memory: AtomicUsize::new(0),
            allocation_count: AtomicUsize::new(0),
        }
    }
    
    /// Allocate memory from the arena
    pub fn allocate(&self, size: usize, align: usize) -> Result<NonNull<u8>> {
        if size == 0 {
            return Err(Error::InvalidArgument("Cannot allocate zero bytes".to_string()));
        }
        
        // Update allocation count
        self.allocation_count.fetch_add(1, Ordering::Relaxed);
        
        // Check if the size is larger than the block size
        if size > self.block_size {
            // Allocate a dedicated block
            let block = MemoryBlock::new(size)?;
            let ptr = block.allocate(size, align).unwrap();
            
            // Add the block to our list
            let mut blocks = self.blocks.lock();
            blocks.push(block);
            
            // Update total memory
            self.total_memory.fetch_add(size, Ordering::Relaxed);
            
            return Ok(ptr);
        }
        
        // Try to allocate from an existing block
        let mut blocks = self.blocks.lock();
        
        for block in blocks.iter() {
            if let Some(ptr) = block.allocate(size, align) {
                return Ok(ptr);
            }
        }
        
        // No existing block has enough space, create a new one
        let new_block = MemoryBlock::new(self.block_size)?;
        let ptr = new_block.allocate(size, align).unwrap();
        
        // Add the block to our list
        blocks.push(new_block);
        
        // Update total memory
        self.total_memory.fetch_add(self.block_size, Ordering::Relaxed);
        
        Ok(ptr)
    }
    
    /// Allocate an object from the arena
    pub fn allocate_object<T>(&self, value: T) -> Result<ArenaBox<T>> {
        // Calculate size and alignment
        let size = std::mem::size_of::<T>();
        let align = std::mem::align_of::<T>();
        
        // Allocate memory
        let ptr = self.allocate(size, align)?;
        
        // Place the value in the allocated memory
        let typed_ptr = ptr.cast();
        unsafe {
            typed_ptr.as_ptr().write(value);
        }
        
        Ok(ArenaBox::new(typed_ptr, self))
    }
    
    /// Allocate a slice from the arena
    pub fn allocate_slice<T: Copy>(&self, values: &[T]) -> Result<&mut [T]> {
        // Calculate size and alignment
        let size = std::mem::size_of::<T>() * values.len();
        let align = std::mem::align_of::<T>();
        
        if size == 0 {
            return Ok(&mut []);
        }
        
        // Allocate memory
        let ptr = self.allocate(size, align)?;
        
        // Create typed slice
        let slice = unsafe {
            std::slice::from_raw_parts_mut(ptr.as_ptr() as *mut T, values.len())
        };
        
        // Copy values to the slice
        slice.copy_from_slice(values);
        
        Ok(slice)
    }
    
    /// Reset the arena, invalidating all allocations
    pub fn reset(&self) {
        let mut blocks = self.blocks.lock();
        
        for block in blocks.iter() {
            block.reset();
        }
    }
    
    /// Shrink the arena by freeing empty blocks
    pub fn shrink(&self) {
        let mut blocks = self.blocks.lock();
        
        // Filter out empty blocks
        let original_len = blocks.len();
        blocks.retain(|block| !block.is_empty());
        
        // Calculate freed memory
        let freed = (original_len - blocks.len()) * self.block_size;
        
        if freed > 0 {
            // Update total memory
            self.total_memory.fetch_sub(freed, Ordering::Relaxed);
        }
    }
    
    /// Get the total memory allocated by this arena
    pub fn total_memory(&self) -> usize {
        self.total_memory.load(Ordering::Relaxed)
    }
    
    /// Get the number of blocks in the arena
    pub fn block_count(&self) -> usize {
        let blocks = self.blocks.lock();
        blocks.len()
    }
    
    /// Get the number of allocations made from this arena
    pub fn allocation_count(&self) -> usize {
        self.allocation_count.load(Ordering::Relaxed)
    }
}

impl Drop for MemoryArena {
    fn drop(&mut self) {
        // Blocks will be dropped automatically
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    
    #[test]
    fn test_arena_allocation() {
        let arena = MemoryArena::new();
        
        // Allocate a simple type
        let result = arena.allocate(8, 8);
        assert!(result.is_ok());
        
        // Check allocation count
        assert_eq!(arena.allocation_count(), 1);
        
        // Check block count
        assert_eq!(arena.block_count(), 1);
    }
    
    #[test]
    fn test_arena_object() {
        let arena = MemoryArena::new();
        
        // Allocate a string
        let hello = String::from("Hello, Arena!");
        let result = arena.allocate_object(hello);
        assert!(result.is_ok());
        
        let boxed = result.unwrap();
        assert_eq!(*boxed, "Hello, Arena!");
    }
    
    #[test]
    fn test_arena_slice() {
        let arena = MemoryArena::new();
        
        // Allocate a slice of integers
        let values = [1, 2, 3, 4, 5];
        let result = arena.allocate_slice(&values);
        assert!(result.is_ok());
        
        let slice = result.unwrap();
        assert_eq!(slice, &[1, 2, 3, 4, 5]);
        
        // Modify the slice
        slice[0] = 10;
        assert_eq!(slice, &[10, 2, 3, 4, 5]);
    }
    
    #[test]
    fn test_arena_reset() {
        let arena = MemoryArena::new();
        
        // Make some allocations
        for _ in 0..100 {
            let _ = arena.allocate(8, 8);
        }
        
        assert_eq!(arena.allocation_count(), 100);
        assert_eq!(arena.block_count(), 1); // Should fit in one block
        
        // Reset the arena
        arena.reset();
        
        // Allocation count doesn't change
        assert_eq!(arena.allocation_count(), 100);
        
        // But we can reuse the memory
        for _ in 0..100 {
            let _ = arena.allocate(8, 8);
        }
        
        assert_eq!(arena.allocation_count(), 200);
        assert_eq!(arena.block_count(), 1); // Still one block
    }
    
    #[test]
    fn test_arena_large_allocation() {
        let arena = MemoryArena::with_block_size(1024);
        
        // Allocate something larger than the block size
        let result = arena.allocate(2048, 8);
        assert!(result.is_ok());
        
        // Should have created a dedicated block
        assert_eq!(arena.block_count(), 1);
    }
    
    #[test]
    fn test_arena_multiple_blocks() {
        let arena = MemoryArena::with_block_size(1024);
        
        // Fill the first block
        for _ in 0..10 {
            let _ = arena.allocate(100, 8); // 10 * 100 = 1000 bytes
        }
        
        assert_eq!(arena.block_count(), 1);
        
        // This should force a new block
        let _ = arena.allocate(100, 8);
        
        assert_eq!(arena.block_count(), 2);
    }
}