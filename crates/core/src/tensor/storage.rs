//! Storage backends for tensor data.
//!
//! This module provides different storage backends for tensor data,
//! including CPU memory, memory-mapped files, and Metal buffers.

use crate::dtype::DataType;
use crate::error::{Error, Result};
use crate::memory::{Buffer, AllocOptions};

/// Storage types for tensor data.
#[derive(Debug, Clone)]
pub enum TensorStorage<T: DataType> {
    /// CPU memory storage
    Cpu(CpuStorage<T>),
    
    /// Metal buffer storage (GPU)
    #[cfg(feature = "metal-backend")]
    Metal(super::metal::MetalStorage<T>),
    
    /// Memory-mapped file storage
    Mmap(MmapStorage<T>),
    
    /// View into another tensor
    View(super::TensorView<T>),
}

/// CPU memory storage for tensor data.
#[derive(Debug, Clone)]
pub struct CpuStorage<T: DataType> {
    /// Buffer for the data
    buffer: std::sync::Arc<parking_lot::RwLock<Buffer>>,
    
    /// Phantom data for type T
    _marker: std::marker::PhantomData<T>,
}

impl<T: DataType> CpuStorage<T> {
    /// Creates a new CPU storage with the given number of elements.
    pub fn new(num_elements: usize) -> Result<Self> {
        if num_elements == 0 {
            return Ok(Self {
                buffer: std::sync::Arc::new(parking_lot::RwLock::new(Buffer::new(0)?)),
                _marker: std::marker::PhantomData,
            });
        }
        
        let size = num_elements * std::mem::size_of::<T>();
        let options = AllocOptions::default().with_alignment(64);
        let mut buffer = Buffer::with_options(size, options)?;
        
        // Initialize to zero
        unsafe {
            std::ptr::write_bytes(buffer.as_mut_ptr(), 0, size);
        }
        
        Ok(Self {
            buffer: std::sync::Arc::new(parking_lot::RwLock::new(buffer)),
            _marker: std::marker::PhantomData,
        })
    }
    
    /// Returns a slice to the data.
    pub fn as_slice(&self) -> Vec<T> {
        let buffer = self.buffer.read();
        let ptr = buffer.as_ptr() as *const T;
        let len = buffer.size() / std::mem::size_of::<T>();
        
        unsafe {
            std::slice::from_raw_parts(ptr, len).to_vec()
        }
    }
    
    /// Returns a mutable slice to the data.
    /// This now returns a cloned slice for thread safety
    pub fn as_mut_slice(&mut self) -> Vec<T> {
        let buffer = self.buffer.read();
        let ptr = buffer.as_ptr() as *const T;
        let len = buffer.size() / std::mem::size_of::<T>();
        
        unsafe {
            std::slice::from_raw_parts(ptr, len).to_vec()
        }
    }
    
    /// Updates the buffer content with the given slice
    pub fn update_from_slice(&self, data: &[T]) -> Result<()> {
        let mut buffer = self.buffer.write();
        let len = buffer.size() / std::mem::size_of::<T>();
        
        if data.len() != len {
            return Err(Error::ShapeMismatch {
                expected: len,
                actual: data.len(),
            });
        }
        
        let dst_ptr = buffer.as_mut_ptr() as *mut T;
        unsafe {
            let dst_slice = std::slice::from_raw_parts_mut(dst_ptr, len);
            dst_slice.copy_from_slice(data);
        }
        
        Ok(())
    }
    
    /// Returns the number of elements in the storage.
    pub fn num_elements(&self) -> usize {
        self.buffer.read().size() / std::mem::size_of::<T>()
    }
    
    /// Creates a storage from an existing slice of data.
    pub fn from_slice(data: &[T]) -> Result<Self> {
        let num_elements = data.len();
        let storage = Self::new(num_elements)?;
        
        storage.update_from_slice(data)?;
        
        Ok(storage)
    }
}

/// Memory-mapped file storage for tensor data.
#[derive(Debug, Clone)]
pub struct MmapStorage<T: DataType> {
    /// Memory-mapped region
    mmap: std::sync::Arc<crate::memory::MemoryMap>,
    
    /// Offset into the memory-mapped region (in elements)
    offset: usize,
    
    /// Number of elements in the storage
    num_elements: usize,
    
    /// Phantom data for type T
    _marker: std::marker::PhantomData<T>,
}

impl<T: DataType> MmapStorage<T> {
    /// Creates a new memory-mapped storage from a file.
    pub fn from_file(path: impl AsRef<std::path::Path>, offset_bytes: usize, num_elements: usize) -> Result<Self> {
        let options = crate::memory::MemoryMapOptions::default()
            .with_read_only(true);
        
        let mmap = crate::memory::MemoryMap::from_file(path, options)?;
        
        // Ensure alignment of offset
        let alignment = 64;  // same as TENSOR_ALIGNMENT
        let aligned_offset = (offset_bytes + alignment - 1) & !(alignment - 1);
        
        // Check bounds
        let elem_size = std::mem::size_of::<T>();
        let required_size = aligned_offset + num_elements * elem_size;
        
        if required_size > mmap.size() {
            return Err(Error::InvalidArgument(format!(
                "Memory mapped region too small: {} bytes needed, {} available",
                required_size,
                mmap.size()
            )));
        }
        
        Ok(Self {
            mmap: std::sync::Arc::new(mmap),
            offset: aligned_offset / elem_size,
            num_elements,
            _marker: std::marker::PhantomData,
        })
    }
    
    /// Returns a slice to the data.
    pub fn as_slice(&self) -> &[T] {
        let ptr = unsafe { self.mmap.as_ptr().add(self.offset * std::mem::size_of::<T>()) } as *const T;
        
        unsafe {
            std::slice::from_raw_parts(ptr, self.num_elements)
        }
    }
    
    /// Copies data to a destination slice.
    pub fn copy_to_slice(&self, dst: &mut [T]) -> Result<()> {
        if dst.len() != self.num_elements {
            return Err(Error::ShapeMismatch {
                expected: self.num_elements,
                actual: dst.len(),
            });
        }
        
        dst.copy_from_slice(self.as_slice());
        
        Ok(())
    }
    
    /// Returns the number of elements in the storage.
    pub fn num_elements(&self) -> usize {
        self.num_elements
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    
    #[test]
    fn test_cpu_storage() {
        let num_elements = 10;
        let mut storage = CpuStorage::<f32>::new(num_elements).unwrap();
        
        assert_eq!(storage.num_elements(), num_elements);
        
        // Test zero initialization
        let slice = storage.as_slice();
        assert_eq!(slice.len(), num_elements);
        assert!(slice.iter().all(|&x| x == 0.0));
        
        // Test mutable access
        let mut_slice = storage.as_mut_slice();
        for (i, item) in mut_slice.iter_mut().enumerate() {
            *item = i as f32;
        }
        
        // Verify changes
        let slice = storage.as_slice();
        for (i, &item) in slice.iter().enumerate() {
            assert_eq!(item, i as f32);
        }
    }
    
    #[test]
    fn test_from_slice() {
        let data = vec![1.0, 2.0, 3.0, 4.0, 5.0];
        let storage = CpuStorage::<f32>::from_slice(&data).unwrap();
        
        assert_eq!(storage.num_elements(), 5);
        assert_eq!(storage.as_slice(), &data);
    }
    
    // Memory-mapped storage tests require a real file, so they are omitted here
}