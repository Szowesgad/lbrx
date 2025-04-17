//! Tensor view implementation for zero-copy slicing.
//!
//! This module provides a tensor view mechanism that allows for creating
//! slices of tensors without copying the underlying data.

use crate::dtype::DataType;
use crate::error::{Error, Result};
use crate::shape::Shape;
use std::sync::Arc;

/// A view into a region of a tensor, allowing for zero-copy slicing.
#[derive(Clone, Debug)]
pub struct TensorView<T: DataType> {
    /// The source tensor to view
    source: Arc<super::Tensor<T>>,
    
    /// Starting indices for each dimension
    start: Vec<usize>,
    
    /// Shape of the view
    shape: Shape,
    
    /// Linear offset into the source tensor's storage
    offset: usize,
    
    /// Custom strides for the view (if not contiguous)
    strides: Vec<usize>,
}

impl<T: DataType> TensorView<T> {
    /// Creates a new tensor view from a source tensor and index range.
    pub fn new(
        source: &super::Tensor<T>,
        start: &[usize],
        end: &[usize],
    ) -> Result<Self> {
        if start.len() != source.shape().rank() || end.len() != source.shape().rank() {
            return Err(Error::DimensionMismatch(format!(
                "Start/end indices must match tensor rank: got {}/{}, expected {}",
                start.len(),
                end.len(),
                source.shape().rank()
            )));
        }
        
        // Validate start and end indices
        for (i, (&s, &e)) in start.iter().zip(end.iter()).enumerate() {
            let dim_size = source.shape()[i];
            
            if s >= dim_size {
                return Err(Error::IndexOutOfBounds { 
                    index: s, 
                    axis: i, 
                    size: dim_size 
                });
            }
            
            if e > dim_size {
                return Err(Error::IndexOutOfBounds { 
                    index: e, 
                    axis: i, 
                    size: dim_size 
                });
            }
            
            if s >= e {
                return Err(Error::InvalidArgument(format!(
                    "Start index {} must be less than end index {} for dimension {}",
                    s, e, i
                )));
            }
        }
        
        // Create new shape for the view
        let new_dims: Vec<usize> = start.iter()
            .zip(end.iter())
            .map(|(&s, &e)| e - s)
            .collect();
        
        let new_shape = Shape::new(new_dims);
        
        // Calculate linear offset into the source storage
        let offset = source.shape().index_for(start)?;
        
        // Compute strides for the view
        let source_strides = source.shape().strides();
        let strides = source_strides.to_vec();
        
        Ok(Self {
            source: Arc::new(source.clone()),
            start: start.to_vec(),
            shape: new_shape,
            offset,
            strides,
        })
    }
    
    /// Returns the shape of the view.
    pub fn shape(&self) -> &Shape {
        &self.shape
    }
    
    /// Returns the number of elements in the view.
    pub fn num_elements(&self) -> usize {
        self.shape.num_elements()
    }
    
    /// Returns whether the view is contiguous in memory.
    pub fn is_contiguous(&self) -> bool {
        // Check if strides match the shape's computed strides
        let computed_strides = Shape::compute_strides(self.shape.dims());
        
        if computed_strides.len() != self.strides.len() {
            return false;
        }
        
        // For a view to be contiguous, each dimension except the last must span
        // the full size of the source tensor in that dimension, or be size 1
        for i in 0..self.shape.rank() - 1 {
            let view_size = self.shape[i];
            let source_size = self.source.shape()[i];
            
            if view_size != 1 && view_size != source_size {
                return false;
            }
        }
        
        true
    }
    
    /// Returns the source tensor being viewed.
    pub fn source(&self) -> &super::Tensor<T> {
        &self.source
    }
    
    /// Copies the data from the view to a destination slice.
    pub fn copy_to_slice(&self, dst: &mut [T]) -> Result<()> {
        if dst.len() != self.num_elements() {
            return Err(Error::ShapeMismatch {
                expected: self.num_elements(),
                actual: dst.len(),
            });
        }
        
        // If the view is contiguous, we can do a single copy
        if self.is_contiguous() {
            match self.source.storage() {
                super::TensorStorage::Cpu(storage) => {
                    let src_slice = storage.as_slice();
                    let end = self.offset + self.num_elements();
                    dst.copy_from_slice(&src_slice[self.offset..end]);
                },
                super::TensorStorage::Mmap(storage) => {
                    let src_slice = storage.as_slice();
                    let end = self.offset + self.num_elements();
                    dst.copy_from_slice(&src_slice[self.offset..end]);
                },
                #[cfg(feature = "metal-backend")]
                super::TensorStorage::Metal(_) => {
                    // Need to go through CPU for Metal storage
                    let cpu_tensor = self.source.to_device(super::Device::Cpu)?;
                    if let super::TensorStorage::Cpu(storage) = cpu_tensor.storage() {
                        let src_slice = storage.as_slice();
                        let end = self.offset + self.num_elements();
                        dst.copy_from_slice(&src_slice[self.offset..end]);
                    } else {
                        unreachable!("to_device(Cpu) should produce CPU storage");
                    }
                },
                super::TensorStorage::View(view) => {
                    // Recursively copy from nested view
                    view.copy_to_slice(dst)?;
                },
            }
        } else {
            // For non-contiguous views, we need to do element-by-element copy
            // This is a naive implementation, could be optimized for specific patterns
            let mut dst_idx = 0;
            
            // Convert the source tensor to CPU if needed
            let cpu_source = self.source.to_device(super::Device::Cpu)?;
            let src_storage = match cpu_source.storage() {
                super::TensorStorage::Cpu(storage) => storage,
                _ => unreachable!("to_device(Cpu) should produce CPU storage"),
            };
            let src_slice = src_storage.as_slice();
            
            // Iterator to go through all indices in the view
            let mut indices = vec![0; self.shape.rank()];
            loop {
                // Calculate source index
                let mut src_idx = self.offset;
                for (dim, &idx) in indices.iter().enumerate() {
                    src_idx += idx * self.strides[dim];
                }
                
                // Copy element
                dst[dst_idx] = src_slice[src_idx];
                dst_idx += 1;
                
                // Move to next index
                let mut dim = self.shape.rank() - 1;
                loop {
                    indices[dim] += 1;
                    if indices[dim] < self.shape[dim] {
                        break;
                    }
                    
                    indices[dim] = 0;
                    if dim == 0 {
                        return Ok(());  // Done with all elements
                    }
                    dim -= 1;
                }
            }
        }
        
        Ok(())
    }
    
    /// Creates a new view that is a subset of this view.
    pub fn subview(&self, start: &[usize], end: &[usize]) -> Result<Self> {
        if start.len() != self.shape.rank() || end.len() != self.shape.rank() {
            return Err(Error::DimensionMismatch(format!(
                "Start/end indices must match tensor rank: got {}/{}, expected {}",
                start.len(),
                end.len(),
                self.shape.rank()
            )));
        }
        
        // Validate start and end indices
        for (i, (&s, &e)) in start.iter().zip(end.iter()).enumerate() {
            let dim_size = self.shape[i];
            
            if s >= dim_size {
                return Err(Error::IndexOutOfBounds { 
                    index: s, 
                    axis: i, 
                    size: dim_size 
                });
            }
            
            if e > dim_size {
                return Err(Error::IndexOutOfBounds { 
                    index: e, 
                    axis: i, 
                    size: dim_size 
                });
            }
            
            if s >= e {
                return Err(Error::InvalidArgument(format!(
                    "Start index {} must be less than end index {} for dimension {}",
                    s, e, i
                )));
            }
        }
        
        // Compute global start indices by adding this view's start
        let global_start: Vec<usize> = start.iter()
            .zip(self.start.iter())
            .map(|(&s, &base)| base + s)
            .collect();
        
        // Compute global end indices
        let global_end: Vec<usize> = end.iter()
            .zip(self.start.iter())
            .map(|(&e, &base)| base + e)
            .collect();
        
        // Create new shape for the subview
        let new_dims: Vec<usize> = start.iter()
            .zip(end.iter())
            .map(|(&s, &e)| e - s)
            .collect();
        
        let new_shape = Shape::new(new_dims);
        
        // Calculate new offset
        let mut new_offset = self.offset;
        for (dim, &idx) in start.iter().enumerate() {
            new_offset += idx * self.strides[dim];
        }
        
        Ok(Self {
            source: self.source.clone(),
            start: global_start,
            shape: new_shape,
            offset: new_offset,
            strides: self.strides.clone(),
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::tensor::Tensor;
    
    #[test]
    fn test_tensor_view_creation() {
        // Create a 3x4 tensor with sequential values
        let data: Vec<f32> = (0..12).map(|i| i as f32).collect();
        let tensor = Tensor::from_slice(&data, [3, 4]).unwrap();
        
        // Create a view of the middle 2x2 region
        let view = TensorView::new(&tensor, &[0, 1], &[2, 3]).unwrap();
        
        assert_eq!(view.shape().dims(), &[2, 2]);
        assert_eq!(view.num_elements(), 4);
        
        // Extract data from the view
        let mut result = vec![0.0; 4];
        view.copy_to_slice(&mut result).unwrap();
        
        // Expected data:
        // Original:
        // [[ 0,  1,  2,  3],
        //  [ 4,  5,  6,  7],
        //  [ 8,  9, 10, 11]]
        //
        // View (0:2, 1:3):
        // [[ 1,  2],
        //  [ 5,  6]]
        assert_eq!(result, vec![1.0, 2.0, 5.0, 6.0]);
    }
    
    #[test]
    fn test_nested_views() {
        // Create a 4x4 tensor with sequential values
        let data: Vec<f32> = (0..16).map(|i| i as f32).collect();
        let tensor = Tensor::from_slice(&data, [4, 4]).unwrap();
        
        // Create a view of the 3x3 region starting at (1, 1)
        let view1 = TensorView::new(&tensor, &[1, 1], &[4, 4]).unwrap();
        
        // Create a subview of the 2x2 region starting at (0, 0) within the first view
        let view2 = view1.subview(&[0, 0], &[2, 2]).unwrap();
        
        assert_eq!(view2.shape().dims(), &[2, 2]);
        
        // Extract data from the nested view
        let mut result = vec![0.0; 4];
        view2.copy_to_slice(&mut result).unwrap();
        
        // Expected:
        // Original:
        // [[ 0,  1,  2,  3],
        //  [ 4,  5,  6,  7],
        //  [ 8,  9, 10, 11],
        //  [12, 13, 14, 15]]
        //
        // View1 (1:4, 1:4):
        // [[ 5,  6,  7],
        //  [ 9, 10, 11],
        //  [13, 14, 15]]
        //
        // View2 (0:2, 0:2) of View1:
        // [[ 5,  6],
        //  [ 9, 10]]
        assert_eq!(result, vec![5.0, 6.0, 9.0, 10.0]);
    }
}