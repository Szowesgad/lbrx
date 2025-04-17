//! Shape handling for tensors.
//!
//! This module provides the `Shape` struct for representing and
//! manipulating tensor shapes, along with related utilities.

use std::fmt;
use std::ops::{Index, IndexMut};
use crate::error::{Error, Result};

/// Maximum number of dimensions supported
pub const MAX_DIMS: usize = 8;

/// Shape of a tensor, defining its dimensionality
#[derive(Clone, Debug, PartialEq, Eq, Hash)]
pub struct Shape {
    /// Dimensions of the shape
    dims: Vec<usize>,
    
    /// Strides for each dimension (in elements, not bytes)
    strides: Vec<usize>,
}

impl Shape {
    /// Creates a new shape with the specified dimensions.
    pub fn new<T: AsRef<[usize]>>(dims: T) -> Self {
        let dims = dims.as_ref().to_vec();
        let strides = Self::compute_strides(&dims);
        
        Self { dims, strides }
    }
    
    /// Computes strides for a shape with the given dimensions.
    fn compute_strides(dims: &[usize]) -> Vec<usize> {
        let mut strides = vec![1; dims.len()];
        
        // Compute row-major (C-style) strides
        for i in (0..dims.len()).rev().skip(1) {
            strides[i] = strides[i + 1] * dims[i + 1];
        }
        
        strides
    }
    
    /// Returns the number of dimensions in the shape.
    pub fn rank(&self) -> usize {
        self.dims.len()
    }
    
    /// Returns the dimensions of the shape.
    pub fn dims(&self) -> &[usize] {
        &self.dims
    }
    
    /// Returns the strides of the shape.
    pub fn strides(&self) -> &[usize] {
        &self.strides
    }
    
    /// Returns the total number of elements in a tensor with this shape.
    pub fn num_elements(&self) -> usize {
        if self.dims.is_empty() {
            return 0;
        }
        
        self.dims.iter().product()
    }
    
    /// Returns the size of the specified dimension.
    pub fn size(&self, dim: usize) -> Result<usize> {
        if dim >= self.rank() {
            return Err(Error::IndexOutOfBounds { 
                index: dim, 
                axis: 0, 
                size: self.rank() 
            });
        }
        
        Ok(self.dims[dim])
    }
    
    /// Returns the stride of the specified dimension.
    pub fn stride(&self, dim: usize) -> Result<usize> {
        if dim >= self.rank() {
            return Err(Error::IndexOutOfBounds { 
                index: dim, 
                axis: 0, 
                size: self.rank() 
            });
        }
        
        Ok(self.strides[dim])
    }
    
    /// Creates a new shape with dimensions from another shape.
    pub fn from_shape(shape: &Shape) -> Self {
        Self::new(shape.dims())
    }
    
    /// Creates a new shape with the specified rank, all dimensions set to 1.
    pub fn ones(rank: usize) -> Self {
        Self::new(vec![1; rank])
    }
    
    /// Creates a new scalar shape (rank 0).
    pub fn scalar() -> Self {
        Self::new(Vec::<usize>::new())
    }
    
    /// Computes the linear index for a set of indices.
    pub fn index_for(&self, indices: &[usize]) -> Result<usize> {
        if indices.len() != self.rank() {
            return Err(Error::DimensionMismatch(format!(
                "Expected {} indices but got {}", 
                self.rank(), 
                indices.len()
            )));
        }
        
        // Check bounds
        for (i, &idx) in indices.iter().enumerate() {
            if idx >= self.dims[i] {
                return Err(Error::IndexOutOfBounds { 
                    index: idx, 
                    axis: i, 
                    size: self.dims[i] 
                });
            }
        }
        
        // Compute linear index
        let mut index = 0;
        for i in 0..indices.len() {
            index += indices[i] * self.strides[i];
        }
        
        Ok(index)
    }
    
    /// Computes the multi-dimensional indices for a linear index.
    pub fn indices_for(&self, index: usize) -> Result<Vec<usize>> {
        if index >= self.num_elements() {
            return Err(Error::IndexOutOfBounds { 
                index, 
                axis: 0, 
                size: self.num_elements() 
            });
        }
        
        let mut indices = vec![0; self.rank()];
        let mut remaining = index;
        
        for i in 0..self.rank() {
            indices[i] = remaining / self.strides[i];
            remaining %= self.strides[i];
        }
        
        Ok(indices)
    }
    
    /// Returns a shape for a slice of this shape.
    pub fn slice(&self, start: &[usize], end: &[usize]) -> Result<Self> {
        if start.len() != self.rank() || end.len() != self.rank() {
            return Err(Error::DimensionMismatch(format!(
                "Expected {} indices but got {} and {}", 
                self.rank(), 
                start.len(), 
                end.len()
            )));
        }
        
        let mut new_dims = Vec::with_capacity(self.rank());
        
        for i in 0..self.rank() {
            if start[i] >= self.dims[i] {
                return Err(Error::IndexOutOfBounds { 
                    index: start[i], 
                    axis: i, 
                    size: self.dims[i] 
                });
            }
            
            if end[i] > self.dims[i] {
                return Err(Error::IndexOutOfBounds { 
                    index: end[i], 
                    axis: i, 
                    size: self.dims[i] 
                });
            }
            
            if start[i] > end[i] {
                return Err(Error::InvalidArgument(format!(
                    "Start index {} is greater than end index {} for dimension {}", 
                    start[i], 
                    end[i], 
                    i
                )));
            }
            
            new_dims.push(end[i] - start[i]);
        }
        
        Ok(Self::new(new_dims))
    }
    
    /// Returns whether the shape is contiguous.
    pub fn is_contiguous(&self) -> bool {
        let computed_strides = Self::compute_strides(&self.dims);
        self.strides == computed_strides
    }
    
    /// Returns whether the shape is a scalar.
    pub fn is_scalar(&self) -> bool {
        self.rank() == 0
    }
    
    /// Returns whether the shape is a vector (rank 1).
    pub fn is_vector(&self) -> bool {
        self.rank() == 1
    }
    
    /// Returns whether the shape is a matrix (rank 2).
    pub fn is_matrix(&self) -> bool {
        self.rank() == 2
    }
    
    /// Returns the dimensions as a 2D shape (M, N).
    pub fn dims_2d(&self) -> Result<(usize, usize)> {
        if self.rank() != 2 {
            return Err(Error::DimensionMismatch(format!(
                "Expected 2D shape but got shape with {} dimensions", 
                self.rank()
            )));
        }
        
        Ok((self.dims[0], self.dims[1]))
    }
    
    /// Broadcasts this shape to be compatible with another shape.
    ///
    /// Broadcasting follows NumPy's broadcasting rules:
    /// 1. Dimensions are aligned from the right
    /// 2. Size-1 dimensions are stretched to match the other shape
    /// 3. New dimensions of size 1 are added if ranks don't match
    pub fn broadcast_to(&self, other: &Shape) -> Result<Shape> {
        // Determine the rank of the result
        let result_rank = std::cmp::max(self.rank(), other.rank());
        
        // Initialize the result dimensions
        let mut result_dims = vec![0; result_rank];
        
        // Align dimensions from the right
        for i in 0..result_rank {
            let self_dim = if i < self.rank() {
                self.dims[self.rank() - 1 - i]
            } else {
                1 // Implicit size-1 dimension
            };
            
            let other_dim = if i < other.rank() {
                other.dims[other.rank() - 1 - i]
            } else {
                1 // Implicit size-1 dimension
            };
            
            // Check if dimensions are compatible
            if self_dim != 1 && other_dim != 1 && self_dim != other_dim {
                return Err(Error::DimensionMismatch(format!(
                    "Cannot broadcast shapes: dimension mismatch at index {} ({} vs {})", 
                    result_rank - 1 - i, 
                    self_dim, 
                    other_dim
                )));
            }
            
            // Use the larger of the two dimensions
            result_dims[result_rank - 1 - i] = std::cmp::max(self_dim, other_dim);
        }
        
        Ok(Shape::new(result_dims))
    }
    
    /// Reshape the tensor to a new shape with the same number of elements.
    pub fn reshape(&self, new_dims: &[usize]) -> Result<Shape> {
        let new_num_elements: usize = new_dims.iter().product();
        
        if new_num_elements != self.num_elements() {
            return Err(Error::ShapeMismatch { 
                expected: self.num_elements(), 
                actual: new_num_elements 
            });
        }
        
        Ok(Shape::new(new_dims))
    }
}

impl<T: AsRef<[usize]>> From<T> for Shape {
    fn from(dims: T) -> Self {
        Self::new(dims)
    }
}

impl Index<usize> for Shape {
    type Output = usize;
    
    fn index(&self, index: usize) -> &Self::Output {
        &self.dims[index]
    }
}

impl IndexMut<usize> for Shape {
    fn index_mut(&mut self, index: usize) -> &mut Self::Output {
        &mut self.dims[index]
    }
}

impl fmt::Display for Shape {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "(")?;
        
        for (i, dim) in self.dims.iter().enumerate() {
            if i > 0 {
                write!(f, ", ")?;
            }
            write!(f, "{}", dim)?;
        }
        
        write!(f, ")")
    }
}

/// A range within a tensor, used for slicing.
#[derive(Clone, Debug)]
pub struct TensorRange {
    /// Start indices for each dimension
    pub start: Vec<usize>,
    
    /// End indices for each dimension
    pub end: Vec<usize>,
}

impl TensorRange {
    /// Creates a new range with the given start and end indices.
    pub fn new(start: Vec<usize>, end: Vec<usize>) -> Self {
        Self { start, end }
    }
    
    /// Creates a range that selects the entire tensor.
    pub fn full(shape: &Shape) -> Self {
        let rank = shape.rank();
        let start = vec![0; rank];
        let end = shape.dims().to_vec();
        
        Self { start, end }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    
    #[test]
    fn test_shape_creation() {
        let shape = Shape::new([2, 3, 4]);
        
        assert_eq!(shape.rank(), 3);
        assert_eq!(shape.dims(), &[2, 3, 4]);
        assert_eq!(shape.num_elements(), 24);
    }
    
    #[test]
    fn test_shape_strides() {
        let shape = Shape::new([2, 3, 4]);
        
        // Row-major strides: (12, 4, 1)
        assert_eq!(shape.strides(), &[12, 4, 1]);
    }
    
    #[test]
    fn test_linear_indexing() {
        let shape = Shape::new([2, 3, 4]);
        
        // Index for [0, 0, 0]
        assert_eq!(shape.index_for(&[0, 0, 0]).unwrap(), 0);
        
        // Index for [1, 0, 0]
        assert_eq!(shape.index_for(&[1, 0, 0]).unwrap(), 12);
        
        // Index for [1, 2, 3]
        assert_eq!(shape.index_for(&[1, 2, 3]).unwrap(), 23);
    }
    
    #[test]
    fn test_multi_indexing() {
        let shape = Shape::new([2, 3, 4]);
        
        // Indices for linear index 0
        assert_eq!(shape.indices_for(0).unwrap(), vec![0, 0, 0]);
        
        // Indices for linear index 12
        assert_eq!(shape.indices_for(12).unwrap(), vec![1, 0, 0]);
        
        // Indices for linear index 23
        assert_eq!(shape.indices_for(23).unwrap(), vec![1, 2, 3]);
    }
    
    #[test]
    fn test_slicing() {
        let shape = Shape::new([10, 10]);
        
        // Slice [2:5, 3:7]
        let slice = shape.slice(&[2, 3], &[5, 7]).unwrap();
        
        assert_eq!(slice.dims(), &[3, 4]);
        assert_eq!(slice.strides(), &[4, 1]);  // Same strides as original
    }
    
    #[test]
    fn test_broadcasting() {
        // Broadcasting scalar to 3D tensor
        let scalar = Shape::scalar();
        let tensor = Shape::new([2, 3, 4]);
        
        let result = scalar.broadcast_to(&tensor).unwrap();
        assert_eq!(result.dims(), &[2, 3, 4]);
        
        // Broadcasting matrices
        let mat1 = Shape::new([3, 1]);
        let mat2 = Shape::new([1, 4]);
        
        let result = mat1.broadcast_to(&mat2).unwrap();
        assert_eq!(result.dims(), &[3, 4]);
        
        // Broadcasting failure
        let shape1 = Shape::new([2, 3]);
        let shape2 = Shape::new([3, 4]);
        
        assert!(shape1.broadcast_to(&shape2).is_err());
    }
}