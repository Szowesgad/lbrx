//! Operations for tensors.
//!
//! This module provides implementations of tensor operations,
//! including element-wise operations, matrix operations, and more.

use crate::dtype::DataType;
use crate::error::{Error, Result};
use crate::shape::Shape;
use crate::tensor::{Tensor, TensorStorage, Device};
use rayon::prelude::*;
use std::ops::{Add, Sub, Mul, Div};

/// Element-wise operations for tensors.
pub trait ElementWiseOp<T: DataType> {
    /// Performs element-wise addition.
    fn add(&self, other: &Tensor<T>) -> Result<Tensor<T>>;
    
    /// Performs element-wise subtraction.
    fn sub(&self, other: &Tensor<T>) -> Result<Tensor<T>>;
    
    /// Performs element-wise multiplication.
    fn mul(&self, other: &Tensor<T>) -> Result<Tensor<T>>;
    
    /// Performs element-wise division.
    fn div(&self, other: &Tensor<T>) -> Result<Tensor<T>>;
    
    /// Applies a function element-wise.
    fn map<F>(&self, f: F) -> Result<Tensor<T>> 
    where F: Fn(T) -> T + Send + Sync;
    
    /// Applies a function with indices element-wise.
    fn map_with_indices<F>(&self, f: F) -> Result<Tensor<T>>
    where F: Fn(T, &[usize]) -> T + Send + Sync;
    
    /// Applies a reduction operation along an axis.
    fn reduce<F>(&self, axis: usize, initial: T, f: F) -> Result<Tensor<T>>
    where F: Fn(T, T) -> T + Send + Sync;
}

/// Matrix operations for 2D tensors.
pub trait MatrixOp<T: DataType> {
    /// Performs matrix multiplication.
    fn matmul(&self, other: &Tensor<T>) -> Result<Tensor<T>>;
    
    /// Computes the transpose of the tensor.
    fn transpose(&self) -> Result<Tensor<T>>;
}

impl<T: DataType> ElementWiseOp<T> for Tensor<T> 
where T: Add<Output = T> + Sub<Output = T> + Mul<Output = T> + Div<Output = T> + Send + Sync
{
    fn add(&self, other: &Tensor<T>) -> Result<Tensor<T>> {
        // Handle broadcasting if shapes don't match
        if self.shape() != other.shape() {
            let broadcast_shape = self.shape().broadcast_to(other.shape())?;
            
            // Check if self needs to be broadcast
            let self_tensor = if self.shape() != &broadcast_shape {
                self.broadcast_to(&broadcast_shape)?
            } else {
                self.clone()
            };
            
            // Check if other needs to be broadcast
            let other_tensor = if other.shape() != &broadcast_shape {
                other.broadcast_to(&broadcast_shape)?
            } else {
                other.clone()
            };
            
            return self_tensor.add(&other_tensor);
        }
        
        // Move tensors to CPU for the operation
        let self_cpu = self.to_device(Device::Cpu)?;
        let other_cpu = other.to_device(Device::Cpu)?;
        
        let self_storage = match self_cpu.storage() {
            TensorStorage::Cpu(storage) => storage,
            _ => return Err(Error::Unsupported("Expected CPU storage".to_string())),
        };
        
        let other_storage = match other_cpu.storage() {
            TensorStorage::Cpu(storage) => storage,
            _ => return Err(Error::Unsupported("Expected CPU storage".to_string())),
        };
        
        let self_data = self_storage.as_slice();
        let other_data = other_storage.as_slice();
        
        let num_elements = self.num_elements();
        let mut result_data = vec![T::zero(); num_elements];
        
        // Parallel processing for large tensors
        if num_elements > 1024 {
            result_data.par_iter_mut().enumerate().for_each(|(i, result)| {
                *result = self_data[i] + other_data[i];
            });
        } else {
            for i in 0..num_elements {
                result_data[i] = self_data[i] + other_data[i];
            }
        }
        
        Tensor::from_slice(&result_data, self.shape().clone())
    }
    
    fn sub(&self, other: &Tensor<T>) -> Result<Tensor<T>> {
        // Similar implementation to add, but with subtraction
        // Handle broadcasting if shapes don't match
        if self.shape() != other.shape() {
            let broadcast_shape = self.shape().broadcast_to(other.shape())?;
            
            // Check if self needs to be broadcast
            let self_tensor = if self.shape() != &broadcast_shape {
                self.broadcast_to(&broadcast_shape)?
            } else {
                self.clone()
            };
            
            // Check if other needs to be broadcast
            let other_tensor = if other.shape() != &broadcast_shape {
                other.broadcast_to(&broadcast_shape)?
            } else {
                other.clone()
            };
            
            return self_tensor.sub(&other_tensor);
        }
        
        // Move tensors to CPU for the operation
        let self_cpu = self.to_device(Device::Cpu)?;
        let other_cpu = other.to_device(Device::Cpu)?;
        
        let self_storage = match self_cpu.storage() {
            TensorStorage::Cpu(storage) => storage,
            _ => return Err(Error::Unsupported("Expected CPU storage".to_string())),
        };
        
        let other_storage = match other_cpu.storage() {
            TensorStorage::Cpu(storage) => storage,
            _ => return Err(Error::Unsupported("Expected CPU storage".to_string())),
        };
        
        let self_data = self_storage.as_slice();
        let other_data = other_storage.as_slice();
        
        let num_elements = self.num_elements();
        let mut result_data = vec![T::zero(); num_elements];
        
        // Parallel processing for large tensors
        if num_elements > 1024 {
            result_data.par_iter_mut().enumerate().for_each(|(i, result)| {
                *result = self_data[i] - other_data[i];
            });
        } else {
            for i in 0..num_elements {
                result_data[i] = self_data[i] - other_data[i];
            }
        }
        
        Tensor::from_slice(&result_data, self.shape().clone())
    }
    
    fn mul(&self, other: &Tensor<T>) -> Result<Tensor<T>> {
        // Similar implementation to add, but with multiplication
        // Handle broadcasting if shapes don't match
        if self.shape() != other.shape() {
            let broadcast_shape = self.shape().broadcast_to(other.shape())?;
            
            // Check if self needs to be broadcast
            let self_tensor = if self.shape() != &broadcast_shape {
                self.broadcast_to(&broadcast_shape)?
            } else {
                self.clone()
            };
            
            // Check if other needs to be broadcast
            let other_tensor = if other.shape() != &broadcast_shape {
                other.broadcast_to(&broadcast_shape)?
            } else {
                other.clone()
            };
            
            return self_tensor.mul(&other_tensor);
        }
        
        // Move tensors to CPU for the operation
        let self_cpu = self.to_device(Device::Cpu)?;
        let other_cpu = other.to_device(Device::Cpu)?;
        
        let self_storage = match self_cpu.storage() {
            TensorStorage::Cpu(storage) => storage,
            _ => return Err(Error::Unsupported("Expected CPU storage".to_string())),
        };
        
        let other_storage = match other_cpu.storage() {
            TensorStorage::Cpu(storage) => storage,
            _ => return Err(Error::Unsupported("Expected CPU storage".to_string())),
        };
        
        let self_data = self_storage.as_slice();
        let other_data = other_storage.as_slice();
        
        let num_elements = self.num_elements();
        let mut result_data = vec![T::zero(); num_elements];
        
        // Parallel processing for large tensors
        if num_elements > 1024 {
            result_data.par_iter_mut().enumerate().for_each(|(i, result)| {
                *result = self_data[i] * other_data[i];
            });
        } else {
            for i in 0..num_elements {
                result_data[i] = self_data[i] * other_data[i];
            }
        }
        
        Tensor::from_slice(&result_data, self.shape().clone())
    }
    
    fn div(&self, other: &Tensor<T>) -> Result<Tensor<T>> {
        // Similar implementation to add, but with division
        // Handle broadcasting if shapes don't match
        if self.shape() != other.shape() {
            let broadcast_shape = self.shape().broadcast_to(other.shape())?;
            
            // Check if self needs to be broadcast
            let self_tensor = if self.shape() != &broadcast_shape {
                self.broadcast_to(&broadcast_shape)?
            } else {
                self.clone()
            };
            
            // Check if other needs to be broadcast
            let other_tensor = if other.shape() != &broadcast_shape {
                other.broadcast_to(&broadcast_shape)?
            } else {
                other.clone()
            };
            
            return self_tensor.div(&other_tensor);
        }
        
        // Move tensors to CPU for the operation
        let self_cpu = self.to_device(Device::Cpu)?;
        let other_cpu = other.to_device(Device::Cpu)?;
        
        let self_storage = match self_cpu.storage() {
            TensorStorage::Cpu(storage) => storage,
            _ => return Err(Error::Unsupported("Expected CPU storage".to_string())),
        };
        
        let other_storage = match other_cpu.storage() {
            TensorStorage::Cpu(storage) => storage,
            _ => return Err(Error::Unsupported("Expected CPU storage".to_string())),
        };
        
        let self_data = self_storage.as_slice();
        let other_data = other_storage.as_slice();
        
        let num_elements = self.num_elements();
        let mut result_data = vec![T::zero(); num_elements];
        
        // Parallel processing for large tensors
        if num_elements > 1024 {
            result_data.par_iter_mut().enumerate().for_each(|(i, result)| {
                *result = self_data[i] / other_data[i];
            });
        } else {
            for i in 0..num_elements {
                result_data[i] = self_data[i] / other_data[i];
            }
        }
        
        Tensor::from_slice(&result_data, self.shape().clone())
    }
    
    fn map<F>(&self, f: F) -> Result<Tensor<T>> 
    where F: Fn(T) -> T + Send + Sync
    {
        // Move tensor to CPU for the operation
        let self_cpu = self.to_device(Device::Cpu)?;
        
        let self_storage = match self_cpu.storage() {
            TensorStorage::Cpu(storage) => storage,
            _ => return Err(Error::Unsupported("Expected CPU storage".to_string())),
        };
        
        let self_data = self_storage.as_slice();
        
        let num_elements = self.num_elements();
        let mut result_data = vec![T::zero(); num_elements];
        
        // Parallel processing for large tensors
        if num_elements > 1024 {
            result_data.par_iter_mut().enumerate().for_each(|(i, result)| {
                *result = f(self_data[i]);
            });
        } else {
            for i in 0..num_elements {
                result_data[i] = f(self_data[i]);
            }
        }
        
        Tensor::from_slice(&result_data, self.shape().clone())
    }
    
    fn map_with_indices<F>(&self, f: F) -> Result<Tensor<T>>
    where F: Fn(T, &[usize]) -> T + Send + Sync
    {
        // Move tensor to CPU for the operation
        let self_cpu = self.to_device(Device::Cpu)?;
        
        let self_storage = match self_cpu.storage() {
            TensorStorage::Cpu(storage) => storage,
            _ => return Err(Error::Unsupported("Expected CPU storage".to_string())),
        };
        
        let self_data = self_storage.as_slice();
        
        let num_elements = self.num_elements();
        let mut result_data = vec![T::zero(); num_elements];
        
        // Process each element with its index
        for i in 0..num_elements {
            let indices = self.shape().indices_for(i)?;
            result_data[i] = f(self_data[i], &indices);
        }
        
        Tensor::from_slice(&result_data, self.shape().clone())
    }
    
    fn reduce<F>(&self, axis: usize, initial: T, f: F) -> Result<Tensor<T>>
    where F: Fn(T, T) -> T + Send + Sync
    {
        if axis >= self.shape().rank() {
            return Err(Error::IndexOutOfBounds {
                index: axis,
                axis: 0,
                size: self.shape().rank(),
            });
        }
        
        // Move tensor to CPU for the operation
        let self_cpu = self.to_device(Device::Cpu)?;
        
        let self_storage = match self_cpu.storage() {
            TensorStorage::Cpu(storage) => storage,
            _ => return Err(Error::Unsupported("Expected CPU storage".to_string())),
        };
        
        let self_data = self_storage.as_slice();
        
        // Create shape for result tensor (remove the specified axis)
        let mut result_dims = self.shape().dims().to_vec();
        let axis_size = result_dims.remove(axis);
        
        let result_shape = Shape::new(result_dims);
        let result_size = result_shape.num_elements();
        let mut result_data = vec![initial; result_size];
        
        // For each element in the result, reduce along the specified axis
        for i in 0..result_size {
            let mut result_indices = result_shape.indices_for(i)?;
            result_indices.insert(axis, 0);
            
            let mut acc = initial;
            
            for j in 0..axis_size {
                result_indices[axis] = j;
                let src_idx = self.shape().index_for(&result_indices)?;
                acc = f(acc, self_data[src_idx]);
            }
            
            result_data[i] = acc;
        }
        
        Tensor::from_slice(&result_data, result_shape)
    }
}

impl<T: DataType> MatrixOp<T> for Tensor<T>
where T: Add<Output = T> + Mul<Output = T> + Send + Sync
{
    fn matmul(&self, other: &Tensor<T>) -> Result<Tensor<T>> {
        // Check if both tensors are matrices (2D)
        if self.shape().rank() != 2 || other.shape().rank() != 2 {
            return Err(Error::DimensionMismatch(
                "Matrix multiplication requires 2D tensors".to_string()
            ));
        }
        
        // Check dimension compatibility: (M,K) * (K,N) = (M,N)
        let (m, k1) = self.shape().dims_2d()?;
        let (k2, n) = other.shape().dims_2d()?;
        
        if k1 != k2 {
            return Err(Error::DimensionMismatch(format!(
                "Matrix dimensions incompatible for multiplication: ({},{}) and ({},{})",
                m, k1, k2, n
            )));
        }
        
        // Move tensors to CPU for the operation
        let self_cpu = self.to_device(Device::Cpu)?;
        let other_cpu = other.to_device(Device::Cpu)?;
        
        let self_storage = match self_cpu.storage() {
            TensorStorage::Cpu(storage) => storage,
            _ => return Err(Error::Unsupported("Expected CPU storage".to_string())),
        };
        
        let other_storage = match other_cpu.storage() {
            TensorStorage::Cpu(storage) => storage,
            _ => return Err(Error::Unsupported("Expected CPU storage".to_string())),
        };
        
        let self_data = self_storage.as_slice();
        let other_data = other_storage.as_slice();
        
        let mut result_data = vec![T::zero(); m * n];
        
        // Simple matrix multiplication (not optimized)
        if m > 64 && n > 64 && k1 > 64 {
            // For large matrices, use parallel processing
            result_data.par_chunks_mut(n).enumerate().for_each(|(i, row)| {
                for j in 0..n {
                    let mut sum = T::zero();
                    for k in 0..k1 {
                        sum = sum + self_data[i * k1 + k] * other_data[k * n + j];
                    }
                    row[j] = sum;
                }
            });
        } else {
            // For small matrices, use simple loop
            for i in 0..m {
                for j in 0..n {
                    let mut sum = T::zero();
                    for k in 0..k1 {
                        sum = sum + self_data[i * k1 + k] * other_data[k * n + j];
                    }
                    result_data[i * n + j] = sum;
                }
            }
        }
        
        Tensor::from_slice(&result_data, [m, n])
    }
    
    fn transpose(&self) -> Result<Tensor<T>> {
        if self.shape().rank() != 2 {
            return Err(Error::DimensionMismatch(
                "Transpose requires a 2D tensor".to_string()
            ));
        }
        
        let (m, n) = self.shape().dims_2d()?;
        
        // Move tensor to CPU for the operation
        let self_cpu = self.to_device(Device::Cpu)?;
        
        let self_storage = match self_cpu.storage() {
            TensorStorage::Cpu(storage) => storage,
            _ => return Err(Error::Unsupported("Expected CPU storage".to_string())),
        };
        
        let self_data = self_storage.as_slice();
        
        let mut result_data = vec![T::zero(); m * n];
        
        for i in 0..m {
            for j in 0..n {
                result_data[j * m + i] = self_data[i * n + j];
            }
        }
        
        Tensor::from_slice(&result_data, [n, m])
    }
}

/// Broadcasts a tensor to a specified shape.
impl<T: DataType> Tensor<T> {
    /// Broadcasts this tensor to match the target shape.
    pub fn broadcast_to(&self, target_shape: &Shape) -> Result<Self> {
        if self.shape() == target_shape {
            return Ok(self.clone());
        }
        
        let broadcast_shape = self.shape().broadcast_to(target_shape)?;
        
        // Move to CPU for the operation
        let self_cpu = self.to_device(Device::Cpu)?;
        
        let self_storage = match self_cpu.storage() {
            TensorStorage::Cpu(storage) => storage,
            _ => return Err(Error::Unsupported("Expected CPU storage".to_string())),
        };
        
        let self_data = self_storage.as_slice();
        
        let num_elements = broadcast_shape.num_elements();
        let mut result_data = vec![T::zero(); num_elements];
        
        // Initialize source and target dimensions/strides from right to left
        let src_dims = self.shape().dims().to_vec();
        let src_rank = src_dims.len();
        let src_strides = self.shape().strides().to_vec();
        
        let tgt_dims = broadcast_shape.dims().to_vec();
        let tgt_rank = tgt_dims.len();
        
        // For each element in the target, find the corresponding element in the source
        for i in 0..num_elements {
            let tgt_indices = broadcast_shape.indices_for(i)?;
            
            // Map target indices to source indices
            let mut src_idx = 0;
            for dim in 0..src_rank {
                // Get the corresponding dimension in the target, accounting for broadcasting
                let tgt_dim = tgt_rank - src_rank + dim;
                if tgt_dim >= 0 {
                    // Extract the target index and convert to source index
                    let tgt_idx = tgt_indices[tgt_dim];
                    // Use modulo for dimensions that are being broadcast
                    let src_idx_dim = tgt_idx % src_dims[dim];
                    src_idx += src_idx_dim * src_strides[dim];
                }
            }
            
            result_data[i] = self_data[src_idx];
        }
        
        Tensor::from_slice(&result_data, broadcast_shape)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    
    #[test]
    fn test_element_wise_add() {
        let a = Tensor::from_slice(&[1.0, 2.0, 3.0, 4.0], [2, 2]).unwrap();
        let b = Tensor::from_slice(&[5.0, 6.0, 7.0, 8.0], [2, 2]).unwrap();
        
        let result = a.add(&b).unwrap();
        
        let expected = Tensor::from_slice(&[6.0, 8.0, 10.0, 12.0], [2, 2]).unwrap();
        
        // Compare tensors
        let result_storage = match result.storage() {
            TensorStorage::Cpu(storage) => storage,
            _ => panic!("Expected CPU storage"),
        };
        
        let expected_storage = match expected.storage() {
            TensorStorage::Cpu(storage) => storage,
            _ => panic!("Expected CPU storage"),
        };
        
        assert_eq!(result_storage.as_slice(), expected_storage.as_slice());
    }
    
    #[test]
    fn test_broadcasting() {
        // Create a 2x1 tensor
        let a = Tensor::from_slice(&[1.0, 2.0], [2, 1]).unwrap();
        
        // Create a 1x3 tensor
        let b = Tensor::from_slice(&[5.0, 6.0, 7.0], [1, 3]).unwrap();
        
        // Add them together (should broadcast to 2x3)
        let result = a.add(&b).unwrap();
        
        assert_eq!(result.shape().dims(), &[2, 3]);
        
        // Expected result:
        // [[1, 1, 1] + [5, 6, 7] = [6, 7, 8]
        //  [2, 2, 2] + [5, 6, 7] = [7, 8, 9]]
        let expected = Tensor::from_slice(&[6.0, 7.0, 8.0, 7.0, 8.0, 9.0], [2, 3]).unwrap();
        
        // Compare tensors
        let result_storage = match result.storage() {
            TensorStorage::Cpu(storage) => storage,
            _ => panic!("Expected CPU storage"),
        };
        
        let expected_storage = match expected.storage() {
            TensorStorage::Cpu(storage) => storage,
            _ => panic!("Expected CPU storage"),
        };
        
        assert_eq!(result_storage.as_slice(), expected_storage.as_slice());
    }
    
    #[test]
    fn test_matrix_multiply() {
        // 2x3 matrix
        let a = Tensor::from_slice(&[1.0, 2.0, 3.0, 4.0, 5.0, 6.0], [2, 3]).unwrap();
        
        // 3x2 matrix
        let b = Tensor::from_slice(&[7.0, 8.0, 9.0, 10.0, 11.0, 12.0], [3, 2]).unwrap();
        
        // Multiply them
        let result = a.matmul(&b).unwrap();
        
        // Expected (2x2) result:
        // [1*7 + 2*9 + 3*11, 1*8 + 2*10 + 3*12]   = [58, 64]
        // [4*7 + 5*9 + 6*11, 4*8 + 5*10 + 6*12]   = [139, 154]
        let expected = Tensor::from_slice(&[58.0, 64.0, 139.0, 154.0], [2, 2]).unwrap();
        
        // Compare tensors
        let result_storage = match result.storage() {
            TensorStorage::Cpu(storage) => storage,
            _ => panic!("Expected CPU storage"),
        };
        
        let expected_storage = match expected.storage() {
            TensorStorage::Cpu(storage) => storage,
            _ => panic!("Expected CPU storage"),
        };
        
        assert_eq!(result_storage.as_slice(), expected_storage.as_slice());
    }
    
    #[test]
    fn test_matrix_transpose() {
        // 2x3 matrix
        let a = Tensor::from_slice(&[1.0, 2.0, 3.0, 4.0, 5.0, 6.0], [2, 3]).unwrap();
        
        // Transpose to 3x2
        let result = a.transpose().unwrap();
        
        assert_eq!(result.shape().dims(), &[3, 2]);
        
        // Expected (3x2) result:
        // [[1, 4],
        //  [2, 5],
        //  [3, 6]]
        let expected = Tensor::from_slice(&[1.0, 4.0, 2.0, 5.0, 3.0, 6.0], [3, 2]).unwrap();
        
        // Compare tensors
        let result_storage = match result.storage() {
            TensorStorage::Cpu(storage) => storage,
            _ => panic!("Expected CPU storage"),
        };
        
        let expected_storage = match expected.storage() {
            TensorStorage::Cpu(storage) => storage,
            _ => panic!("Expected CPU storage"),
        };
        
        assert_eq!(result_storage.as_slice(), expected_storage.as_slice());
    }
    
    #[test]
    fn test_map_function() {
        let a = Tensor::from_slice(&[1.0, 2.0, 3.0, 4.0], [2, 2]).unwrap();
        
        // Square each element
        let result = a.map(|x| x * x).unwrap();
        
        let expected = Tensor::from_slice(&[1.0, 4.0, 9.0, 16.0], [2, 2]).unwrap();
        
        // Compare tensors
        let result_storage = match result.storage() {
            TensorStorage::Cpu(storage) => storage,
            _ => panic!("Expected CPU storage"),
        };
        
        let expected_storage = match expected.storage() {
            TensorStorage::Cpu(storage) => storage,
            _ => panic!("Expected CPU storage"),
        };
        
        assert_eq!(result_storage.as_slice(), expected_storage.as_slice());
    }
    
    #[test]
    fn test_reduce_sum() {
        let a = Tensor::from_slice(&[1.0, 2.0, 3.0, 4.0, 5.0, 6.0], [2, 3]).unwrap();
        
        // Sum along axis 0 (rows)
        let result = a.reduce(0, 0.0, |acc, x| acc + x).unwrap();
        
        assert_eq!(result.shape().dims(), &[3]);
        
        // Expected result: [1+4, 2+5, 3+6] = [5, 7, 9]
        let expected = Tensor::from_slice(&[5.0, 7.0, 9.0], [3]).unwrap();
        
        // Compare tensors
        let result_storage = match result.storage() {
            TensorStorage::Cpu(storage) => storage,
            _ => panic!("Expected CPU storage"),
        };
        
        let expected_storage = match expected.storage() {
            TensorStorage::Cpu(storage) => storage,
            _ => panic!("Expected CPU storage"),
        };
        
        assert_eq!(result_storage.as_slice(), expected_storage.as_slice());
    }
}