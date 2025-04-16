//! Tensor implementation for efficient numeric operations.
//!
//! This module provides the core `Tensor` abstraction that serves as the
//! foundation for all numeric operations in LBRX-MLX.

mod storage;
mod ops;
mod view;

#[cfg(feature = "metal-backend")]
pub mod metal;

use crate::dtype::DataType;
use crate::error::Error;
use crate::shape::Shape;
use std::sync::Arc;

pub use storage::{TensorStorage, CpuStorage, MmapStorage};
pub use ops::{ElementWiseOp, MatrixOp};
pub use view::TensorView;

/// Tensor metadata for tracking tensor information.
#[derive(Clone, Debug, Default)]
pub struct TensorMetadata {
    /// Optional name for the tensor
    pub name: Option<String>,
    
    /// Source of the tensor (file, model, computation)
    pub source: Option<String>,
    
    /// Quantization information if applicable
    pub quantization: Option<QuantizationInfo>,
    
    /// Custom metadata as key-value pairs
    pub custom: std::collections::HashMap<String, String>,
}

/// Quantization information for tensors.
#[derive(Clone, Debug)]
pub struct QuantizationInfo {
    /// Number of bits per element
    pub bits: u8,
    
    /// Group size for block-wise quantization
    pub group_size: u32,
    
    /// Quantization method used
    pub method: QuantizationMethod,
    
    /// Scale factors for dequantization
    pub scales: Option<Arc<Tensor<f32>>>,
    
    /// Zero points for dequantization
    pub zero_points: Option<Arc<Tensor<i32>>>,
}

/// Quantization methods supported by the system.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum QuantizationMethod {
    /// Linear quantization
    Linear,
    
    /// Absolute max quantization
    AbsMax,
    
    /// Block-wise quantization
    BlockWise,
    
    /// Mixed precision quantization
    MixedPrecision(u8, u8),
}

/// Options for tensor creation and manipulation.
#[derive(Clone, Debug, Default)]
pub struct TensorOptions {
    /// Optional name for the tensor
    pub name: Option<String>,
    
    /// Device to store the tensor on
    pub device: Device,
    
    /// Data type for the tensor elements
    pub dtype: Option<DataTypeEnum>,
    
    /// Memory layout for the tensor
    pub layout: MemoryLayout,
    
    /// Quantization options if applicable
    pub quantization: Option<QuantizationInfo>,
}

/// Device types for tensor storage.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Device {
    /// CPU storage
    Cpu,
    
    /// GPU (Metal) storage
    Metal,
    
    /// Memory-mapped file storage
    Mmap,
}

impl Default for Device {
    fn default() -> Self {
        Device::Cpu
    }
}

/// Memory layout for tensor data.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum MemoryLayout {
    /// Row-major (C-style) layout
    RowMajor,
    
    /// Column-major (Fortran-style) layout
    ColumnMajor,
    
    /// Custom strided layout
    Strided,
}

impl Default for MemoryLayout {
    fn default() -> Self {
        MemoryLayout::RowMajor
    }
}

/// The main tensor type representing a multidimensional array.
#[derive(Clone, Debug)]
pub struct Tensor<T: DataType> {
    /// Shape of the tensor
    shape: Shape,
    
    /// Storage for the tensor data
    storage: TensorStorage<T>,
    
    /// Additional metadata
    metadata: TensorMetadata,
}

impl<T: DataType> Tensor<T> {
    /// Creates a new tensor with the specified shape and storage.
    pub fn new(shape: Shape, storage: TensorStorage<T>, metadata: TensorMetadata) -> Self {
        Self {
            shape,
            storage,
            metadata,
        }
    }
    
    /// Creates a new tensor filled with zeros.
    pub fn zeros(shape: impl Into<Shape>) -> Result<Self, Error> {
        let shape = shape.into();
        let num_elements = shape.num_elements();
        
        let storage = CpuStorage::new(num_elements)?;
        
        Ok(Self {
            shape,
            storage: TensorStorage::Cpu(storage),
            metadata: TensorMetadata::default(),
        })
    }
    
    /// Creates a new tensor filled with ones.
    pub fn ones(shape: impl Into<Shape>) -> Result<Self, Error> {
        let shape = shape.into();
        let num_elements = shape.num_elements();
        
        let mut storage = CpuStorage::new(num_elements)?;
        
        // Fill with ones
        let slice = storage.as_mut_slice();
        for i in 0..num_elements {
            slice[i] = T::one();
        }
        
        Ok(Self {
            shape,
            storage: TensorStorage::Cpu(storage),
            metadata: TensorMetadata::default(),
        })
    }
    
    /// Creates a tensor from an existing slice of data.
    pub fn from_slice(data: &[T], shape: impl Into<Shape>) -> Result<Self, Error> {
        let shape = shape.into();
        let num_elements = shape.num_elements();
        
        if data.len() != num_elements {
            return Err(Error::ShapeMismatch {
                expected: num_elements,
                actual: data.len(),
            });
        }
        
        let mut storage = CpuStorage::new(num_elements)?;
        storage.as_mut_slice().copy_from_slice(data);
        
        Ok(Self {
            shape,
            storage: TensorStorage::Cpu(storage),
            metadata: TensorMetadata::default(),
        })
    }
    
    /// Returns the shape of the tensor.
    pub fn shape(&self) -> &Shape {
        &self.shape
    }
    
    /// Returns the number of elements in the tensor.
    pub fn num_elements(&self) -> usize {
        self.shape.num_elements()
    }
    
    /// Returns a reference to the tensor's storage.
    pub fn storage(&self) -> &TensorStorage<T> {
        &self.storage
    }
    
    /// Returns a mutable reference to the tensor's storage.
    pub fn storage_mut(&mut self) -> &mut TensorStorage<T> {
        &mut self.storage
    }
    
    /// Returns a reference to the tensor's metadata.
    pub fn metadata(&self) -> &TensorMetadata {
        &self.metadata
    }
    
    /// Returns a mutable reference to the tensor's metadata.
    pub fn metadata_mut(&mut self) -> &mut TensorMetadata {
        &mut self.metadata
    }
    
    /// Creates a new view into a subset of this tensor.
    pub fn view(&self, start: &[usize], end: &[usize]) -> Result<Self, Error> {
        let view = TensorView::new(self, start, end)?;
        let new_shape = view.shape().clone();
        
        Ok(Self {
            shape: new_shape,
            storage: TensorStorage::View(view),
            metadata: self.metadata.clone(),
        })
    }
    
    /// Moves the tensor to the specified device.
    pub fn to_device(&self, device: Device) -> Result<Self, Error> {
        match (device, &self.storage) {
            // Already on the right device
            (Device::Cpu, TensorStorage::Cpu(_)) => Ok(self.clone()),
            (Device::Metal, TensorStorage::Metal(_)) => Ok(self.clone()),
            (Device::Mmap, TensorStorage::Mmap(_)) => Ok(self.clone()),
            
            // Move to CPU
            (Device::Cpu, _) => {
                let num_elements = self.num_elements();
                let mut cpu_storage = CpuStorage::new(num_elements)?;
                
                // Copy data to CPU
                match &self.storage {
                    TensorStorage::Cpu(_) => unreachable!(),
                    TensorStorage::Metal(metal_storage) => {
                        #[cfg(feature = "metal-backend")]
                        metal_storage.copy_to_cpu(cpu_storage.as_mut_slice())?;
                        
                        #[cfg(not(feature = "metal-backend"))]
                        return Err(Error::FeatureDisabled("metal-backend".to_string()));
                    }
                    TensorStorage::Mmap(mmap_storage) => {
                        mmap_storage.copy_to_slice(cpu_storage.as_mut_slice())?;
                    }
                    TensorStorage::View(view) => {
                        view.copy_to_slice(cpu_storage.as_mut_slice())?;
                    }
                }
                
                Ok(Self {
                    shape: self.shape.clone(),
                    storage: TensorStorage::Cpu(cpu_storage),
                    metadata: self.metadata.clone(),
                })
            }
            
            // Move to Metal
            (Device::Metal, _) => {
                #[cfg(feature = "metal-backend")]
                {
                    let num_elements = self.num_elements();
                    let metal_storage = metal::MetalStorage::new(num_elements)?;
                    
                    // Copy data to Metal
                    match &self.storage {
                        TensorStorage::Cpu(cpu_storage) => {
                            metal_storage.copy_from_cpu(cpu_storage.as_slice())?;
                        }
                        TensorStorage::Metal(_) => unreachable!(),
                        TensorStorage::Mmap(mmap_storage) => {
                            let mut buffer = vec![T::zero(); num_elements];
                            mmap_storage.copy_to_slice(&mut buffer)?;
                            metal_storage.copy_from_cpu(&buffer)?;
                        }
                        TensorStorage::View(view) => {
                            let mut buffer = vec![T::zero(); num_elements];
                            view.copy_to_slice(&mut buffer)?;
                            metal_storage.copy_from_cpu(&buffer)?;
                        }
                    }
                    
                    Ok(Self {
                        shape: self.shape.clone(),
                        storage: TensorStorage::Metal(metal_storage),
                        metadata: self.metadata.clone(),
                    })
                }
                
                #[cfg(not(feature = "metal-backend"))]
                Err(Error::FeatureDisabled("metal-backend".to_string()))
            }
            
            // Move to memory-mapped storage
            (Device::Mmap, _) => {
                Err(Error::Unsupported("Cannot directly move tensor to memory-mapped storage".to_string()))
            }
        }
    }
    
    /// Set the name of the tensor.
    pub fn with_name(mut self, name: impl Into<String>) -> Self {
        self.metadata.name = Some(name.into());
        self
    }
}

impl TensorOptions {
    /// Creates a new set of options with the specified name.
    pub fn with_name(name: impl Into<String>) -> Self {
        let mut options = Self::default();
        options.name = Some(name.into());
        options
    }
    
    /// Sets the device for the tensor.
    pub fn device(mut self, device: Device) -> Self {
        self.device = device;
        self
    }
    
    /// Sets the data type for the tensor.
    pub fn dtype(mut self, dtype: DataTypeEnum) -> Self {
        self.dtype = Some(dtype);
        self
    }
    
    /// Sets the memory layout for the tensor.
    pub fn layout(mut self, layout: MemoryLayout) -> Self {
        self.layout = layout;
        self
    }
    
    /// Sets the quantization options for the tensor.
    pub fn quantization(mut self, quantization: QuantizationInfo) -> Self {
        self.quantization = Some(quantization);
        self
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    
    #[test]
    fn test_create_zeros_tensor() {
        let shape = [2, 3, 4];
        let tensor = Tensor::<f32>::zeros(shape).unwrap();
        
        assert_eq!(tensor.shape().dims(), &[2, 3, 4]);
        assert_eq!(tensor.num_elements(), 24);
        
        if let TensorStorage::Cpu(storage) = tensor.storage() {
            let slice = storage.as_slice();
            assert_eq!(slice.len(), 24);
            assert!(slice.iter().all(|&x| x == 0.0));
        } else {
            panic!("Expected CPU storage");
        }
    }
    
    #[test]
    fn test_create_ones_tensor() {
        let shape = [2, 3];
        let tensor = Tensor::<f32>::ones(shape).unwrap();
        
        assert_eq!(tensor.shape().dims(), &[2, 3]);
        assert_eq!(tensor.num_elements(), 6);
        
        if let TensorStorage::Cpu(storage) = tensor.storage() {
            let slice = storage.as_slice();
            assert_eq!(slice.len(), 6);
            assert!(slice.iter().all(|&x| x == 1.0));
        } else {
            panic!("Expected CPU storage");
        }
    }
    
    #[test]
    fn test_create_tensor_from_slice() {
        let data = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0];
        let shape = [2, 3];
        let tensor = Tensor::<f32>::from_slice(&data, shape).unwrap();
        
        assert_eq!(tensor.shape().dims(), &[2, 3]);
        assert_eq!(tensor.num_elements(), 6);
        
        if let TensorStorage::Cpu(storage) = tensor.storage() {
            let slice = storage.as_slice();
            assert_eq!(slice, &data);
        } else {
            panic!("Expected CPU storage");
        }
    }
    
    #[test]
    fn test_tensor_metadata() {
        let tensor = Tensor::<f32>::zeros([2, 2]).unwrap().with_name("test_tensor");
        
        assert_eq!(tensor.metadata().name, Some("test_tensor".to_string()));
    }
}