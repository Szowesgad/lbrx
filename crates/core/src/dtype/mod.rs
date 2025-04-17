//! Data type definitions for tensors.
//!
//! This module provides trait implementations and types for handling
//! different numeric data types in tensors, including floating point,
//! integer, and quantized types.

use std::fmt;
use num_traits::{Zero, One, NumCast, AsPrimitive};
use half::{f16, bf16};

/// Enum of all supported data types for tensors.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum DataTypeEnum {
    /// 32-bit floating point
    F32,
    /// 16-bit floating point
    F16,
    /// 16-bit brain floating point
    BF16,
    /// 64-bit floating point
    F64,
    /// 8-bit signed integer
    I8,
    /// 16-bit signed integer
    I16,
    /// 32-bit signed integer
    I32,
    /// 64-bit signed integer
    I64,
    /// 8-bit unsigned integer
    U8,
    /// 16-bit unsigned integer
    U16,
    /// 32-bit unsigned integer
    U32,
    /// 64-bit unsigned integer
    U64,
    /// Boolean
    Bool,
    /// 4-bit quantized (requires special handling)
    Q4,
    /// 8-bit quantized (requires special handling)
    Q8,
}

impl DataTypeEnum {
    /// Returns the size in bytes of a single element of this data type.
    pub fn size_in_bytes(&self) -> usize {
        match self {
            DataTypeEnum::F32 => 4,
            DataTypeEnum::F16 => 2,
            DataTypeEnum::BF16 => 2,
            DataTypeEnum::F64 => 8,
            DataTypeEnum::I8 => 1,
            DataTypeEnum::I16 => 2,
            DataTypeEnum::I32 => 4,
            DataTypeEnum::I64 => 8,
            DataTypeEnum::U8 => 1,
            DataTypeEnum::U16 => 2,
            DataTypeEnum::U32 => 4,
            DataTypeEnum::U64 => 8,
            DataTypeEnum::Bool => 1,
            DataTypeEnum::Q4 => 1, // 8 elements per byte, but typically stored with metadata
            DataTypeEnum::Q8 => 1,
        }
    }
    
    /// Returns a string representation of this data type.
    pub fn as_str(&self) -> &'static str {
        match self {
            DataTypeEnum::F32 => "f32",
            DataTypeEnum::F16 => "f16",
            DataTypeEnum::BF16 => "bf16",
            DataTypeEnum::F64 => "f64",
            DataTypeEnum::I8 => "i8",
            DataTypeEnum::I16 => "i16",
            DataTypeEnum::I32 => "i32",
            DataTypeEnum::I64 => "i64",
            DataTypeEnum::U8 => "u8",
            DataTypeEnum::U16 => "u16",
            DataTypeEnum::U32 => "u32",
            DataTypeEnum::U64 => "u64",
            DataTypeEnum::Bool => "bool",
            DataTypeEnum::Q4 => "q4",
            DataTypeEnum::Q8 => "q8",
        }
    }
    
    /// Returns whether this data type is a floating point type.
    pub fn is_float(&self) -> bool {
        matches!(self, 
            DataTypeEnum::F32 | 
            DataTypeEnum::F16 | 
            DataTypeEnum::BF16 | 
            DataTypeEnum::F64
        )
    }
    
    /// Returns whether this data type is an integer type.
    pub fn is_int(&self) -> bool {
        matches!(self, 
            DataTypeEnum::I8 | 
            DataTypeEnum::I16 | 
            DataTypeEnum::I32 | 
            DataTypeEnum::I64 | 
            DataTypeEnum::U8 | 
            DataTypeEnum::U16 | 
            DataTypeEnum::U32 | 
            DataTypeEnum::U64
        )
    }
    
    /// Returns whether this data type is a quantized type.
    pub fn is_quantized(&self) -> bool {
        matches!(self, DataTypeEnum::Q4 | DataTypeEnum::Q8)
    }
}

impl fmt::Display for DataTypeEnum {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}", self.as_str())
    }
}

/// Trait for data types that can be used in tensors.
///
/// This trait provides common operations and characteristics for
/// numeric types used in tensor operations.
pub trait DataType: 
    Clone + Copy + fmt::Debug + Send + Sync + 'static +
    Zero + One + NumCast +
    AsPrimitive<f32> + AsPrimitive<f64> +
    AsPrimitive<i32> + AsPrimitive<i64>
{
    /// The enum variant representing this data type.
    const DTYPE: DataTypeEnum;
    
    /// The size of this data type in bytes.
    const SIZE: usize = Self::DTYPE.size_in_bytes();
    
    /// The minimum value for this data type.
    fn min_value() -> Self;
    
    /// The maximum value for this data type.
    fn max_value() -> Self;
    
    /// Converts a value from another data type to this type.
    fn from_dtype<S: DataType>(value: S) -> Self;
    
    /// Converts this value to another data type.
    fn to_dtype<D: DataType>(self) -> D;
    
    /// Returns the enum representation of this data type.
    fn dtype() -> DataTypeEnum {
        Self::DTYPE
    }
}

macro_rules! impl_data_type {
    ($type:ty, $dtype:expr, $min:expr, $max:expr) => {
        impl DataType for $type {
            const DTYPE: DataTypeEnum = $dtype;
            
            fn min_value() -> Self {
                $min
            }
            
            fn max_value() -> Self {
                $max
            }
            
            fn from_dtype<S: DataType>(value: S) -> Self {
                NumCast::from(value).unwrap_or_else(|| {
                    if value.as_() > Self::max_value().as_() {
                        Self::max_value()
                    } else if value.as_() < Self::min_value().as_() {
                        Self::min_value()
                    } else {
                        Self::zero()
                    }
                })
            }
            
            fn to_dtype<D: DataType>(self) -> D {
                D::from_dtype(self)
            }
        }
    };
}

// Implement DataType for standard types
impl_data_type!(f32, DataTypeEnum::F32, f32::MIN, f32::MAX);
impl_data_type!(f64, DataTypeEnum::F64, f64::MIN, f64::MAX);
impl_data_type!(i8, DataTypeEnum::I8, i8::MIN, i8::MAX);
impl_data_type!(i16, DataTypeEnum::I16, i16::MIN, i16::MAX);
impl_data_type!(i32, DataTypeEnum::I32, i32::MIN, i32::MAX);
impl_data_type!(i64, DataTypeEnum::I64, i64::MIN, i64::MAX);
impl_data_type!(u8, DataTypeEnum::U8, u8::MIN, u8::MAX);
impl_data_type!(u16, DataTypeEnum::U16, u16::MIN, u16::MAX);
impl_data_type!(u32, DataTypeEnum::U32, u32::MIN, u32::MAX);
impl_data_type!(u64, DataTypeEnum::U64, u64::MIN, u64::MAX);
impl_data_type!(bool, DataTypeEnum::Bool, false, true);

// Implement DataType for half-precision types
impl_data_type!(f16, DataTypeEnum::F16, f16::MIN, f16::MAX);
impl_data_type!(bf16, DataTypeEnum::BF16, bf16::MIN, bf16::MAX);

/// Module for quantized data types and operations
pub mod quantized;

/// Convert a type to another type with proper bounds checking.
pub fn convert<T: DataType, U: DataType>(value: T) -> U {
    value.to_dtype()
}

#[cfg(test)]
mod tests {
    use super::*;
    
    #[test]
    fn test_dtype_size() {
        assert_eq!(DataTypeEnum::F32.size_in_bytes(), 4);
        assert_eq!(DataTypeEnum::F16.size_in_bytes(), 2);
        assert_eq!(DataTypeEnum::I8.size_in_bytes(), 1);
        assert_eq!(DataTypeEnum::Q4.size_in_bytes(), 1);
    }
    
    #[test]
    fn test_dtype_classification() {
        assert!(DataTypeEnum::F32.is_float());
        assert!(DataTypeEnum::F16.is_float());
        assert!(!DataTypeEnum::I32.is_float());
        
        assert!(DataTypeEnum::I32.is_int());
        assert!(DataTypeEnum::U8.is_int());
        assert!(!DataTypeEnum::F32.is_int());
        
        assert!(DataTypeEnum::Q4.is_quantized());
        assert!(DataTypeEnum::Q8.is_quantized());
        assert!(!DataTypeEnum::F32.is_quantized());
    }
    
    #[test]
    fn test_dtype_conversion() {
        let f: f32 = 3.14;
        let i: i32 = convert(f);
        assert_eq!(i, 3);
        
        let large: i32 = 1000;
        let small: i8 = convert(large);
        assert_eq!(small, 127); // Clamped to i8::MAX
        
        let negative: i32 = -10;
        let unsigned: u8 = convert(negative);
        assert_eq!(unsigned, 0); // Clamped to u8::MIN
    }
}