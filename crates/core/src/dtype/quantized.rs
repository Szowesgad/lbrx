//! Quantized data types for reduced precision representation.
//!
//! This module provides implementations of quantized data types which
//! allow trading off precision for reduced memory usage and higher
//! computation speed.

use super::DataType;
use super::DataTypeEnum;
use num_traits::{Zero, One, NumCast, AsPrimitive};
use std::fmt;
use std::ops::{Add, Sub, Mul, Div};

/// 8-bit quantized type with zero point at 0 (symmetric quantization)
#[derive(Copy, Clone, Debug, PartialEq, Eq)]
#[repr(transparent)]
pub struct Q8_0(i8);

/// 4-bit quantized type (0-15 range mapped to floating point)
#[derive(Copy, Clone, Debug, PartialEq, Eq)]
#[repr(transparent)]
pub struct Q4_0(u8); // Represents two 4-bit values

/// Trait for quantized data types
pub trait QuantizedType: DataType {
    /// The underlying storage type (typically i8, u8, etc.)
    type Storage;
    
    /// The full precision type represented by this quantized type
    type FullPrecision: DataType;
    
    /// Number of bits used per element
    const BITS: u8;
    
    /// Quantize a full precision value to this quantized type
    fn quantize(value: Self::FullPrecision, scale: f32) -> Self;
    
    /// Dequantize this value to full precision
    fn dequantize(self, scale: f32) -> Self::FullPrecision;
    
    /// Get the raw storage value
    fn raw_value(self) -> Self::Storage;
}

impl Zero for Q8_0 {
    fn zero() -> Self {
        Q8_0(0)
    }
    
    fn is_zero(&self) -> bool {
        self.0 == 0
    }
}

impl One for Q8_0 {
    fn one() -> Self {
        Q8_0(1)
    }
}

impl NumCast for Q8_0 {
    fn from<T: num_traits::NumCast>(n: T) -> Option<Self> {
        i8::from(n).map(Q8_0)
    }
}

impl AsPrimitive<f32> for Q8_0 {
    fn as_(self) -> f32 {
        self.0 as f32
    }
}

impl AsPrimitive<f64> for Q8_0 {
    fn as_(self) -> f64 {
        self.0 as f64
    }
}

impl AsPrimitive<i32> for Q8_0 {
    fn as_(self) -> i32 {
        self.0 as i32
    }
}

impl AsPrimitive<i64> for Q8_0 {
    fn as_(self) -> i64 {
        self.0 as i64
    }
}

impl Add for Q8_0 {
    type Output = Self;
    
    fn add(self, rhs: Self) -> Self::Output {
        // Saturating add to prevent overflow
        Q8_0(self.0.saturating_add(rhs.0))
    }
}

impl Sub for Q8_0 {
    type Output = Self;
    
    fn sub(self, rhs: Self) -> Self::Output {
        // Saturating sub to prevent overflow
        Q8_0(self.0.saturating_sub(rhs.0))
    }
}

impl Mul for Q8_0 {
    type Output = Self;
    
    fn mul(self, rhs: Self) -> Self::Output {
        // Scale down to prevent overflow
        Q8_0((self.0 as i16 * rhs.0 as i16 / 127) as i8)
    }
}

impl Div for Q8_0 {
    type Output = Self;
    
    fn div(self, rhs: Self) -> Self::Output {
        if rhs.0 == 0 {
            return Q8_0(if self.0 >= 0 { i8::MAX } else { i8::MIN });
        }
        Q8_0((self.0 as i16 * 127 / rhs.0 as i16) as i8)
    }
}

impl fmt::Display for Q8_0 {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "Q8_0({})", self.0)
    }
}

impl DataType for Q8_0 {
    const DTYPE: DataTypeEnum = DataTypeEnum::Q8;
    
    fn min_value() -> Self {
        Q8_0(i8::MIN)
    }
    
    fn max_value() -> Self {
        Q8_0(i8::MAX)
    }
    
    fn from_dtype<S: DataType>(value: S) -> Self {
        let f = value.as_();
        let clamped = if f > 127.0 {
            127.0
        } else if f < -127.0 {
            -127.0
        } else {
            f
        };
        
        Q8_0(clamped as i8)
    }
    
    fn to_dtype<D: DataType>(self) -> D {
        D::from_dtype(self)
    }
}

impl QuantizedType for Q8_0 {
    type Storage = i8;
    type FullPrecision = f32;
    const BITS: u8 = 8;
    
    fn quantize(value: f32, scale: f32) -> Self {
        let scaled = value / scale;
        let clamped = if scaled > 127.0 {
            127
        } else if scaled < -127.0 {
            -127
        } else {
            scaled as i8
        };
        
        Q8_0(clamped)
    }
    
    fn dequantize(self, scale: f32) -> f32 {
        self.0 as f32 * scale
    }
    
    fn raw_value(self) -> i8 {
        self.0
    }
}

// Similar implementation for Q4_0 would go here, but with additional
// packing/unpacking logic since two values share a byte

/// Quantization parameters for a tensor block
#[derive(Clone, Debug)]
pub struct QuantizationParams {
    /// Scale factor applied during quantization
    pub scale: f32,
    
    /// Zero point (used in asymmetric quantization)
    pub zero_point: i32,
    
    /// Block size for block-wise quantization
    pub block_size: usize,
}

impl QuantizationParams {
    /// Create new quantization parameters with symmetric quantization (zero_point = 0)
    pub fn symmetric(scale: f32, block_size: usize) -> Self {
        Self {
            scale,
            zero_point: 0,
            block_size,
        }
    }
    
    /// Create new quantization parameters with asymmetric quantization
    pub fn asymmetric(scale: f32, zero_point: i32, block_size: usize) -> Self {
        Self {
            scale,
            zero_point,
            block_size,
        }
    }
}

/// Compute quantization parameters for a block of data using symmetric quantization
pub fn compute_symmetric_params(data: &[f32]) -> QuantizationParams {
    if data.is_empty() {
        return QuantizationParams::symmetric(1.0, 1);
    }
    
    // Find absolute maximum value
    let max_abs = data.iter()
        .map(|&x| x.abs())
        .fold(0.0f32, |a, b| a.max(b));
    
    // Use max value to determine scale, with small epsilon to prevent division by zero
    let scale = (max_abs + 1e-6) / 127.0;
    
    QuantizationParams::symmetric(scale, data.len())
}

/// Quantize a block of data to 8-bit symmetric quantization
pub fn quantize_block_q8_0(data: &[f32]) -> (Vec<Q8_0>, QuantizationParams) {
    let params = compute_symmetric_params(data);
    let quantized = data.iter()
        .map(|&x| Q8_0::quantize(x, params.scale))
        .collect();
    
    (quantized, params)
}

/// Dequantize a block of 8-bit symmetric quantized data
pub fn dequantize_block_q8_0(data: &[Q8_0], params: &QuantizationParams) -> Vec<f32> {
    data.iter()
        .map(|&x| x.dequantize(params.scale))
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;
    
    #[test]
    fn test_q8_0_quantization() {
        let values = [-10.0, -1.0, 0.0, 1.0, 10.0];
        let (quantized, params) = quantize_block_q8_0(&values);
        
        // Scale should be approximately 10/127 = 0.0787
        assert!((params.scale - 0.0787).abs() < 0.01);
        
        // Check expected quantized values
        assert_eq!(quantized[0].raw_value(), -127);
        assert_eq!(quantized[1].raw_value(), -13);
        assert_eq!(quantized[2].raw_value(), 0);
        assert_eq!(quantized[3].raw_value(), 13);
        assert_eq!(quantized[4].raw_value(), 127);
        
        // Check round-trip accuracy
        let dequantized = dequantize_block_q8_0(&quantized, &params);
        
        // Check approximate equality after round-trip, with tolerance
        for (i, (&original, &dequant)) in values.iter().zip(dequantized.iter()).enumerate() {
            // The error should be within one quantization step
            assert!((original - dequant).abs() < params.scale, 
                "Value at index {}: original={}, dequantized={}, diff={}", 
                i, original, dequant, (original - dequant).abs());
        }
    }
}