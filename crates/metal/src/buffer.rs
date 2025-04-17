use metal;
use std::collections::HashMap;

/// Pool for reusing Metal buffers to reduce allocation overhead
pub struct BufferPool {
    device: metal::Device,
    pools: HashMap<usize, Vec<metal::Buffer>>,
}

impl BufferPool {
    /// Create a new buffer pool for the given device
    pub fn new(device: &metal::Device) -> Self {
        Self {
            device: device.clone(),
            pools: HashMap::new(),
        }
    }

    /// Get a buffer of at least `size` bytes, reusing from pool if possible
    pub fn get_buffer(&mut self, size: usize) -> metal::Buffer {
        // Align size to 256 bytes
        let aligned = (size + 255) & !255;
        if let Some(vec) = self.pools.get_mut(&aligned) {
            if let Some(buf) = vec.pop() {
                return buf;
            }
        }
        // Allocate new shared buffer
        self.device.new_buffer(
            aligned as u64,
            metal::MTLResourceOptions::StorageModeShared,
        )
    }

    /// Return a buffer to the pool for reuse
    pub fn return_buffer(&mut self, buffer: metal::Buffer) {
        let size = buffer.length() as usize;
        self.pools.entry(size).or_default().push(buffer);
    }
}