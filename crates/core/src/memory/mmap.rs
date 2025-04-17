//! Memory-mapped file access for handling tensors larger than RAM.
//!
//! This module provides utilities for memory-mapped file access, allowing
//! tensors to be read directly from disk without loading the entire file
//! into memory.

use std::fs::{File, OpenOptions};
use std::path::{Path, PathBuf};
use std::sync::Arc;
use memmap2::{Mmap, MmapOptions, MmapMut};
use crate::error::{Error, Result};

/// Options for memory-mapped file access
#[derive(Debug, Clone)]
pub struct MemoryMapOptions {
    /// Whether to map the file as read-only
    pub read_only: bool,
    
    /// Whether to populate the page tables immediately (eager loading)
    pub prefault: bool,
    
    /// Whether to use huge pages if available
    pub huge_pages: bool,
    
    /// Cache window size for moving window operations (0 = no windowing)
    pub window_size: usize,
}

impl Default for MemoryMapOptions {
    fn default() -> Self {
        Self {
            read_only: true,
            prefault: false,
            huge_pages: false,
            window_size: 0,
        }
    }
}

impl MemoryMapOptions {
    /// Create a new set of memory map options
    pub fn new() -> Self {
        Self::default()
    }
    
    /// Set the read-only flag
    pub fn read_only(mut self, read_only: bool) -> Self {
        self.read_only = read_only;
        self
    }
    
    /// Set the prefault flag
    pub fn prefault(mut self, prefault: bool) -> Self {
        self.prefault = prefault;
        self
    }
    
    /// Set the huge pages flag
    pub fn huge_pages(mut self, huge_pages: bool) -> Self {
        self.huge_pages = huge_pages;
        self
    }
    
    /// Set the window size for moving window operations
    pub fn window_size(mut self, window_size: usize) -> Self {
        self.window_size = window_size;
        self
    }
}

/// A memory-mapped file or region
#[derive(Debug)]
pub struct MemoryMap {
    /// Path to the mapped file
    path: PathBuf,
    
    /// Memory mapping (read-only)
    map: Option<Mmap>,
    
    /// Mutable memory mapping (read-write)
    map_mut: Option<MmapMut>,
    
    /// Mapping options
    options: MemoryMapOptions,
    
    /// File handle (kept to ensure file remains open)
    _file: Option<File>,
    
    /// Current window offset for moving window operations
    window_offset: usize,
    
    /// File size
    file_size: u64,
}

impl MemoryMap {
    /// Open a memory-mapped file with the given options
    pub fn open<P: AsRef<Path>>(path: P, options: MemoryMapOptions) -> Result<Self> {
        let path = path.as_ref();
        
        if !path.exists() {
            return Err(Error::FileNotFound(path.to_path_buf()));
        }
        
        let file = if options.read_only {
            OpenOptions::new().read(true).open(path)?
        } else {
            OpenOptions::new().read(true).write(true).open(path)?
        };
        
        let file_size = file.metadata()?.len();
        
        let (map, map_mut) = if options.read_only {
            let mut map_options = MmapOptions::new();
            
            if options.prefault {
                map_options.populate();
            }
            
            // Use windowing if specified and file is large
            if options.window_size > 0 && file_size as usize > options.window_size {
                // Only map the first window
                map_options.len(options.window_size);
            }
            
            let map = unsafe { map_options.map(&file)? };
            (Some(map), None)
        } else {
            let mut map_options = MmapOptions::new();
            
            if options.prefault {
                map_options.populate();
            }
            
            // Use windowing if specified and file is large
            if options.window_size > 0 && file_size as usize > options.window_size {
                // Only map the first window
                map_options.len(options.window_size);
            }
            
            let map = unsafe { map_options.map_mut(&file)? };
            (None, Some(map))
        };
        
        Ok(Self {
            path: path.to_path_buf(),
            map,
            map_mut,
            options,
            _file: Some(file),
            window_offset: 0,
            file_size,
        })
    }
    
    /// Create a new memory-mapped file with the given size
    pub fn create<P: AsRef<Path>>(path: P, size: usize) -> Result<Self> {
        let path = path.as_ref();
        
        // Create the file
        let file = OpenOptions::new()
            .read(true)
            .write(true)
            .create(true)
            .open(path)?;
        
        // Set the file size
        file.set_len(size as u64)?;
        
        // Map the file
        let map = unsafe { MmapOptions::new().map_mut(&file)? };
        
        Ok(Self {
            path: path.to_path_buf(),
            map: None,
            map_mut: Some(map),
            options: MemoryMapOptions {
                read_only: false,
                ..Default::default()
            },
            _file: Some(file),
            window_offset: 0,
            file_size: size as u64,
        })
    }
    
    /// Get a slice of the memory-mapped region
    pub fn as_slice(&self) -> Result<&[u8]> {
        if let Some(map) = &self.map {
            Ok(&map[..])
        } else if let Some(map) = &self.map_mut {
            Ok(&map[..])
        } else {
            Err(Error::Internal("No mapping available".to_string()))
        }
    }
    
    /// Get a mutable slice of the memory-mapped region
    pub fn as_mut_slice(&mut self) -> Result<&mut [u8]> {
        if let Some(map) = &mut self.map_mut {
            Ok(&mut map[..])
        } else {
            Err(Error::Unsupported("Memory map is read-only".to_string()))
        }
    }
    
    /// Get a typed slice of the memory-mapped region
    pub fn as_slice_of<T: Copy>(&self) -> Result<&[T]> {
        let bytes = self.as_slice()?;
        
        let elem_size = std::mem::size_of::<T>();
        if bytes.len() % elem_size != 0 {
            return Err(Error::InvalidArgument(format!(
                "Byte length {} is not a multiple of element size {}", 
                bytes.len(), 
                elem_size
            )));
        }
        
        let num_elements = bytes.len() / elem_size;
        
        // Safety: We've verified that the byte length is a multiple of the element size,
        // and the alignment should be fine for memory-mapped regions
        unsafe {
            Ok(std::slice::from_raw_parts(
                bytes.as_ptr() as *const T,
                num_elements
            ))
        }
    }
    
    /// Get a typed mutable slice of the memory-mapped region
    pub fn as_mut_slice_of<T: Copy>(&mut self) -> Result<&mut [T]> {
        let bytes = self.as_mut_slice()?;
        
        let elem_size = std::mem::size_of::<T>();
        if bytes.len() % elem_size != 0 {
            return Err(Error::InvalidArgument(format!(
                "Byte length {} is not a multiple of element size {}", 
                bytes.len(), 
                elem_size
            )));
        }
        
        let num_elements = bytes.len() / elem_size;
        
        // Safety: We've verified that the byte length is a multiple of the element size,
        // and the alignment should be fine for memory-mapped regions
        unsafe {
            Ok(std::slice::from_raw_parts_mut(
                bytes.as_mut_ptr() as *mut T,
                num_elements
            ))
        }
    }
    
    /// Get the path to the mapped file
    pub fn path(&self) -> &Path {
        &self.path
    }
    
    /// Get the size of the mapped file
    pub fn file_size(&self) -> u64 {
        self.file_size
    }
    
    /// Get the current mapped region size
    pub fn mapped_size(&self) -> usize {
        if let Some(map) = &self.map {
            map.len()
        } else if let Some(map) = &self.map_mut {
            map.len()
        } else {
            0
        }
    }
    
    /// Check if the mapping is read-only
    pub fn is_read_only(&self) -> bool {
        self.options.read_only
    }
    
    /// Flush changes to disk (for mutable mappings)
    pub fn flush(&mut self) -> Result<()> {
        if let Some(map) = &mut self.map_mut {
            map.flush()?;
            Ok(())
        } else {
            Err(Error::Unsupported("Cannot flush read-only mapping".to_string()))
        }
    }
    
    /// Move the window to a new offset (for moving window operations)
    pub fn move_window(&mut self, offset: usize) -> Result<()> {
        if self.options.window_size == 0 {
            return Err(Error::Unsupported("Windowing is not enabled".to_string()));
        }
        
        if offset as u64 >= self.file_size {
            return Err(Error::IndexOutOfBounds { 
                index: offset, 
                axis: 0, 
                size: self.file_size as usize
            });
        }
        
        // Calculate window size (might be smaller at the end of the file)
        let window_size = std::cmp::min(
            self.options.window_size,
            (self.file_size - offset as u64) as usize
        );
        
        // Remap the file
        if self.options.read_only {
            if let Some(file) = &self._file {
                let mut map_options = MmapOptions::new();
                map_options.offset(offset as u64).len(window_size);
                
                if self.options.prefault {
                    map_options.populate();
                }
                
                self.map = Some(unsafe { map_options.map(file)? });
            }
        } else {
            if let Some(file) = &self._file {
                let mut map_options = MmapOptions::new();
                map_options.offset(offset as u64).len(window_size);
                
                if self.options.prefault {
                    map_options.populate();
                }
                
                self.map_mut = Some(unsafe { map_options.map_mut(file)? });
            }
        }
        
        self.window_offset = offset;
        
        Ok(())
    }
    
    /// Get the current window offset
    pub fn window_offset(&self) -> usize {
        self.window_offset
    }
    
    /// Copy data from the memory map to a slice
    pub fn copy_to_slice<T: Copy>(&self, dest: &mut [T], offset: usize) -> Result<()> {
        let src = self.as_slice_of::<T>()?;
        
        if offset >= src.len() {
            return Err(Error::IndexOutOfBounds { 
                index: offset, 
                axis: 0, 
                size: src.len()
            });
        }
        
        let copy_len = std::cmp::min(dest.len(), src.len() - offset);
        dest[..copy_len].copy_from_slice(&src[offset..offset + copy_len]);
        
        Ok(())
    }
    
    /// Copy data from a slice to the memory map
    pub fn copy_from_slice<T: Copy>(&mut self, src: &[T], offset: usize) -> Result<()> {
        let dest = self.as_mut_slice_of::<T>()?;
        
        if offset >= dest.len() {
            return Err(Error::IndexOutOfBounds { 
                index: offset, 
                axis: 0, 
                size: dest.len()
            });
        }
        
        let copy_len = std::cmp::min(src.len(), dest.len() - offset);
        dest[offset..offset + copy_len].copy_from_slice(&src[..copy_len]);
        
        Ok(())
    }
}

impl Drop for MemoryMap {
    fn drop(&mut self) {
        // Ensure any changes are flushed
        if !self.options.read_only {
            let _ = self.flush();
        }
    }
}

/// A windowed memory map that provides access to a large file through a moving window
pub struct WindowedMap {
    /// Underlying memory map
    map: MemoryMap,
    
    /// Window size
    window_size: usize,
    
    /// Cache for data outside the current window
    cache: Vec<u8>,
    
    /// Cache offset
    cache_offset: usize,
    
    /// Cache size
    cache_size: usize,
}

impl WindowedMap {
    /// Create a new windowed memory map
    pub fn new<P: AsRef<Path>>(path: P, window_size: usize) -> Result<Self> {
        let options = MemoryMapOptions {
            window_size,
            ..Default::default()
        };
        
        let map = MemoryMap::open(path, options)?;
        
        Ok(Self {
            map,
            window_size,
            cache: Vec::new(),
            cache_offset: 0,
            cache_size: 0,
        })
    }
    
    /// Read a slice of data from the map, automatically moving the window if needed
    pub fn read<T: Copy>(&mut self, offset: usize, dest: &mut [T]) -> Result<()> {
        let elem_size = std::mem::size_of::<T>();
        let byte_offset = offset * elem_size;
        let byte_len = dest.len() * elem_size;
        
        // Check if the requested region is within the current window
        let window_offset = self.map.window_offset();
        let window_end = window_offset + self.map.mapped_size();
        
        if byte_offset >= window_offset && byte_offset + byte_len <= window_end {
            // Fully within the current window, we can read directly
            let relative_offset = (byte_offset - window_offset) / elem_size;
            self.map.copy_to_slice(dest, relative_offset)?;
            return Ok(());
        }
        
        // Need to use window or cache
        if byte_len > self.window_size {
            // Region is larger than our window, need to read in chunks
            let mut remaining = dest.len();
            let mut read_offset = offset;
            let mut write_offset = 0;
            
            while remaining > 0 {
                // Move window to cover the current chunk
                self.map.move_window(read_offset * elem_size)?;
                
                // Calculate how much we can read from this window
                let window_elements = self.map.mapped_size() / elem_size;
                let chunk_size = std::cmp::min(remaining, window_elements);
                
                // Read the chunk
                self.map.copy_to_slice(&mut dest[write_offset..write_offset + chunk_size], 0)?;
                
                // Update offsets
                read_offset += chunk_size;
                write_offset += chunk_size;
                remaining -= chunk_size;
            }
        } else {
            // Move the window to cover the requested region
            self.map.move_window(byte_offset)?;
            
            // Read the data
            let rel_offset = 0; // We moved the window to exactly where we need
            self.map.copy_to_slice(dest, rel_offset)?;
        }
        
        Ok(())
    }
    
    /// Write a slice of data to the map, automatically moving the window if needed
    pub fn write<T: Copy>(&mut self, offset: usize, src: &[T]) -> Result<()> {
        let elem_size = std::mem::size_of::<T>();
        let byte_offset = offset * elem_size;
        let byte_len = src.len() * elem_size;
        
        // Check if the requested region is within the current window
        let window_offset = self.map.window_offset();
        let window_end = window_offset + self.map.mapped_size();
        
        if byte_offset >= window_offset && byte_offset + byte_len <= window_end {
            // Fully within the current window, we can write directly
            let relative_offset = (byte_offset - window_offset) / elem_size;
            self.map.copy_from_slice(src, relative_offset)?;
            return Ok(());
        }
        
        // Need to use window
        if byte_len > self.window_size {
            // Region is larger than our window, need to write in chunks
            let mut remaining = src.len();
            let mut read_offset = 0;
            let mut write_offset = offset;
            
            while remaining > 0 {
                // Move window to cover the current chunk
                self.map.move_window(write_offset * elem_size)?;
                
                // Calculate how much we can write to this window
                let window_elements = self.map.mapped_size() / elem_size;
                let chunk_size = std::cmp::min(remaining, window_elements);
                
                // Write the chunk
                self.map.copy_from_slice(&src[read_offset..read_offset + chunk_size], 0)?;
                
                // Update offsets
                read_offset += chunk_size;
                write_offset += chunk_size;
                remaining -= chunk_size;
            }
        } else {
            // Move the window to cover the requested region
            self.map.move_window(byte_offset)?;
            
            // Write the data
            let rel_offset = 0; // We moved the window to exactly where we need
            self.map.copy_from_slice(src, rel_offset)?;
        }
        
        Ok(())
    }
    
    /// Get the size of the underlying file
    pub fn file_size(&self) -> u64 {
        self.map.file_size()
    }
    
    /// Flush changes to disk
    pub fn flush(&mut self) -> Result<()> {
        self.map.flush()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::io::Write;
    use tempfile::tempdir;
    
    #[test]
    fn test_mmap_read() -> Result<()> {
        // Create a temporary file
        let dir = tempdir()?;
        let file_path = dir.path().join("test_mmap.bin");
        
        // Write some test data
        let data: Vec<u32> = (0..1000).collect();
        let mut file = File::create(&file_path)?;
        let bytes: &[u8] = bytemuck::cast_slice(&data);
        file.write_all(bytes)?;
        file.flush()?;
        
        // Map the file
        let mmap = MemoryMap::open(&file_path, Default::default())?;
        
        // Read as typed slice
        let mapped: &[u32] = mmap.as_slice_of()?;
        
        // Verify data
        assert_eq!(mapped.len(), 1000);
        for i in 0..1000 {
            assert_eq!(mapped[i], i as u32);
        }
        
        Ok(())
    }
    
    #[test]
    fn test_mmap_write() -> Result<()> {
        // Create a temporary file
        let dir = tempdir()?;
        let file_path = dir.path().join("test_mmap_write.bin");
        
        // Create a file of the right size
        let size = 1000 * std::mem::size_of::<u32>();
        let mmap = MemoryMap::create(&file_path, size)?;
        
        // Write data through the mapping
        let mut mmap = MemoryMap::open(&file_path, MemoryMapOptions::new().read_only(false))?;
        let data: Vec<u32> = (0..1000).collect();
        
        let slice = mmap.as_mut_slice_of::<u32>()?;
        slice.copy_from_slice(&data);
        
        // Flush to disk
        mmap.flush()?;
        
        // Re-open as read-only and verify
        let mmap = MemoryMap::open(&file_path, Default::default())?;
        let mapped: &[u32] = mmap.as_slice_of()?;
        
        assert_eq!(mapped.len(), 1000);
        for i in 0..1000 {
            assert_eq!(mapped[i], i as u32);
        }
        
        Ok(())
    }
    
    #[test]
    fn test_windowed_map() -> Result<()> {
        // Create a temporary file
        let dir = tempdir()?;
        let file_path = dir.path().join("test_windowed.bin");
        
        // Create a larger file (10MB)
        let size = 10 * 1024 * 1024;
        let mut mmap = MemoryMap::create(&file_path, size)?;
        
        // Fill with a pattern
        let slice = mmap.as_mut_slice()?;
        for i in 0..slice.len() {
            slice[i] = (i % 256) as u8;
        }
        mmap.flush()?;
        
        // Create a windowed map with a small window (1MB)
        let window_size = 1 * 1024 * 1024;
        let mut windowed = WindowedMap::new(&file_path, window_size)?;
        
        // Read from different parts of the file
        let mut buffer = vec![0u8; 1024];
        
        // Read near the start
        windowed.read(1000, &mut buffer)?;
        for i in 0..buffer.len() {
            assert_eq!(buffer[i], ((1000 + i) % 256) as u8);
        }
        
        // Read near the middle
        windowed.read(5 * 1024 * 1024, &mut buffer)?;
        for i in 0..buffer.len() {
            assert_eq!(buffer[i], ((5 * 1024 * 1024 + i) % 256) as u8);
        }
        
        // Read near the end
        windowed.read(9 * 1024 * 1024, &mut buffer)?;
        for i in 0..buffer.len() {
            assert_eq!(buffer[i], ((9 * 1024 * 1024 + i) % 256) as u8);
        }
        
        Ok(())
    }
}