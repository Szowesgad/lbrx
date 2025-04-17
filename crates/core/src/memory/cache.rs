//! Memory caching for frequently accessed tensor data.
//!
//! This module provides a cache for frequently accessed tensor data,
//! helping to reduce memory mapping and I/O overhead.

use std::collections::HashMap;
use std::sync::{Arc, Mutex};
use std::time::{Duration, Instant};
use parking_lot::RwLock;
use crate::error::{Error, Result};

/// Cache options for configuring cache behavior
#[derive(Debug, Clone)]
pub struct CacheOptions {
    /// Maximum size of the cache in bytes
    pub max_size: usize,
    
    /// Time-to-live for cache entries (0 = no expiry)
    pub ttl: Duration,
    
    /// Eviction policy
    pub policy: EvictionPolicy,
}

impl Default for CacheOptions {
    fn default() -> Self {
        Self {
            max_size: 1024 * 1024 * 1024, // 1GB default cache size
            ttl: Duration::from_secs(60 * 60), // 1 hour TTL
            policy: EvictionPolicy::LRU,
        }
    }
}

/// Eviction policy for cache entries
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum EvictionPolicy {
    /// Least Recently Used: evict the entry that hasn't been accessed for the longest time
    LRU,
    
    /// Least Frequently Used: evict the entry that has been accessed the least
    LFU,
    
    /// First In First Out: evict the oldest entry
    FIFO,
}

/// A cache entry
#[derive(Debug)]
struct CacheEntry {
    /// The cached data
    data: Arc<Vec<u8>>,
    
    /// When the entry was created
    created_at: Instant,
    
    /// When the entry was last accessed
    last_accessed: Instant,
    
    /// How many times the entry has been accessed
    access_count: usize,
    
    /// Size of the entry in bytes
    size: usize,
}

/// A memory cache for frequently accessed data
pub struct MemoryCache {
    /// Cached entries
    entries: RwLock<HashMap<String, CacheEntry>>,
    
    /// Cache options
    options: CacheOptions,
    
    /// Current size of the cache in bytes
    current_size: Mutex<usize>,
    
    /// Cache statistics
    stats: CacheStats,
}

/// Cache statistics
#[derive(Debug, Clone, Default)]
pub struct CacheStats {
    /// Number of cache hits
    pub hits: usize,
    
    /// Number of cache misses
    pub misses: usize,
    
    /// Number of entries evicted due to size constraints
    pub size_evictions: usize,
    
    /// Number of entries evicted due to TTL expiry
    pub ttl_evictions: usize,
    
    /// Total number of bytes stored in the cache over its lifetime
    pub total_bytes_stored: usize,
    
    /// Total number of bytes read from the cache over its lifetime
    pub total_bytes_read: usize,
}

impl MemoryCache {
    /// Create a new memory cache with the specified options
    pub fn new(options: CacheOptions) -> Self {
        Self {
            entries: RwLock::new(HashMap::new()),
            options,
            current_size: Mutex::new(0),
            stats: CacheStats::default(),
        }
    }
    
    /// Get a value from the cache
    pub fn get(&self, key: &str) -> Option<Arc<Vec<u8>>> {
        let mut entries = self.entries.write();
        
        if let Some(entry) = entries.get_mut(key) {
            // Check if entry has expired
            if self.options.ttl.as_secs() > 0 && entry.created_at.elapsed() > self.options.ttl {
                // Entry expired, remove it
                let size = entry.size;
                entries.remove(key);
                
                // Update size and stats
                let mut current_size = self.current_size.lock().unwrap();
                *current_size -= size;
                self.stats.ttl_evictions += 1;
                
                return None;
            }
            
            // Update access stats
            entry.last_accessed = Instant::now();
            entry.access_count += 1;
            
            // Update cache stats
            self.stats.hits += 1;
            self.stats.total_bytes_read += entry.size;
            
            Some(entry.data.clone())
        } else {
            // Cache miss
            self.stats.misses += 1;
            None
        }
    }
    
    /// Put a value in the cache
    pub fn put(&self, key: &str, value: Vec<u8>) -> Result<()> {
        let size = value.len();
        
        // Check if value exceeds max cache size
        if size > self.options.max_size {
            return Err(Error::InvalidArgument(format!(
                "Value size {} exceeds max cache size {}", 
                size, 
                self.options.max_size
            )));
        }
        
        // Make room for the new entry if needed
        self.ensure_space(size)?;
        
        // Create the entry
        let entry = CacheEntry {
            data: Arc::new(value),
            created_at: Instant::now(),
            last_accessed: Instant::now(),
            access_count: 0,
            size,
        };
        
        // Add to cache
        let mut entries = self.entries.write();
        
        // Remove old entry if it exists
        if let Some(old_entry) = entries.remove(key) {
            let mut current_size = self.current_size.lock().unwrap();
            *current_size -= old_entry.size;
        }
        
        // Add new entry
        entries.insert(key.to_string(), entry);
        
        // Update size
        let mut current_size = self.current_size.lock().unwrap();
        *current_size += size;
        
        // Update stats
        self.stats.total_bytes_stored += size;
        
        Ok(())
    }
    
    /// Remove a value from the cache
    pub fn remove(&self, key: &str) -> bool {
        let mut entries = self.entries.write();
        
        if let Some(entry) = entries.remove(key) {
            // Update size
            let mut current_size = self.current_size.lock().unwrap();
            *current_size -= entry.size;
            
            true
        } else {
            false
        }
    }
    
    /// Check if the cache contains a key
    pub fn contains(&self, key: &str) -> bool {
        let entries = self.entries.read();
        entries.contains_key(key)
    }
    
    /// Clear the cache
    pub fn clear(&self) {
        let mut entries = self.entries.write();
        entries.clear();
        
        let mut current_size = self.current_size.lock().unwrap();
        *current_size = 0;
    }
    
    /// Get the current size of the cache in bytes
    pub fn size(&self) -> usize {
        *self.current_size.lock().unwrap()
    }
    
    /// Get the number of entries in the cache
    pub fn len(&self) -> usize {
        let entries = self.entries.read();
        entries.len()
    }
    
    /// Check if the cache is empty
    pub fn is_empty(&self) -> bool {
        let entries = self.entries.read();
        entries.is_empty()
    }
    
    /// Get cache statistics
    pub fn stats(&self) -> CacheStats {
        self.stats.clone()
    }
    
    /// Ensure there's enough space for a new entry
    fn ensure_space(&self, size: usize) -> Result<()> {
        let current_size = self.size();
        
        // Check if we need to evict entries
        if current_size + size <= self.options.max_size {
            return Ok(());
        }
        
        // Calculate how much space we need to free
        let target = (current_size + size) - self.options.max_size;
        
        // Acquire write lock for entries
        let mut entries = self.entries.write();
        
        // Make a list of entries for eviction consideration
        let mut candidates: Vec<_> = entries.iter().collect();
        
        // Sort candidates according to eviction policy
        match self.options.policy {
            EvictionPolicy::LRU => {
                // Sort by last access time (oldest first)
                candidates.sort_by(|a, b| a.1.last_accessed.cmp(&b.1.last_accessed));
            }
            EvictionPolicy::LFU => {
                // Sort by access count (least first)
                candidates.sort_by(|a, b| a.1.access_count.cmp(&b.1.access_count));
            }
            EvictionPolicy::FIFO => {
                // Sort by creation time (oldest first)
                candidates.sort_by(|a, b| a.1.created_at.cmp(&b.1.created_at));
            }
        }
        
        // Evict entries until we have enough space
        let mut freed = 0;
        let mut evicted = 0;
        
        for (key, entry) in candidates {
            if freed >= target {
                break;
            }
            
            freed += entry.size;
            entries.remove(key);
            evicted += 1;
        }
        
        // Update size
        let mut current_size = self.current_size.lock().unwrap();
        *current_size -= freed;
        
        // Update stats
        self.stats.size_evictions += evicted;
        
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    
    #[test]
    fn test_cache_basic() {
        let options = CacheOptions {
            max_size: 1024 * 1024, // 1MB
            ttl: Duration::from_secs(60),
            policy: EvictionPolicy::LRU,
        };
        
        let cache = MemoryCache::new(options);
        
        // Put a value
        let key = "test_key";
        let value = vec![1, 2, 3, 4];
        cache.put(key, value).unwrap();
        
        // Check contains
        assert!(cache.contains(key));
        
        // Get the value
        let retrieved = cache.get(key).unwrap();
        assert_eq!(&*retrieved, &[1, 2, 3, 4]);
        
        // Remove the value
        cache.remove(key);
        assert!(!cache.contains(key));
    }
    
    #[test]
    fn test_cache_eviction() {
        // Create a small cache (100 bytes)
        let options = CacheOptions {
            max_size: 100,
            ttl: Duration::from_secs(60),
            policy: EvictionPolicy::LRU,
        };
        
        let cache = MemoryCache::new(options);
        
        // Add several entries
        for i in 0..5 {
            let key = format!("key_{}", i);
            let value = vec![i as u8; 20]; // Each entry is 20 bytes
            cache.put(&key, value).unwrap();
        }
        
        // Cache should have evicted oldest entries
        assert_eq!(cache.len(), 5); // All entries fit (5 * 20 = 100)
        
        // Add one more entry to trigger eviction
        let key = "key_5";
        let value = vec![5; 20];
        cache.put(key, value).unwrap();
        
        // Should have evicted at least one entry
        assert!(cache.len() < 6);
        assert!(cache.size() <= 100);
        
        // The most recently added entry should still be there
        assert!(cache.contains("key_5"));
    }
    
    #[test]
    fn test_cache_ttl() {
        // Create a cache with a very short TTL
        let options = CacheOptions {
            max_size: 1024 * 1024,
            ttl: Duration::from_millis(50), // 50ms TTL
            policy: EvictionPolicy::LRU,
        };
        
        let cache = MemoryCache::new(options);
        
        // Add an entry
        let key = "ttl_test";
        let value = vec![1, 2, 3, 4];
        cache.put(key, value).unwrap();
        
        // Should be able to retrieve immediately
        assert!(cache.get(key).is_some());
        
        // Wait for TTL to expire
        std::thread::sleep(Duration::from_millis(100));
        
        // Entry should be gone
        assert!(cache.get(key).is_none());
    }
    
    #[test]
    fn test_cache_stats() {
        let cache = MemoryCache::new(Default::default());
        
        // Initial stats
        let initial_stats = cache.stats();
        assert_eq!(initial_stats.hits, 0);
        assert_eq!(initial_stats.misses, 0);
        
        // Add an entry
        let key = "stats_test";
        let value = vec![1, 2, 3, 4];
        cache.put(key, value).unwrap();
        
        // Miss (non-existent key)
        cache.get("nonexistent");
        
        // Hit
        cache.get(key);
        cache.get(key);
        
        // Check stats
        let stats = cache.stats();
        assert_eq!(stats.hits, 2);
        assert_eq!(stats.misses, 1);
        assert_eq!(stats.total_bytes_stored, 4);
        assert_eq!(stats.total_bytes_read, 8); // 2 reads * 4 bytes
    }
}