//! Parallel execution for tensor operations.
//!
//! This module provides utilities for parallel execution of tensor operations,
//! leveraging multiple CPU cores and scheduling tasks efficiently.

use crate::error::{Error, Result};
use once_cell::sync::OnceCell;
use rayon::ThreadPool;
use std::sync::atomic::{AtomicUsize, Ordering};

/// Global thread pool for CPU operations.
static THREAD_POOL: OnceCell<ThreadPool> = OnceCell::new();

/// Number of active tasks.
static ACTIVE_TASKS: AtomicUsize = AtomicUsize::new(0);

/// Maximum number of tasks that can be scheduled.
static MAX_TASKS: AtomicUsize = AtomicUsize::new(1024);

/// Initialize the parallel execution subsystem.
pub fn init() -> Result<()> {
    let num_threads = match std::thread::available_parallelism() {
        Ok(n) => n.get(),
        Err(_) => 4, // Fallback if we can't determine
    };
    
    let pool = rayon::ThreadPoolBuilder::new()
        .num_threads(num_threads)
        .build()
        .map_err(|e| Error::Internal(format!("Failed to build thread pool: {}", e)))?;
    
    THREAD_POOL.set(pool).map_err(|_| {
        Error::Internal("Failed to set thread pool".to_string())
    })?;
    
    log::debug!("Parallel subsystem initialized with {} threads", num_threads);
    
    Ok(())
}

/// Set the maximum number of concurrent tasks.
pub fn set_max_tasks(max: usize) {
    MAX_TASKS.store(max, Ordering::SeqCst);
}

/// Get the maximum number of concurrent tasks.
pub fn max_tasks() -> usize {
    MAX_TASKS.load(Ordering::SeqCst)
}

/// Get the number of active tasks.
pub fn active_tasks() -> usize {
    ACTIVE_TASKS.load(Ordering::SeqCst)
}

/// Get the number of threads in the pool.
pub fn num_threads() -> usize {
    THREAD_POOL.get()
        .map(|pool| pool.current_num_threads())
        .unwrap_or(1)
}

/// Spawn a task on the thread pool.
pub fn spawn<F, R>(f: F) -> Result<TaskHandle<R>>
where
    F: FnOnce() -> R + Send + 'static,
    R: Send + 'static,
{
    let pool = THREAD_POOL.get().ok_or_else(|| {
        Error::Internal("Thread pool not initialized".to_string())
    })?;
    
    // Check if we've reached the maximum number of tasks
    let active = ACTIVE_TASKS.fetch_add(1, Ordering::SeqCst);
    if active >= MAX_TASKS.load(Ordering::SeqCst) {
        ACTIVE_TASKS.fetch_sub(1, Ordering::SeqCst);
        return Err(Error::Internal(
            "Maximum number of concurrent tasks reached".to_string()
        ));
    }
    
    let (tx, rx) = std::sync::mpsc::channel();
    
    pool.spawn(move || {
        let result = f();
        let _ = tx.send(result);
        ACTIVE_TASKS.fetch_sub(1, Ordering::SeqCst);
    });
    
    Ok(TaskHandle { receiver: rx })
}

/// Spawn multiple tasks on the thread pool and wait for all of them to complete.
pub fn join<A, B, RA, RB>(a: A, b: B) -> (RA, RB)
where
    A: FnOnce() -> RA + Send,
    B: FnOnce() -> RB + Send,
    RA: Send,
    RB: Send,
{
    rayon::join(a, b)
}

/// Handle for a spawned task.
pub struct TaskHandle<R> {
    receiver: std::sync::mpsc::Receiver<R>,
}

impl<R> TaskHandle<R> {
    /// Wait for the task to complete and get the result.
    pub fn join(self) -> Result<R> {
        self.receiver.recv().map_err(|e| {
            Error::Internal(format!("Failed to receive task result: {}", e))
        })
    }
    
    /// Check if the task has completed.
    pub fn is_completed(&self) -> bool {
        self.receiver.try_recv().is_ok()
    }
}

/// Execute a closure in parallel for each element in a range.
pub fn parallel_for<F>(start: usize, end: usize, f: F) -> Result<()>
where
    F: Fn(usize) + Send + Sync,
{
    let pool = THREAD_POOL.get().ok_or_else(|| {
        Error::Internal("Thread pool not initialized".to_string())
    })?;
    
    pool.install(|| {
        (start..end).into_par_iter().for_each(f);
    });
    
    Ok(())
}

/// Execute a closure in parallel for each element in a range with a result.
pub fn parallel_map<F, R>(start: usize, end: usize, f: F) -> Result<Vec<R>>
where
    F: Fn(usize) -> R + Send + Sync,
    R: Send,
{
    let pool = THREAD_POOL.get().ok_or_else(|| {
        Error::Internal("Thread pool not initialized".to_string())
    })?;
    
    let result = pool.install(|| {
        (start..end).into_par_iter().map(f).collect()
    });
    
    Ok(result)
}

/// Extension trait for rayon's parallel iterators.
pub use rayon::prelude::{ParallelIterator, IntoParallelIterator, IntoParallelRefIterator, IntoParallelRefMutIterator};

#[cfg(test)]
mod tests {
    use super::*;
    
    #[test]
    fn test_thread_pool_initialization() {
        assert!(init().is_ok());
        assert!(THREAD_POOL.get().is_some());
        assert!(num_threads() > 0);
    }
    
    #[test]
    fn test_parallel_spawn() {
        init().unwrap();
        
        let handle = spawn(|| {
            let mut sum = 0;
            for i in 0..1000 {
                sum += i;
            }
            sum
        }).unwrap();
        
        let result = handle.join().unwrap();
        assert_eq!(result, (0..1000).sum());
    }
    
    #[test]
    fn test_parallel_join() {
        let (a, b) = join(
            || (0..100).sum::<usize>(),
            || (100..200).sum::<usize>()
        );
        
        assert_eq!(a, 4950);  // Sum of 0..100
        assert_eq!(b, 14950); // Sum of 100..200
    }
    
    #[test]
    fn test_parallel_for() {
        init().unwrap();
        
        let counter = std::sync::Arc::new(std::sync::atomic::AtomicUsize::new(0));
        let counter_clone = counter.clone();
        
        parallel_for(0, 1000, move |_| {
            counter_clone.fetch_add(1, Ordering::SeqCst);
        }).unwrap();
        
        assert_eq!(counter.load(Ordering::SeqCst), 1000);
    }
    
    #[test]
    fn test_parallel_map() {
        init().unwrap();
        
        let results = parallel_map(0, 10, |i| i * i).unwrap();
        
        assert_eq!(results, vec![0, 1, 4, 9, 16, 25, 36, 49, 64, 81]);
    }
}