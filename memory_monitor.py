# =============================================================================
# MEMORY MONITORING AND RESOURCE CLEANUP SYSTEM
# =============================================================================

import gc
import os
import psutil
import logging
import asyncio
import tracemalloc
from typing import Dict, Any, Optional, List
from datetime import datetime, timezone
from contextlib import contextmanager
from functools import wraps
import threading

# Global memory tracking
_MEMORY_SNAPSHOTS = []
_RESOURCE_TRACKER = {
    "db_connections": 0,
    "open_files": 0,
    "async_tasks": 0,
    "playwright_browsers": 0,
    "http_sessions": 0
}

class MemoryMonitor:
    """
    Comprehensive memory and resource monitoring
    """
    
    def __init__(self):
        self.process = psutil.Process()
        self.start_memory = None
        self.snapshots = []
        self.tracemalloc_started = False
        
    def start_monitoring(self):
        """Start memory monitoring with tracemalloc"""
        if not self.tracemalloc_started:
            tracemalloc.start()
            self.tracemalloc_started = True
        
        self.start_memory = self.get_memory_info()
        logging.info(f"MEMORY MONITOR: Started - Initial memory: {self.start_memory['memory_mb']:.1f}MB")
    
    def get_memory_info(self) -> Dict[str, Any]:
        """Get current memory and resource usage"""
        try:
            # Memory info
            memory_info = self.process.memory_info()
            memory_percent = self.process.memory_percent()
            
            # CPU info
            cpu_percent = self.process.cpu_percent()
            
            # File descriptors
            try:
                num_fds = self.process.num_fds() if hasattr(self.process, 'num_fds') else len(self.process.open_files())
            except:
                num_fds = 0
            
            # Threads
            num_threads = self.process.num_threads()
            
            # System memory
            system_memory = psutil.virtual_memory()
            
            return {
                "timestamp": datetime.now(timezone.utc).isoformat(),
                "memory_mb": memory_info.rss / 1024 / 1024,
                "memory_percent": memory_percent,
                "cpu_percent": cpu_percent,
                "file_descriptors": num_fds,
                "threads": num_threads,
                "system_memory_percent": system_memory.percent,
                "system_available_mb": system_memory.available / 1024 / 1024,
            }
        except Exception as e:
            logging.error(f"Error getting memory info: {e}")
            return {"error": str(e)}
    
    def take_snapshot(self, label: str):
        """Take a memory snapshot with label"""
        info = self.get_memory_info()
        info["label"] = label
        info["resource_tracker"] = _RESOURCE_TRACKER.copy()
        
        self.snapshots.append(info)
        
        # Log memory change since start
        if self.start_memory:
            memory_change = info["memory_mb"] - self.start_memory["memory_mb"]
            logging.info(f"MEMORY SNAPSHOT [{label}]: {info['memory_mb']:.1f}MB ({memory_change:+.1f}MB since start)")
        
        # Log if memory usage is concerning
        if info["memory_mb"] > 1000:  # 1GB
            logging.warning(f"HIGH MEMORY USAGE [{label}]: {info['memory_mb']:.1f}MB")
        
        return info
    
    def get_tracemalloc_top(self, limit: int = 10) -> List[str]:
        """Get top memory allocations from tracemalloc"""
        if not self.tracemalloc_started:
            return ["Tracemalloc not started"]
        
        try:
            snapshot = tracemalloc.take_snapshot()
            top_stats = snapshot.statistics('lineno')
            
            return [
                f"{stat.traceback.format()[-1]} - {stat.size / 1024 / 1024:.1f}MB"
                for stat in top_stats[:limit]
            ]
        except Exception as e:
            return [f"Error getting tracemalloc: {e}"]
    
    def force_garbage_collection(self) -> Dict[str, int]:
        """Force garbage collection and return stats"""
        before_objects = len(gc.get_objects())
        
        # Force collection
        collected_0 = gc.collect(0)
        collected_1 = gc.collect(1) 
        collected_2 = gc.collect(2)
        
        after_objects = len(gc.get_objects())
        
        stats = {
            "objects_before": before_objects,
            "objects_after": after_objects,
            "objects_freed": before_objects - after_objects,
            "collected_gen0": collected_0,
            "collected_gen1": collected_1,
            "collected_gen2": collected_2,
        }
        
        logging.info(f"GARBAGE COLLECTION: Freed {stats['objects_freed']} objects")
        return stats
    
    def stop_monitoring(self):
        """Stop monitoring and return summary"""
        if self.tracemalloc_started:
            tracemalloc.stop()
            self.tracemalloc_started = False
        
        if self.start_memory and self.snapshots:
            final_memory = self.snapshots[-1]["memory_mb"]
            total_change = final_memory - self.start_memory["memory_mb"]
            
            logging.info(f"MEMORY MONITOR: Stopped - Final memory: {final_memory:.1f}MB ({total_change:+.1f}MB total change)")
            
            return {
                "start_memory_mb": self.start_memory["memory_mb"],
                "final_memory_mb": final_memory,
                "total_change_mb": total_change,
                "snapshots": self.snapshots
            }
        
        return None

# Global monitor instance
memory_monitor = MemoryMonitor()

# =============================================================================
# RESOURCE TRACKING DECORATORS AND CONTEXT MANAGERS
# =============================================================================

def track_resource(resource_type: str):
    """Decorator to track resource usage"""
    def decorator(func):
        @wraps(func)
        async def async_wrapper(*args, **kwargs):
            _RESOURCE_TRACKER[resource_type] = _RESOURCE_TRACKER.get(resource_type, 0) + 1
            try:
                result = await func(*args, **kwargs)
                return result
            finally:
                _RESOURCE_TRACKER[resource_type] = max(0, _RESOURCE_TRACKER.get(resource_type, 0) - 1)
        
        @wraps(func)
        def sync_wrapper(*args, **kwargs):
            _RESOURCE_TRACKER[resource_type] = _RESOURCE_TRACKER.get(resource_type, 0) + 1
            try:
                result = func(*args, **kwargs)
                return result
            finally:
                _RESOURCE_TRACKER[resource_type] = max(0, _RESOURCE_TRACKER.get(resource_type, 0) - 1)
        
        return async_wrapper if asyncio.iscoroutinefunction(func) else sync_wrapper
    return decorator

@contextmanager
def resource_cleanup_context(resource_name: str, cleanup_func=None):
    """Context manager for resource cleanup"""
    _RESOURCE_TRACKER[resource_name] = _RESOURCE_TRACKER.get(resource_name, 0) + 1
    logging.debug(f"RESOURCE ACQUIRED: {resource_name} (total: {_RESOURCE_TRACKER[resource_name]})")
    
    try:
        yield
    finally:
        _RESOURCE_TRACKER[resource_name] = max(0, _RESOURCE_TRACKER.get(resource_name, 0) - 1)
        logging.debug(f"RESOURCE RELEASED: {resource_name} (remaining: {_RESOURCE_TRACKER[resource_name]})")
        
        if cleanup_func:
            try:
                cleanup_func()
            except Exception as e:
                logging.error(f"Error in cleanup for {resource_name}: {e}")

# =============================================================================
# PHASE MONITORING DECORATOR
# =============================================================================

def monitor_phase(phase_name: str):
    """Decorator to monitor memory during processing phases"""
    def decorator(func):
        @wraps(func)
        async def async_wrapper(*args, **kwargs):
            # Take snapshot before phase
            memory_monitor.take_snapshot(f"{phase_name}_START")
            
            try:
                result = await func(*args, **kwargs)
                
                # Take snapshot after successful completion
                memory_monitor.take_snapshot(f"{phase_name}_SUCCESS")
                
                # Force garbage collection after phase
                gc_stats = memory_monitor.force_garbage_collection()
                memory_monitor.take_snapshot(f"{phase_name}_POST_GC")
                
                return result
                
            except Exception as e:
                # Take snapshot on error
                memory_monitor.take_snapshot(f"{phase_name}_ERROR")
                logging.error(f"Phase {phase_name} failed: {e}")
                raise
        
        @wraps(func)
        def sync_wrapper(*args, **kwargs):
            memory_monitor.take_snapshot(f"{phase_name}_START")
            
            try:
                result = func(*args, **kwargs)
                memory_monitor.take_snapshot(f"{phase_name}_SUCCESS")
                
                gc_stats = memory_monitor.force_garbage_collection()
                memory_monitor.take_snapshot(f"{phase_name}_POST_GC")
                
                return result
                
            except Exception as e:
                memory_monitor.take_snapshot(f"{phase_name}_ERROR")
                logging.error(f"Phase {phase_name} failed: {e}")
                raise
        
        return async_wrapper if asyncio.iscoroutinefunction(func) else sync_wrapper
    return decorator

# =============================================================================
# RESOURCE CLEANUP FUNCTIONS
# =============================================================================

async def cleanup_async_tasks():
    """Safe cleanup - avoid cancelling server/system tasks"""
    # DO NOT use asyncio.all_tasks() - it includes uvicorn/server tasks
    # which causes recursive cancellation errors
    
    logging.info("CLEANUP: Skipping async task cleanup to prevent recursion")
    
    # If you need to track specific tasks, maintain your own task list:
    # tracked_tasks = getattr(cleanup_async_tasks, '_my_tasks', [])
    # cancelled = 0
    # for task in tracked_tasks:
    #     if not task.done():
    #         task.cancel()
    #         cancelled += 1
    # return cancelled
    
    return 0

def cleanup_database_connections():
    """Clean up database connections - customize for your DB library"""
    # Example for SQLAlchemy
    # try:
    #     from your_db_module import engine
    #     engine.dispose()
    #     logging.info("Database connections cleaned up")
    # except Exception as e:
    #     logging.error(f"Error cleaning up database: {e}")
    pass

def cleanup_playwright_resources():
    """Clean up Playwright resources"""
    # Example cleanup - customize for your Playwright usage
    # try:
    #     # Close any open browsers/pages
    #     pass
    # except Exception as e:
    #     logging.error(f"Error cleaning up Playwright: {e}")
    pass

async def full_resource_cleanup():
    """Perform safe resource cleanup without dangerous task cancellation"""
    logging.info("Starting safe resource cleanup...")
    
    # SKIP the dangerous async task cleanup that causes recursion
    cancelled_tasks = 0  # Don't call cleanup_async_tasks()
    
    # Safe cleanup operations only
    try:
        cleanup_database_connections()
    except Exception as e:
        logging.error(f"Database cleanup error: {e}")
    
    try:
        cleanup_playwright_resources()
    except Exception as e:
        logging.error(f"Playwright cleanup error: {e}")
    
    # Force garbage collection (this is always safe)
    try:
        gc_stats = memory_monitor.force_garbage_collection()
    except Exception as e:
        logging.error(f"Garbage collection error: {e}")
        gc_stats = {"objects_freed": 0}
    
    logging.info(f"SAFE CLEANUP COMPLETE: freed {gc_stats.get('objects_freed', 0)} objects")
    logging.info(f"FINAL RESOURCE STATE: {_RESOURCE_TRACKER}")
    
    return {
        "cancelled_tasks": cancelled_tasks,
        "gc_stats": gc_stats,
        "final_resources": _RESOURCE_TRACKER.copy()
    }

# =============================================================================
# MEMORY MONITORING ENDPOINTS
# =============================================================================

"""
Add these endpoints to your FastAPI app:

@app.get("/admin/memory-status")
async def get_memory_status():
    info = memory_monitor.get_memory_info()
    info["resource_tracker"] = _RESOURCE_TRACKER.copy()
    info["gc_stats"] = {
        "objects": len(gc.get_objects()),
        "generations": gc.get_stats()
    }
    return info

@app.get("/admin/memory-snapshots")
async def get_memory_snapshots():
    return {
        "snapshots": memory_monitor.snapshots,
        "tracemalloc_top": memory_monitor.get_tracemalloc_top()
    }

@app.post("/admin/force-cleanup")
async def force_cleanup():
    cleanup_result = await full_resource_cleanup()
    return {
        "status": "success",
        "cleanup_result": cleanup_result
    }
"""
