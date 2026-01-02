"""
Real-time streaming processing pipeline.
"""
import asyncio
import queue
import threading
from typing import Callable, Optional, Any, Dict
from collections import deque
import time


class StreamProcessor:
    """
    Processes data streams in real-time.
    """
    def __init__(self, processor_fn: Callable, buffer_size: int = 1000):
        self.processor_fn = processor_fn
        self.buffer = deque(maxlen=buffer_size)
        self.running = False
        self.thread = None
    
    def start(self):
        """Start processing stream"""
        self.running = True
        self.thread = threading.Thread(target=self._process_loop)
        self.thread.start()
    
    def stop(self):
        """Stop processing stream"""
        self.running = False
        if self.thread:
            self.thread.join()
    
    def push(self, item: Any):
        """Push item to stream"""
        self.buffer.append(item)
    
    def _process_loop(self):
        """Processing loop"""
        while self.running:
            if self.buffer:
                item = self.buffer.popleft()
                try:
                    self.processor_fn(item)
                except Exception as e:
                    print(f"Error processing item: {e}")
            else:
                time.sleep(0.01)  # Small delay when buffer is empty


class AsyncQueue:
    """
    Async queue for non-blocking I/O.
    """
    def __init__(self, maxsize: int = 1000):
        self.queue = asyncio.Queue(maxsize=maxsize)
    
    async def put(self, item: Any):
        """Put item in queue"""
        await self.queue.put(item)
    
    async def get(self) -> Any:
        """Get item from queue"""
        return await self.queue.get()
    
    def size(self) -> int:
        """Get queue size"""
        return self.queue.qsize()


class TaskQueue:
    """
    Task queue for parallel processing.
    """
    def __init__(self, num_workers: int = 4):
        self.num_workers = num_workers
        self.queue = queue.Queue()
        self.workers = []
        self.running = False
    
    def start(self):
        """Start worker threads"""
        self.running = True
        for i in range(self.num_workers):
            worker = threading.Thread(target=self._worker_loop, args=(i,))
            worker.start()
            self.workers.append(worker)
    
    def stop(self):
        """Stop worker threads"""
        self.running = False
        # Add sentinel values to wake up workers
        for _ in self.workers:
            self.queue.put(None)
        
        for worker in self.workers:
            worker.join()
    
    def submit(self, task: Callable, *args, **kwargs):
        """Submit task to queue"""
        self.queue.put((task, args, kwargs))
    
    def _worker_loop(self, worker_id: int):
        """Worker processing loop"""
        while self.running:
            try:
                item = self.queue.get(timeout=1.0)
                if item is None:
                    break
                
                task, args, kwargs = item
                task(*args, **kwargs)
                self.queue.task_done()
            except queue.Empty:
                continue
            except Exception as e:
                print(f"Error in worker {worker_id}: {e}")


class ResourceManager:
    """
    Manages CPU/GPU/memory allocation.
    """
    def __init__(self):
        self.resource_usage = {}
        self.lock = threading.Lock()
    
    def allocate(self, resource_type: str, amount: float, task_id: str) -> bool:
        """Allocate resources"""
        with self.lock:
            current_usage = self.resource_usage.get(resource_type, 0.0)
            max_capacity = self._get_max_capacity(resource_type)
            
            if current_usage + amount <= max_capacity:
                self.resource_usage[resource_type] = current_usage + amount
                return True
            return False
    
    def deallocate(self, resource_type: str, amount: float, task_id: str):
        """Deallocate resources"""
        with self.lock:
            current_usage = self.resource_usage.get(resource_type, 0.0)
            self.resource_usage[resource_type] = max(0.0, current_usage - amount)
    
    def _get_max_capacity(self, resource_type: str) -> float:
        """Get maximum capacity for resource type"""
        # Simplified: return fixed capacities
        capacities = {
            "cpu": 100.0,  # Percentage
            "gpu": 100.0,  # Percentage
            "memory": 16.0  # GB
        }
        return capacities.get(resource_type, 100.0)
    
    def get_usage(self, resource_type: str) -> float:
        """Get current usage"""
        with self.lock:
            return self.resource_usage.get(resource_type, 0.0)

