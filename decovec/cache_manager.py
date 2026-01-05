"""
Cache Management Module

Unified management of various caches during experiments.
"""
import os
import pickle
import hashlib
from typing import Any, Optional


class CacheManager:
    """Cache Manager"""
    
    def __init__(self, cache_dir: str = "cache_icl", use_cache: bool = False):
        """
        Initialize the cache manager
        
        Args:
            cache_dir: Cache directory path
            use_cache: Whether to enable caching
        """
        # Anchor relative paths to project root directory
        project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        self.cache_dir = cache_dir if os.path.isabs(cache_dir) else os.path.join(project_root, cache_dir)
        self.use_cache = use_cache
        
        if self.use_cache:
            os.makedirs(self.cache_dir, exist_ok=True)
            print(f"✓ Cache mode enabled, cache directory: {self.cache_dir}")
    
    def get_cache_key(self, prefix: str, **kwargs) -> str:
        """
        Generate cache key
        
        Args:
            prefix: Cache key prefix
            **kwargs: Cache parameters
        
        Returns:
            Cache key string
        """
        param_str = "_".join([f"{k}={v}" for k, v in sorted(kwargs.items())])
        hash_obj = hashlib.md5(param_str.encode())
        return f"{prefix}_{hash_obj.hexdigest()[:8]}"
    
    def load(self, cache_key: str) -> Optional[Any]:
        """
        Load data from cache
        
        Args:
            cache_key: Cache key
        
        Returns:
            Cached data, or None if not exists
        """
        if not self.use_cache:
            return None
        
        cache_file = os.path.join(self.cache_dir, f"{cache_key}.pkl")
        if os.path.exists(cache_file):
            try:
                with open(cache_file, 'rb') as f:
                    data = pickle.load(f)
                print(f"  ✓ Loaded from cache: {cache_key}")
                return data
            except Exception as e:
                print(f"  ⚠️  Cache load failed: {e}")
                return None
        return None
    
    def save(self, cache_key: str, data: Any):
        """
        Save data to cache
        
        Args:
            cache_key: Cache key
            data: Data to cache
        """
        if not self.use_cache:
            return
        
        cache_file = os.path.join(self.cache_dir, f"{cache_key}.pkl")
        try:
            with open(cache_file, 'wb') as f:
                pickle.dump(data, f)
            print(f"  ✓ Saved to cache: {cache_key}")
        except Exception as e:
            print(f"  ⚠️  Cache save failed: {e}")
    
    def clear(self, cache_pattern: str = None):
        """
        Clear cache files
        
        Args:
            cache_pattern: Pattern matching for cache keys, e.g., "mu_bar", "delta_z_cache"
                         If None, clears all caches
        """
        if not os.path.exists(self.cache_dir):
            print("Cache directory does not exist")
            return
        
        cache_files = os.listdir(self.cache_dir)
        deleted_count = 0
        
        for cache_file in cache_files:
            if cache_file.endswith('.pkl'):
                # If pattern specified, only delete matching caches
                if cache_pattern is None or cache_file.startswith(cache_pattern):
                    file_path = os.path.join(self.cache_dir, cache_file)
                    try:
                        os.remove(file_path)
                        deleted_count += 1
                        print(f"  ✓ Deleted cache: {cache_file}")
                    except Exception as e:
                        print(f"  ⚠️  Delete failed: {cache_file} - {e}")
        
        if deleted_count == 0:
            print(f"No matching cache files found (pattern: {cache_pattern})")
        else:
            print(f"\n✓ Successfully deleted {deleted_count} cache files")
    
    def list(self):
        """List all cache files"""
        if not os.path.exists(self.cache_dir):
            print("Cache directory does not exist")
            return
        
        cache_files = [f for f in os.listdir(self.cache_dir) if f.endswith('.pkl')]
        
        if not cache_files:
            print("No cache files")
            return
        
        print(f"\nCache directory: {self.cache_dir}")
        print(f"Number of cache files: {len(cache_files)}\n")
        
        for cache_file in sorted(cache_files):
            file_path = os.path.join(self.cache_dir, cache_file)
            file_size = os.path.getsize(file_path)
            file_size_str = f"{file_size / 1024:.2f} KB" if file_size < 1024 * 1024 else f"{file_size / (1024 * 1024):.2f} MB"
            print(f"  - {cache_file} ({file_size_str})")

