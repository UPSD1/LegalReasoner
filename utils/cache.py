"""
Aggressive Caching System for Multi-Task Legal Reward System

This module implements a sophisticated caching system designed to minimize API costs
during training by aggressively caching evaluation results with intelligent
content-based hashing and cost-aware management.

Key Features:
- Content-based hashing: Same legal content = same cached result
- Compression for storage efficiency (up to 70% space savings)
- Jurisdiction-aware caching with task-type specificity
- LRU eviction with cost-based priorities
- Persistent storage across training sessions
- Cost tracking and optimization metrics
- Expected 60-80% API cost reduction during training

Caching Strategy:
- Cache keys based on: query + response + task_type + jurisdiction + judge_type
- Aggressive TTL: 1 week default (legal evaluations are stable)
- Compression: gzip compression for storage efficiency
- Persistence: SQLite database for cross-session caching
- Memory + Disk: Hot cache in memory, persistent cache on disk
"""

import hashlib
import gzip
import json
import time
import sqlite3
import threading
from typing import Dict, Any, Optional, List, Tuple, Union
from datetime import datetime, timedelta
from pathlib import Path
from dataclasses import dataclass, field
from collections import OrderedDict
import pickle

# Import core components
from ..core import (
    LegalRewardSystemError, CacheError, LegalTaskType, 
    USJurisdiction, CacheStrategy, create_error_context
)
from .logging import get_legal_logger


@dataclass
class CacheEntry:
    """
    Individual cache entry with metadata and cost tracking.
    
    Stores evaluation results with comprehensive metadata for
    cost tracking, performance analysis, and cache management.
    """
    key: str
    data: Dict[str, Any]
    timestamp: float
    ttl: int  # Time to live in seconds
    access_count: int = 0
    last_accessed: float = field(default_factory=time.time)
    estimated_cost_saved: float = 0.0
    compression_ratio: float = 1.0
    task_type: Optional[str] = None
    jurisdiction: Optional[str] = None
    judge_type: Optional[str] = None
    
    @property
    def is_expired(self) -> bool:
        """Check if cache entry has expired"""
        return (time.time() - self.timestamp) > self.ttl
    
    @property
    def age_seconds(self) -> float:
        """Get age of cache entry in seconds"""
        return time.time() - self.timestamp
    
    @property
    def age_hours(self) -> float:
        """Get age of cache entry in hours"""
        return self.age_seconds / 3600
    
    def touch(self):
        """Update last accessed time and increment access count"""
        self.last_accessed = time.time()
        self.access_count += 1
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert cache entry to dictionary for serialization"""
        return {
            'key': self.key,
            'data': self.data,
            'timestamp': self.timestamp,
            'ttl': self.ttl,
            'access_count': self.access_count,
            'last_accessed': self.last_accessed,
            'estimated_cost_saved': self.estimated_cost_saved,
            'compression_ratio': self.compression_ratio,
            'task_type': self.task_type,
            'jurisdiction': self.jurisdiction,
            'judge_type': self.judge_type
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'CacheEntry':
        """Create cache entry from dictionary"""
        return cls(**data)


class CacheKeyGenerator:
    """
    Generate consistent, collision-resistant cache keys for legal evaluations.
    
    Creates deterministic cache keys based on legal content and context,
    ensuring that identical legal evaluation requests hit the same cache entry.
    """
    
    def __init__(self):
        self.logger = get_legal_logger("cache_key_generator")
    
    def generate_cache_key(self, 
                          query: str,
                          response: str, 
                          task_type: str,
                          jurisdiction: str,
                          judge_type: str,
                          additional_context: Optional[Dict[str, Any]] = None) -> str:
        """
        Generate cache key for legal evaluation.
        
        Args:
            query: Legal query/question
            response: Model response to evaluate
            task_type: Legal task type (judicial_reasoning, etc.)
            jurisdiction: US jurisdiction context
            judge_type: Type of judge evaluation (helpfulness, etc.)
            additional_context: Additional context for cache key
            
        Returns:
            SHA-256 based cache key for consistent lookups
        """
        try:
            # Normalize inputs for consistent hashing
            normalized_query = self._normalize_text(query)
            normalized_response = self._normalize_text(response)
            normalized_task_type = task_type.lower().strip()
            normalized_jurisdiction = jurisdiction.lower().strip()
            normalized_judge_type = judge_type.lower().strip()
            
            # Create base content string
            content_parts = [
                f"query:{normalized_query}",
                f"response:{normalized_response}",
                f"task:{normalized_task_type}",
                f"jurisdiction:{normalized_jurisdiction}",
                f"judge:{normalized_judge_type}"
            ]
            
            # Add additional context if provided
            if additional_context:
                # Sort keys for consistent ordering
                sorted_context = sorted(additional_context.items())
                context_str = "|".join(f"{k}:{v}" for k, v in sorted_context)
                content_parts.append(f"context:{context_str}")
            
            # Join all parts
            content_string = "|".join(content_parts)
            
            # Generate SHA-256 hash
            cache_key = hashlib.sha256(content_string.encode('utf-8')).hexdigest()
            
            self.logger.debug(f"Generated cache key for {task_type}/{judge_type}: {cache_key[:16]}...")
            
            return cache_key
            
        except Exception as e:
            self.logger.error(f"Error generating cache key: {e}")
            # Return fallback key based on basic content
            fallback_content = f"{query[:100]}|{response[:100]}|{task_type}|{judge_type}"
            return hashlib.sha256(fallback_content.encode('utf-8')).hexdigest()
    
    def _normalize_text(self, text: str) -> str:
        """
        Normalize text for consistent cache key generation.
        
        Args:
            text: Text to normalize
            
        Returns:
            Normalized text with consistent formatting
        """
        if not text:
            return ""
        
        # Remove extra whitespace, normalize line endings
        normalized = " ".join(text.split())
        
        # Convert to lowercase for case-insensitive matching
        normalized = normalized.lower()
        
        # Remove common variations that don't affect evaluation
        # (but be careful not to remove legally significant content)
        replacements = [
            ('\n', ' '),
            ('\r', ' '),
            ('\t', ' ')
        ]
        
        for old, new in replacements:
            normalized = normalized.replace(old, new)
        
        return normalized.strip()
    
    def generate_prefix_key(self, task_type: str, jurisdiction: str) -> str:
        """
        Generate prefix key for cache management operations.
        
        Useful for cache invalidation and statistics by task type or jurisdiction.
        
        Args:
            task_type: Legal task type
            jurisdiction: US jurisdiction
            
        Returns:
            Prefix key for cache management
        """
        prefix_content = f"prefix:{task_type.lower()}:{jurisdiction.lower()}"
        return hashlib.sha256(prefix_content.encode('utf-8')).hexdigest()[:16]


class CacheCompressor:
    """
    Handle compression and decompression of cache data for storage efficiency.
    
    Provides significant storage savings (up to 70%) for large legal evaluation
    responses while maintaining fast compression/decompression times.
    """
    
    def __init__(self, compression_level: int = 6):
        self.compression_level = compression_level  # Balance between speed and compression
        self.logger = get_legal_logger("cache_compressor")
    
    def compress_data(self, data: Dict[str, Any]) -> Tuple[bytes, float]:
        """
        Compress cache data for storage.
        
        Args:
            data: Dictionary data to compress
            
        Returns:
            Tuple of (compressed_bytes, compression_ratio)
        """
        try:
            # Serialize to JSON first
            json_data = json.dumps(data, default=str, ensure_ascii=False)
            original_size = len(json_data.encode('utf-8'))
            
            # Compress using gzip
            compressed_data = gzip.compress(
                json_data.encode('utf-8'), 
                compresslevel=self.compression_level
            )
            compressed_size = len(compressed_data)
            
            # Calculate compression ratio
            compression_ratio = compressed_size / original_size if original_size > 0 else 1.0
            
            self.logger.debug(f"Compressed {original_size} bytes to {compressed_size} bytes "
                            f"(ratio: {compression_ratio:.2f})")
            
            return compressed_data, compression_ratio
            
        except Exception as e:
            self.logger.error(f"Error compressing cache data: {e}")
            # Fallback: return uncompressed data as bytes
            fallback_data = json.dumps(data, default=str).encode('utf-8')
            return fallback_data, 1.0
    
    def decompress_data(self, compressed_data: bytes) -> Dict[str, Any]:
        """
        Decompress cache data from storage.
        
        Args:
            compressed_data: Compressed data bytes
            
        Returns:
            Decompressed dictionary data
        """
        try:
            # Try gzip decompression first
            try:
                decompressed_data = gzip.decompress(compressed_data)
                json_str = decompressed_data.decode('utf-8')
            except (gzip.BadGzipFile, OSError):
                # Fallback: assume uncompressed data
                json_str = compressed_data.decode('utf-8')
            
            # Parse JSON
            data = json.loads(json_str)
            
            self.logger.debug(f"Decompressed {len(compressed_data)} bytes successfully")
            
            return data
            
        except Exception as e:
            self.logger.error(f"Error decompressing cache data: {e}")
            raise CacheError(
                f"Failed to decompress cache data: {e}",
                operation="decompress",
                error_context=create_error_context("cache", "decompress")
            )


class PersistentCacheStorage:
    """
    SQLite-based persistent cache storage for cross-session caching.
    
    Provides durable cache storage that survives training restarts,
    enabling cost savings across multiple training sessions.
    """
    
    def __init__(self, cache_dir: Path, max_db_size_mb: int = 1000):
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.db_path = self.cache_dir / "legal_cache.db"
        self.max_db_size_mb = max_db_size_mb
        self.lock = threading.Lock()
        self.logger = get_legal_logger("persistent_cache")
        
        # Initialize database
        self._initialize_database()
    
    def _initialize_database(self):
        """Initialize SQLite database with proper schema"""
        try:
            with sqlite3.connect(self.db_path) as conn:
                conn.execute("""
                    CREATE TABLE IF NOT EXISTS cache_entries (
                        key TEXT PRIMARY KEY,
                        data BLOB NOT NULL,
                        timestamp REAL NOT NULL,
                        ttl INTEGER NOT NULL,
                        access_count INTEGER DEFAULT 0,
                        last_accessed REAL DEFAULT 0,
                        estimated_cost_saved REAL DEFAULT 0,
                        compression_ratio REAL DEFAULT 1.0,
                        task_type TEXT,
                        jurisdiction TEXT,
                        judge_type TEXT,
                        created_at DATETIME DEFAULT CURRENT_TIMESTAMP
                    )
                """)
                
                # Create indexes for performance
                conn.execute("CREATE INDEX IF NOT EXISTS idx_timestamp ON cache_entries(timestamp)")
                conn.execute("CREATE INDEX IF NOT EXISTS idx_task_type ON cache_entries(task_type)")
                conn.execute("CREATE INDEX IF NOT EXISTS idx_jurisdiction ON cache_entries(jurisdiction)")
                conn.execute("CREATE INDEX IF NOT EXISTS idx_ttl ON cache_entries(timestamp, ttl)")
                
                conn.commit()
                
            self.logger.info(f"Initialized persistent cache database: {self.db_path}")
            
        except Exception as e:
            self.logger.error(f"Error initializing cache database: {e}")
            raise CacheError(
                f"Failed to initialize cache database: {e}",
                operation="initialize_db",
                error_context=create_error_context("persistent_cache", "initialize")
            )
    
    def store_entry(self, entry: CacheEntry, compressed_data: bytes):
        """Store cache entry in persistent storage"""
        try:
            with self.lock:
                with sqlite3.connect(self.db_path) as conn:
                    conn.execute("""
                        INSERT OR REPLACE INTO cache_entries (
                            key, data, timestamp, ttl, access_count, last_accessed,
                            estimated_cost_saved, compression_ratio, task_type, 
                            jurisdiction, judge_type
                        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                    """, (
                        entry.key, compressed_data, entry.timestamp, entry.ttl,
                        entry.access_count, entry.last_accessed, entry.estimated_cost_saved,
                        entry.compression_ratio, entry.task_type, entry.jurisdiction, entry.judge_type
                    ))
                    conn.commit()
                    
        except Exception as e:
            self.logger.error(f"Error storing cache entry {entry.key}: {e}")
            raise CacheError(
                f"Failed to store cache entry: {e}",
                cache_key=entry.key,
                operation="store_entry",
                error_context=create_error_context("persistent_cache", "store")
            )
    
    def retrieve_entry(self, cache_key: str) -> Optional[Tuple[CacheEntry, bytes]]:
        """Retrieve cache entry from persistent storage"""
        try:
            with self.lock:
                with sqlite3.connect(self.db_path) as conn:
                    cursor = conn.execute("""
                        SELECT key, data, timestamp, ttl, access_count, last_accessed,
                               estimated_cost_saved, compression_ratio, task_type,
                               jurisdiction, judge_type
                        FROM cache_entries WHERE key = ?
                    """, (cache_key,))
                    
                    row = cursor.fetchone()
                    if row is None:
                        return None
                    
                    # Create cache entry from database row
                    entry = CacheEntry(
                        key=row[0],
                        data={},  # Will be populated after decompression
                        timestamp=row[2],
                        ttl=row[3],
                        access_count=row[4],
                        last_accessed=row[5],
                        estimated_cost_saved=row[6],
                        compression_ratio=row[7],
                        task_type=row[8],
                        jurisdiction=row[9],
                        judge_type=row[10]
                    )
                    
                    compressed_data = row[1]
                    
                    return entry, compressed_data
                    
        except Exception as e:
            self.logger.error(f"Error retrieving cache entry {cache_key}: {e}")
            return None
    
    def delete_entry(self, cache_key: str):
        """Delete cache entry from persistent storage"""
        try:
            with self.lock:
                with sqlite3.connect(self.db_path) as conn:
                    conn.execute("DELETE FROM cache_entries WHERE key = ?", (cache_key,))
                    conn.commit()
                    
        except Exception as e:
            self.logger.error(f"Error deleting cache entry {cache_key}: {e}")
    
    def cleanup_expired_entries(self) -> int:
        """Clean up expired cache entries and return count of deleted entries"""
        try:
            current_time = time.time()
            with self.lock:
                with sqlite3.connect(self.db_path) as conn:
                    cursor = conn.execute("""
                        DELETE FROM cache_entries 
                        WHERE (timestamp + ttl) < ?
                    """, (current_time,))
                    deleted_count = cursor.rowcount
                    conn.commit()
                    
            self.logger.info(f"Cleaned up {deleted_count} expired cache entries")
            return deleted_count
            
        except Exception as e:
            self.logger.error(f"Error cleaning up expired entries: {e}")
            return 0
    
    def get_storage_stats(self) -> Dict[str, Any]:
        """Get storage statistics"""
        try:
            with self.lock:
                with sqlite3.connect(self.db_path) as conn:
                    # Total entries
                    cursor = conn.execute("SELECT COUNT(*) FROM cache_entries")
                    total_entries = cursor.fetchone()[0]
                    
                    # Total cost saved
                    cursor = conn.execute("SELECT SUM(estimated_cost_saved) FROM cache_entries")
                    total_cost_saved = cursor.fetchone()[0] or 0.0
                    
                    # Average compression ratio
                    cursor = conn.execute("SELECT AVG(compression_ratio) FROM cache_entries")
                    avg_compression = cursor.fetchone()[0] or 1.0
                    
                    # Database file size
                    db_size_mb = self.db_path.stat().st_size / (1024 * 1024) if self.db_path.exists() else 0
                    
                    # Task type breakdown
                    cursor = conn.execute("""
                        SELECT task_type, COUNT(*) 
                        FROM cache_entries 
                        GROUP BY task_type
                    """)
                    task_breakdown = dict(cursor.fetchall())
            
            return {
                'total_entries': total_entries,
                'total_cost_saved': total_cost_saved,
                'avg_compression_ratio': avg_compression,
                'db_size_mb': db_size_mb,
                'task_type_breakdown': task_breakdown,
                'storage_utilization': db_size_mb / self.max_db_size_mb
            }
            
        except Exception as e:
            self.logger.error(f"Error getting storage stats: {e}")
            return {'error': str(e)}


class MultiStrategyLegalRewardCache:
    """
    Multi-strategy caching system designed to minimize API costs during training.
    
    Supports 4 caching strategies:
    - AGGRESSIVE: Cache everything, long TTL, compression enabled
    - BALANCED: Cache selectively, medium TTL, moderate compression
    - CONSERVATIVE: Cache minimally, short TTL, no compression
    - DISABLED: No caching (pass-through mode)
    
    Features:
    - Content-based hashing (same legal content = same cached result)
    - Configurable compression for storage efficiency  
    - Jurisdiction-aware caching
    - Task-type specific cache keys
    - LRU eviction with cost-based priorities
    - Persistence across training sessions
    - Expected 0-80% cost reduction based on strategy
    """
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        
        # Determine caching strategy
        self.cache_strategy = CacheStrategy(config.get("cache_strategy", "aggressive"))
        
        # Apply strategy-specific configuration
        self._apply_strategy_config()
        
        self.cache_dir = Path(config.get("cache_dir", "/tmp/legal_reward_cache"))
        self.max_cache_size_gb = config.get("max_cache_size_gb", self.strategy_config["max_cache_size_gb"])
        self.cache_ttl_hours = config.get("cache_ttl_hours", self.strategy_config["cache_ttl_hours"])
        self.compression_enabled = config.get("compression", self.strategy_config["compression"])
        self.persist_across_sessions = config.get("persist_across_sessions", self.strategy_config["persist_across_sessions"])
        
        # Initialize components based on strategy
        if self.cache_strategy == CacheStrategy.DISABLED:
            # No caching components needed
            self.key_generator = None
            self.compressor = None
            self.persistent_storage = None
            self.memory_cache = None
        else:
            # Initialize normal caching components
            self.key_generator = CacheKeyGenerator()
            self.compressor = CacheCompressor() if self.compression_enabled else None
            self.persistent_storage = PersistentCacheStorage(
                self.cache_dir, 
                int(self.max_cache_size_gb * 1024)  # Convert GB to MB
            ) if self.persist_across_sessions else None
            
            # Memory cache (hot cache for frequently accessed items)
            self.memory_cache: OrderedDict[str, CacheEntry] = OrderedDict()
        
        self.max_memory_entries = config.get("max_memory_entries", self.strategy_config["max_memory_entries"])
        
        # Cost tracking
        self.cache_hit_savings = 0.0
        self.total_cache_hits = 0
        self.total_requests = 0
        self.cache_misses = 0
        
        # Thread safety
        self.lock = threading.Lock()
        
        # Logger
        self.logger = get_legal_logger("multi_strategy_cache")
        
        self.logger.info(f"Initialized {self.cache_strategy.value} cache: {self.cache_dir}, "
                        f"TTL: {self.cache_ttl_hours}h, Compression: {self.compression_enabled}")
    
    def _apply_strategy_config(self):
        """Apply configuration based on selected caching strategy"""
        if self.cache_strategy == CacheStrategy.AGGRESSIVE:
            self.strategy_config = {
                "cache_ttl_hours": 168,  # 1 week
                "max_cache_size_gb": 10,
                "compression": True,
                "persist_across_sessions": True,
                "max_memory_entries": 1000,
                "cache_threshold": 0.0,  # Cache everything
                "expected_hit_rate": 0.8
            }
        elif self.cache_strategy == CacheStrategy.BALANCED:
            self.strategy_config = {
                "cache_ttl_hours": 72,   # 3 days
                "max_cache_size_gb": 5,
                "compression": True,
                "persist_across_sessions": True,
                "max_memory_entries": 500,
                "cache_threshold": 0.02,  # Only cache if cost > $0.02
                "expected_hit_rate": 0.6
            }
        elif self.cache_strategy == CacheStrategy.CONSERVATIVE:
            self.strategy_config = {
                "cache_ttl_hours": 24,   # 1 day
                "max_cache_size_gb": 2,
                "compression": False,
                "persist_across_sessions": False,
                "max_memory_entries": 200,
                "cache_threshold": 0.05,  # Only cache if cost > $0.05
                "expected_hit_rate": 0.3
            }
        else:  # DISABLED
            self.strategy_config = {
                "cache_ttl_hours": 0,
                "max_cache_size_gb": 0,
                "compression": False,
                "persist_across_sessions": False,
                "max_memory_entries": 0,
                "cache_threshold": float('inf'),  # Never cache
                "expected_hit_rate": 0.0
            }
    
    def get_cache_key(self, 
                     query: str,
                     response: str, 
                     task_type: str,
                     jurisdiction: str,
                     judge_type: str,
                     additional_context: Optional[Dict[str, Any]] = None) -> str:
        """
        Generate cache key for legal evaluation.
        
        Args:
            query: Legal query/question
            response: Model response to evaluate  
            task_type: Legal task type
            jurisdiction: US jurisdiction context
            judge_type: Type of judge evaluation
            additional_context: Additional context
            
        Returns:
            Cache key for this evaluation
        """
        return self.key_generator.generate_cache_key(
            query, response, task_type, jurisdiction, judge_type, additional_context
        )
    
    def cache_response(self, 
                      cache_key: str,
                      response_data: Dict[str, Any], 
                      estimated_cost: float,
                      task_type: str,
                      jurisdiction: str,
                      judge_type: str):
        """
        Cache API response with strategy-aware cost tracking.
        
        Args:
            cache_key: Cache key for this response
            response_data: API response data to cache
            estimated_cost: Estimated cost of this API call
            task_type: Legal task type  
            jurisdiction: US jurisdiction
            judge_type: Judge type
        """
        # Check if caching is disabled
        if self.cache_strategy == CacheStrategy.DISABLED:
            self.logger.debug(f"Caching disabled - skipping cache for {task_type}/{judge_type}")
            return
        
        # Check cost threshold for strategy
        cache_threshold = self.strategy_config["cache_threshold"]
        if estimated_cost < cache_threshold:
            self.logger.debug(f"Cost ${estimated_cost:.4f} below threshold ${cache_threshold:.4f} "
                            f"for {self.cache_strategy.value} strategy - skipping cache")
            return
        
        try:
            with self.lock:
                # Create cache entry
                entry = CacheEntry(
                    key=cache_key,
                    data=response_data,
                    timestamp=time.time(),
                    ttl=self.cache_ttl_hours * 3600,  # Convert hours to seconds
                    estimated_cost_saved=estimated_cost,
                    task_type=task_type,
                    jurisdiction=jurisdiction,
                    judge_type=judge_type
                )
                
                # Store in memory cache (with LRU eviction)
                self._store_in_memory_cache(entry)
                
                # Store in persistent cache if enabled
                if self.persistent_storage and self.compression_enabled:
                    compressed_data, compression_ratio = self.compressor.compress_data(response_data)
                    entry.compression_ratio = compression_ratio
                    self.persistent_storage.store_entry(entry, compressed_data)
                elif self.persistent_storage:
                    # Store without compression
                    data_bytes = json.dumps(response_data, default=str).encode('utf-8')
                    self.persistent_storage.store_entry(entry, data_bytes)
                
                self.logger.debug(f"Cached response ({self.cache_strategy.value}) for {task_type}/{judge_type}: "
                                f"cost_saved=${estimated_cost:.4f}")
                
        except Exception as e:
            self.logger.error(f"Error caching response: {e}")
            raise CacheError(
                f"Failed to cache response: {e}",
                cache_key=cache_key,
                operation="cache_response",
                error_context=create_error_context("cache", "store", cost_impact=estimated_cost)
            )
    
    def get_cached_response(self, cache_key: str) -> Optional[Dict[str, Any]]:
        """
        Retrieve cached response and track savings.
        
        Args:
            cache_key: Cache key to lookup
            
        Returns:
            Cached response data if found, None otherwise
        """
        # Check if caching is disabled
        if self.cache_strategy == CacheStrategy.DISABLED:
            self.total_requests += 1
            self.cache_misses += 1
            return None
        
        try:
            with self.lock:
                self.total_requests += 1
                
                # Check memory cache first (fastest)
                entry = self._get_from_memory_cache(cache_key)
                
                # Check persistent cache if not in memory
                if entry is None and self.persistent_storage:
                    entry = self._get_from_persistent_cache(cache_key)
                    if entry:
                        # Promote to memory cache
                        self._store_in_memory_cache(entry)
                
                # Check if entry exists and is not expired
                if entry and not entry.is_expired:
                    # Update access tracking
                    entry.touch()
                    
                    # Track cache hit and savings
                    self.total_cache_hits += 1
                    self.cache_hit_savings += entry.estimated_cost_saved
                    
                    self.logger.debug(f"Cache hit ({self.cache_strategy.value}): {cache_key[:16]}... "
                                    f"(saved ${entry.estimated_cost_saved:.4f})")
                    
                    return entry.data
                
                elif entry and entry.is_expired:
                    # Remove expired entry
                    self._remove_expired_entry(cache_key, entry)
                
                # Cache miss
                self.cache_misses += 1
                self.logger.debug(f"Cache miss ({self.cache_strategy.value}): {cache_key[:16]}...")
                return None
                
        except Exception as e:
            self.logger.error(f"Error retrieving cached response: {e}")
            self.cache_misses += 1
            return None
    
    def _store_in_memory_cache(self, entry: CacheEntry):
        """Store entry in memory cache with LRU eviction"""
        if self.cache_strategy == CacheStrategy.DISABLED or self.memory_cache is None:
            return
        
        # Remove existing entry if present (for reordering)
        if entry.key in self.memory_cache:
            del self.memory_cache[entry.key]
        
        # Add entry (moves to end in OrderedDict)
        self.memory_cache[entry.key] = entry
        
        # Evict oldest entries if cache is full
        while len(self.memory_cache) > self.max_memory_entries:
            oldest_key, oldest_entry = self.memory_cache.popitem(last=False)
            self.logger.debug(f"Evicted from memory cache: {oldest_key[:16]}...")
    
    def _get_from_memory_cache(self, cache_key: str) -> Optional[CacheEntry]:
        """Get entry from memory cache"""
        if self.cache_strategy == CacheStrategy.DISABLED or self.memory_cache is None:
            return None
            
        if cache_key in self.memory_cache:
            # Move to end (mark as recently used)
            entry = self.memory_cache.pop(cache_key)
            self.memory_cache[cache_key] = entry
            return entry
        return None
    
    def _get_from_persistent_cache(self, cache_key: str) -> Optional[CacheEntry]:
        """Get entry from persistent cache"""
        if not self.persistent_storage:
            return None
        
        try:
            result = self.persistent_storage.retrieve_entry(cache_key)
            if result is None:
                return None
            
            entry, compressed_data = result
            
            # Decompress data
            if self.compression_enabled and self.compressor:
                entry.data = self.compressor.decompress_data(compressed_data)
            else:
                entry.data = json.loads(compressed_data.decode('utf-8'))
            
            return entry
            
        except Exception as e:
            self.logger.error(f"Error loading from persistent cache: {e}")
            return None
    
    def _remove_expired_entry(self, cache_key: str, entry: CacheEntry):
        """Remove expired entry from both memory and persistent cache"""
        # Remove from memory cache
        if cache_key in self.memory_cache:
            del self.memory_cache[cache_key]
        
        # Remove from persistent cache
        if self.persistent_storage:
            self.persistent_storage.delete_entry(cache_key)
        
        self.logger.debug(f"Removed expired entry: {cache_key[:16]}... "
                         f"(age: {entry.age_hours:.1f}h)")
    
    def cleanup_expired_cache(self):
        """Remove expired cache entries"""
        if self.cache_strategy == CacheStrategy.DISABLED:
            return
            
        try:
            with self.lock:
                # Clean memory cache
                expired_keys = []
                if self.memory_cache:
                    expired_keys = [
                        key for key, entry in self.memory_cache.items() 
                        if entry.is_expired
                    ]
                    
                    for key in expired_keys:
                        del self.memory_cache[key]
                
                # Clean persistent cache
                deleted_count = 0
                if self.persistent_storage:
                    deleted_count = self.persistent_storage.cleanup_expired_entries()
                    
                self.logger.info(f"Cache cleanup ({self.cache_strategy.value}): "
                               f"removed {len(expired_keys)} memory entries, "
                               f"{deleted_count} persistent entries")
                
        except Exception as e:
            self.logger.error(f"Error during cache cleanup: {e}")
    
    def get_cache_stats(self) -> Dict[str, Any]:
        """Get comprehensive cache performance and cost savings statistics"""
        try:
            with self.lock:
                # Calculate hit rate
                hit_rate = self.total_cache_hits / max(self.total_requests, 1)
                
                # Memory cache stats
                memory_stats = {
                    'entries': len(self.memory_cache) if self.memory_cache else 0,
                    'max_entries': self.max_memory_entries,
                    'utilization': (len(self.memory_cache) if self.memory_cache else 0) / max(self.max_memory_entries, 1)
                }
                
                # Persistent cache stats
                persistent_stats = {}
                if self.persistent_storage:
                    persistent_stats = self.persistent_storage.get_storage_stats()
                
                # Cost analysis
                projected_monthly_savings = self.cache_hit_savings * 30  # Rough projection
                
                return {
                    'cache_strategy': self.cache_strategy.value,
                    'strategy_config': self.strategy_config,
                    'total_requests': self.total_requests,
                    'cache_hits': self.total_cache_hits,
                    'cache_misses': self.cache_misses,
                    'hit_rate': hit_rate,
                    'total_savings': self.cache_hit_savings,
                    'projected_monthly_savings': projected_monthly_savings,
                    'memory_cache': memory_stats,
                    'persistent_cache': persistent_stats,
                    'compression_enabled': self.compression_enabled,
                    'ttl_hours': self.cache_ttl_hours,
                    'last_updated': datetime.now().isoformat()
                }
                
        except Exception as e:
            self.logger.error(f"Error getting cache stats: {e}")
            return {'error': str(e)}
    
    def get_cost_optimization_report(self) -> Dict[str, Any]:
        """Generate comprehensive cost optimization report"""
        stats = self.get_cache_stats()
        
        # Calculate optimization metrics
        cost_reduction_rate = stats['hit_rate']
        potential_monthly_savings = stats.get('projected_monthly_savings', 0)
        
        # Generate recommendations
        recommendations = []
        
        if stats['hit_rate'] < 0.5:
            recommendations.append("Low cache hit rate - consider increasing TTL or cache size")
        
        if stats['hit_rate'] > 0.8:
            recommendations.append("Excellent cache performance - consider increasing cache size for even better savings")
        
        if stats.get('memory_cache', {}).get('utilization', 0) > 0.9:
            recommendations.append("Memory cache near capacity - consider increasing max_memory_entries")
        
        persistent_stats = stats.get('persistent_cache', {})
        if persistent_stats.get('storage_utilization', 0) > 0.8:
            recommendations.append("Persistent cache storage near capacity - consider cleanup or expansion")
        
        return {
            'cost_optimization_summary': {
                'api_cost_reduction_rate': f"{cost_reduction_rate:.1%}",
                'estimated_monthly_savings': f"${potential_monthly_savings:.2f}",
                'total_savings_to_date': f"${stats['total_savings']:.2f}",
                'cache_efficiency': 'Excellent' if cost_reduction_rate > 0.7 else 'Good' if cost_reduction_rate > 0.5 else 'Needs Improvement'
            },
            'performance_metrics': stats,
            'optimization_recommendations': recommendations,
            'cache_health': {
                'memory_cache_healthy': stats.get('memory_cache', {}).get('utilization', 0) < 0.9,
                'persistent_cache_healthy': persistent_stats.get('storage_utilization', 0) < 0.8,
                'hit_rate_healthy': stats['hit_rate'] > 0.5
            }
        }
    
    def invalidate_cache(self, 
                        task_type: Optional[str] = None,
                        jurisdiction: Optional[str] = None,
                        judge_type: Optional[str] = None):
        """
        Invalidate cache entries based on criteria.
        
        Useful for cache invalidation when evaluation logic changes.
        
        Args:
            task_type: Invalidate entries for specific task type
            jurisdiction: Invalidate entries for specific jurisdiction  
            judge_type: Invalidate entries for specific judge type
        """
        if self.cache_strategy == CacheStrategy.DISABLED:
            return
            
        try:
            with self.lock:
                # Find matching entries in memory cache
                keys_to_remove = []
                if self.memory_cache:
                    for key, entry in self.memory_cache.items():
                        if self._entry_matches_criteria(entry, task_type, jurisdiction, judge_type):
                            keys_to_remove.append(key)
                    
                    # Remove from memory cache
                    for key in keys_to_remove:
                        del self.memory_cache[key]
                
                # TODO: Add persistent cache invalidation if needed
                # (would require more complex SQL queries)
                
                self.logger.info(f"Invalidated {len(keys_to_remove)} cache entries "
                               f"(strategy: {self.cache_strategy.value}, task: {task_type}, "
                               f"jurisdiction: {jurisdiction}, judge: {judge_type})")
                
        except Exception as e:
            self.logger.error(f"Error invalidating cache: {e}")
    
    def _entry_matches_criteria(self, 
                               entry: CacheEntry,
                               task_type: Optional[str],
                               jurisdiction: Optional[str], 
                               judge_type: Optional[str]) -> bool:
        """Check if cache entry matches invalidation criteria"""
        if task_type and entry.task_type != task_type:
            return False
        if jurisdiction and entry.jurisdiction != jurisdiction:
            return False
        if judge_type and entry.judge_type != judge_type:
            return False
        return True
    
    def close(self):
        """Clean up resources"""
        try:
            # Final cleanup
            self.cleanup_expired_cache()
            
            # Log final stats
            final_stats = self.get_cache_stats()
            self.logger.info(f"Cache closing - final stats: {final_stats['hit_rate']:.1%} hit rate, "
                           f"${final_stats['total_savings']:.2f} total savings")
            
        except Exception as e:
            self.logger.error(f"Error during cache cleanup: {e}")


# Factory function for easy cache creation
def create_cache_with_strategy(strategy: CacheStrategy, config: Optional[Dict[str, Any]] = None) -> MultiStrategyLegalRewardCache:
    """
    Create cache with specific strategy and optimal settings.
    
    Args:
        strategy: Caching strategy to use
        config: Optional configuration overrides
        
    Returns:
        Configured cache instance
    """
    # Base configuration
    base_config = {
        "cache_strategy": strategy.value,
        "cache_dir": "/tmp/legal_reward_cache"
    }
    
    # Merge with provided config
    if config:
        base_config.update(config)
    
    return MultiStrategyLegalRewardCache(base_config)


def create_aggressive_cache(config: Dict[str, Any]) -> MultiStrategyLegalRewardCache:
    """
    Create aggressive cache with optimal settings for legal reward system.
    
    Args:
        config: Cache configuration dictionary
        
    Returns:
        Configured MultiStrategyLegalRewardCache instance
    """
    # Set optimal defaults for aggressive caching strategy
    default_config = {
        "cache_strategy": "aggressive",
        "cache_dir": "/tmp/legal_reward_cache",
        "max_cache_size_gb": 10,
        "cache_ttl_hours": 168,  # 1 week - legal evaluations are stable
        "compression": True,
        "persist_across_sessions": True,
        "max_memory_entries": 1000
    }
    
    # Merge with provided config
    final_config = {**default_config, **config}
    
    return MultiStrategyLegalRewardCache(final_config)


def get_strategy_configurations() -> Dict[str, Dict[str, Any]]:
    """
    Get all available caching strategies and their default configurations.
    
    Returns:
        Dictionary mapping strategy names to their configurations
    """
    return {
        "aggressive": {
            "description": "Cache everything with long TTL for maximum cost savings",
            "cache_ttl_hours": 168,  # 1 week
            "max_cache_size_gb": 10,
            "compression": True,
            "persist_across_sessions": True,
            "max_memory_entries": 1000,
            "cache_threshold": 0.0,  # Cache everything
            "expected_hit_rate": 0.8,
            "expected_cost_reduction": "60-80%",
            "use_case": "Production training with stable legal content"
        },
        "balanced": {
            "description": "Balanced caching for moderate cost savings with lower storage",
            "cache_ttl_hours": 72,   # 3 days
            "max_cache_size_gb": 5,
            "compression": True,
            "persist_across_sessions": True,
            "max_memory_entries": 500,
            "cache_threshold": 0.02,  # Only cache if cost > $0.02
            "expected_hit_rate": 0.6,
            "expected_cost_reduction": "40-60%",
            "use_case": "Development and testing with moderate storage constraints"
        },
        "conservative": {
            "description": "Minimal caching for low storage usage",
            "cache_ttl_hours": 24,   # 1 day
            "max_cache_size_gb": 2,
            "compression": False,
            "persist_across_sessions": False,
            "max_memory_entries": 200,
            "cache_threshold": 0.05,  # Only cache if cost > $0.05
            "expected_hit_rate": 0.3,
            "expected_cost_reduction": "15-30%",
            "use_case": "Resource-constrained environments or debugging"
        },
        "disabled": {
            "description": "No caching - all requests go to API",
            "cache_ttl_hours": 0,
            "max_cache_size_gb": 0,
            "compression": False,
            "persist_across_sessions": False,
            "max_memory_entries": 0,
            "cache_threshold": float('inf'),
            "expected_hit_rate": 0.0,
            "expected_cost_reduction": "0%",
            "use_case": "Testing, debugging, or when fresh responses are always required"
        }
    }


# Context manager for automatic cache management
class ManagedCache:
    """Context manager for automatic cache lifecycle management"""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.cache = None
    
    def __enter__(self) -> MultiStrategyLegalRewardCache:
        self.cache = create_aggressive_cache(self.config)
        return self.cache
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        if self.cache:
            self.cache.close()