"""
Caching service for AI model results and API responses
"""
import json
import hashlib
from typing import Any, Optional, Dict, List
from datetime import datetime, timedelta
import structlog

from ..core.config import settings

logger = structlog.get_logger(__name__)


class InMemoryCache:
    """Simple in-memory cache with TTL support"""
    
    def __init__(self, default_ttl: int = 3600):
        self.cache: Dict[str, Dict[str, Any]] = {}
        self.default_ttl = default_ttl
        self.logger = logger.bind(service="cache")
    
    def _generate_key(self, data: Any) -> str:
        """Generate cache key from data"""
        if isinstance(data, (dict, list)):
            serialized = json.dumps(data, sort_keys=True)
        else:
            serialized = str(data)
        return hashlib.md5(serialized.encode()).hexdigest()
    
    def _is_expired(self, cache_entry: Dict[str, Any]) -> bool:
        """Check if cache entry is expired"""
        expires_at = cache_entry.get("expires_at")
        if not expires_at:
            return False
        return datetime.utcnow() > expires_at
    
    async def get(self, key: str) -> Optional[Any]:
        """Get value from cache"""
        if key in self.cache:
            entry = self.cache[key]
            if not self._is_expired(entry):
                self.logger.debug("Cache hit", key=key)
                return entry["value"]
            else:
                # Remove expired entry
                del self.cache[key]
                self.logger.debug("Cache miss (expired)", key=key)
        
        self.logger.debug("Cache miss", key=key)
        return None
    
    async def set(self, key: str, value: Any, ttl: Optional[int] = None) -> None:
        """Set value in cache with TTL"""
        ttl = ttl or self.default_ttl
        expires_at = datetime.utcnow() + timedelta(seconds=ttl)
        
        self.cache[key] = {
            "value": value,
            "created_at": datetime.utcnow(),
            "expires_at": expires_at
        }
        
        self.logger.debug("Cache set", key=key, ttl=ttl)
    
    async def delete(self, key: str) -> bool:
        """Delete value from cache"""
        if key in self.cache:
            del self.cache[key]
            self.logger.debug("Cache delete", key=key)
            return True
        return False
    
    async def clear(self) -> None:
        """Clear all cache entries"""
        count = len(self.cache)
        self.cache.clear()
        self.logger.info("Cache cleared", cleared_count=count)
    
    async def cleanup_expired(self) -> int:
        """Remove expired entries"""
        expired_keys = []
        for key, entry in self.cache.items():
            if self._is_expired(entry):
                expired_keys.append(key)
        
        for key in expired_keys:
            del self.cache[key]
        
        if expired_keys:
            self.logger.info("Cleaned up expired cache entries", removed_count=len(expired_keys))
        
        return len(expired_keys)
    
    def get_stats(self) -> Dict[str, Any]:
        """Get cache statistics"""
        total_entries = len(self.cache)
        expired_count = sum(1 for entry in self.cache.values() if self._is_expired(entry))
        
        return {
            "total_entries": total_entries,
            "active_entries": total_entries - expired_count,
            "expired_entries": expired_count,
            "cache_size_mb": len(str(self.cache)) / 1024 / 1024
        }


class CacheService:
    """
    Cache service for AI model results and API responses.
    Provides unified caching interface with different backends.
    """
    
    def __init__(self, cache_backend: Optional[str] = None):
        self.backend_type = cache_backend or "memory"
        self.logger = logger.bind(service="cache_service")
        
        # Initialize cache backend
        if self.backend_type == "memory":
            self.cache = InMemoryCache(default_ttl=settings.cache_ttl)
        else:
            # TODO: Implement Redis backend
            self.cache = InMemoryCache(default_ttl=settings.cache_ttl)
            self.logger.warning("Redis not implemented, falling back to memory cache")
    
    async def cache_gpt_result(self, prompt: str, context: Dict[str, Any], result: Any, ttl: int = 7200) -> None:
        """Cache GPT API result"""
        cache_key = f"gpt:{self.cache._generate_key({'prompt': prompt, 'context': context})}"
        await self.cache.set(cache_key, result, ttl)
        self.logger.debug("GPT result cached", cache_key=cache_key[:16])
    
    async def get_gpt_result(self, prompt: str, context: Dict[str, Any]) -> Optional[Any]:
        """Get cached GPT result"""
        cache_key = f"gpt:{self.cache._generate_key({'prompt': prompt, 'context': context})}"
        result = await self.cache.get(cache_key)
        if result:
            self.logger.debug("GPT cache hit", cache_key=cache_key[:16])
        return result
    
    async def cache_clip_result(self, image_path: str, elements: list[Any], prompt: str, result: Any, ttl: int = 3600) -> None:
        """Cache CLIP filtering result"""
        # Create cache key from image path, elements, and prompt
        cache_data = {
            "image_path": image_path,
            "elements_hash": self.cache._generate_key(elements),
            "prompt": prompt
        }
        cache_key = f"clip:{self.cache._generate_key(cache_data)}"
        await self.cache.set(cache_key, result, ttl)
        self.logger.debug("CLIP result cached", cache_key=cache_key[:16])
    
    async def get_clip_result(self, image_path: str, elements: list[Any], prompt: str) -> Optional[Any]:
        """Get cached CLIP result"""
        cache_data = {
            "image_path": image_path,
            "elements_hash": self.cache._generate_key(elements),
            "prompt": prompt
        }
        cache_key = f"clip:{self.cache._generate_key(cache_data)}"
        result = await self.cache.get(cache_key)
        if result:
            self.logger.debug("CLIP cache hit", cache_key=cache_key[:16])
        return result
    
    async def cache_omniparser_result(self, image_path: str, result: Any, ttl: int = 1800) -> None:
        """Cache OmniParser result"""
        cache_key = f"omniparser:{self.cache._generate_key(image_path)}"
        await self.cache.set(cache_key, result, ttl)
        self.logger.debug("OmniParser result cached", cache_key=cache_key[:16])
    
    async def get_omniparser_result(self, image_path: str) -> Optional[Any]:
        """Get cached OmniParser result"""
        cache_key = f"omniparser:{self.cache._generate_key(image_path)}"
        result = await self.cache.get(cache_key)
        if result:
            self.logger.debug("OmniParser cache hit", cache_key=cache_key[:16])
        return result
    
    async def get_health_status(self) -> Dict[str, Any]:
        """Get cache health status"""
        try:
            stats = self.cache.get_stats()
            return {
                "status": "healthy",
                "backend": self.backend_type,
                "stats": stats
            }
        except Exception as e:
            return {
                "status": "unhealthy",
                "error": str(e),
                "backend": self.backend_type
            }
    
    async def cleanup(self) -> None:
        """Cleanup cache resources"""
        await self.cache.cleanup_expired()
        self.logger.info("Cache cleanup completed")


# Global cache service instance
_cache_service: Optional[CacheService] = None


async def get_cache_service() -> CacheService:
    """Get or create the global cache service instance"""
    global _cache_service
    
    if _cache_service is None:
        _cache_service = CacheService()
    
    return _cache_service