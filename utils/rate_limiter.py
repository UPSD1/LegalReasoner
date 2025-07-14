"""
Smart Rate Limiting System for Multi-Task Legal Reward System

This module implements intelligent rate limiting to prevent API overages and optimize
cost across multiple providers (OpenAI, Anthropic, Google) with sophisticated
fallback strategies and cost-aware request routing.

Key Features:
- Multi-provider token bucket rate limiting
- Automatic failover to cheaper providers when rate limited
- Cost-aware request routing (expensive models for complex tasks)
- Training-optimized batching and request distribution
- Intelligent backoff strategies with exponential backoff
- Budget-aware request management
- Real-time rate limit monitoring and adjustment

Rate Limiting Strategy:
- Token Bucket Algorithm: Smooth rate limiting with burst handling
- Per-Provider Limits: Separate rate limits for each API provider
- Intelligent Fallback: Cheaper providers when primary is rate limited
- Cost Optimization: Route simple tasks to cheaper providers
- Training Focus: Optimized for high-throughput legal AI training
"""

import asyncio
import time
import threading
from typing import Dict, List, Optional, Tuple, Any, Union
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from collections import defaultdict, deque
import random
import math

# Import core components
from core import (
    APIProvider, RateLimitExceededError, APIClientError, 
    LegalTaskType, create_error_context, ERROR_CODES
)
from utils.logging import get_legal_logger


@dataclass
class RateLimitConfig:
    """Configuration for a single API provider's rate limits"""
    requests_per_minute: int
    tokens_per_minute: int
    requests_per_hour: Optional[int] = None
    tokens_per_hour: Optional[int] = None
    max_concurrent_requests: int = 10
    cost_per_1k_input_tokens: float = 0.01
    cost_per_1k_output_tokens: float = 0.03
    fallback_priority: int = 1  # Lower number = higher priority for fallback


@dataclass
class RateLimitStatus:
    """Current rate limit status for a provider"""
    provider: APIProvider
    requests_remaining: int
    tokens_remaining: int
    reset_time: float
    is_rate_limited: bool = False
    last_request_time: float = 0.0
    consecutive_failures: int = 0
    backoff_until: float = 0.0
    total_requests_made: int = 0
    total_cost_incurred: float = 0.0


class TokenBucket:
    """
    Token bucket implementation for smooth rate limiting.
    
    Allows burst requests up to bucket capacity while maintaining
    average rate over time. Ideal for API rate limiting with
    burst tolerance.
    """
    
    def __init__(self, capacity: int, refill_rate: float, refill_period: float = 60.0):
        """
        Initialize token bucket.
        
        Args:
            capacity: Maximum tokens in bucket (burst capacity)
            refill_rate: Tokens added per refill period
            refill_period: Period in seconds for refill (default: 60 seconds)
        """
        self.capacity = capacity
        self.refill_rate = refill_rate
        self.refill_period = refill_period
        self.tokens = float(capacity)  # Start with full bucket
        self.last_refill = time.time()
        self.lock = threading.Lock()
    
    def consume(self, tokens: int) -> bool:
        """
        Attempt to consume tokens from bucket.
        
        Args:
            tokens: Number of tokens to consume
            
        Returns:
            True if tokens were consumed, False if insufficient tokens
        """
        with self.lock:
            self._refill()
            
            if self.tokens >= tokens:
                self.tokens -= tokens
                return True
            return False
    
    def peek(self) -> float:
        """
        Check current token count without consuming.
        
        Returns:
            Current number of tokens available
        """
        with self.lock:
            self._refill()
            return self.tokens
    
    def _refill(self):
        """Refill tokens based on elapsed time"""
        now = time.time()
        elapsed = now - self.last_refill
        
        if elapsed > 0:
            # Calculate tokens to add based on elapsed time
            tokens_to_add = (elapsed / self.refill_period) * self.refill_rate
            self.tokens = min(self.capacity, self.tokens + tokens_to_add)
            self.last_refill = now
    
    def time_until_tokens(self, tokens: int) -> float:
        """
        Calculate time until specified tokens are available.
        
        Args:
            tokens: Number of tokens needed
            
        Returns:
            Time in seconds until tokens are available (0 if available now)
        """
        with self.lock:
            self._refill()
            
            if self.tokens >= tokens:
                return 0.0
            
            tokens_needed = tokens - self.tokens
            time_needed = (tokens_needed / self.refill_rate) * self.refill_period
            return time_needed


class TokenBucketLimiter:
    """
    Token bucket rate limiter for individual API providers.
    
    Implements separate token buckets for requests and tokens,
    allowing fine-grained control over both request rate and
    token consumption rate.
    """
    
    def __init__(self, config: RateLimitConfig):
        self.config = config
        self.provider = None  # Set when attached to provider
        
        # Request rate limiting
        self.request_bucket = TokenBucket(
            capacity=config.requests_per_minute,
            refill_rate=config.requests_per_minute,
            refill_period=60.0
        )
        
        # Token rate limiting
        self.token_bucket = TokenBucket(
            capacity=config.tokens_per_minute,
            refill_rate=config.tokens_per_minute,
            refill_period=60.0
        )
        
        # Concurrent request limiting
        self.concurrent_requests = 0
        self.concurrent_lock = threading.Lock()
        
        self.logger = get_legal_logger("token_bucket_limiter")
    
    def can_make_request(self, estimated_tokens: int = 1000) -> bool:
        """
        Check if request can be made within rate limits.
        
        Args:
            estimated_tokens: Estimated tokens for the request
            
        Returns:
            True if request can be made immediately
        """
        # Check concurrent request limit
        with self.concurrent_lock:
            if self.concurrent_requests >= self.config.max_concurrent_requests:
                return False
        
        # Check if we have tokens for both request and token buckets
        return (self.request_bucket.peek() >= 1 and 
                self.token_bucket.peek() >= estimated_tokens)
    
    async def acquire(self, estimated_tokens: int = 1000) -> bool:
        """
        Acquire permission to make request (async version).
        
        Args:
            estimated_tokens: Estimated tokens for the request
            
        Returns:
            True if permission acquired, False if should fallback
        """
        # Check concurrent requests
        with self.concurrent_lock:
            if self.concurrent_requests >= self.config.max_concurrent_requests:
                # Wait a bit and try again
                await asyncio.sleep(0.1)
                if self.concurrent_requests >= self.config.max_concurrent_requests:
                    return False
        
        # Try to consume tokens
        request_success = self.request_bucket.consume(1)
        token_success = self.token_bucket.consume(estimated_tokens)
        
        if request_success and token_success:
            # Increment concurrent requests
            with self.concurrent_lock:
                self.concurrent_requests += 1
            return True
        
        # If we consumed one but not the other, we need to handle partial consumption
        # This is a simplified approach - in production you might want more sophisticated logic
        return False
    
    def release(self):
        """Release a concurrent request slot"""
        with self.concurrent_lock:
            if self.concurrent_requests > 0:
                self.concurrent_requests -= 1
    
    def get_wait_time(self, estimated_tokens: int = 1000) -> float:
        """
        Get estimated wait time until request can be made.
        
        Args:
            estimated_tokens: Estimated tokens for the request
            
        Returns:
            Wait time in seconds
        """
        request_wait = self.request_bucket.time_until_tokens(1)
        token_wait = self.token_bucket.time_until_tokens(estimated_tokens)
        
        return max(request_wait, token_wait)
    
    def get_status(self) -> Dict[str, Any]:
        """Get current rate limiter status"""
        return {
            'requests_available': int(self.request_bucket.peek()),
            'tokens_available': int(self.token_bucket.peek()),
            'concurrent_requests': self.concurrent_requests,
            'max_concurrent': self.config.max_concurrent_requests,
            'requests_per_minute': self.config.requests_per_minute,
            'tokens_per_minute': self.config.tokens_per_minute
        }


class IntelligentBackoffStrategy:
    """
    Intelligent backoff strategy for rate limited requests.
    
    Implements exponential backoff with jitter and provider-specific
    strategies to handle different types of rate limiting scenarios.
    """
    
    def __init__(self, base_delay: float = 1.0, max_delay: float = 300.0, 
                 exponential_base: float = 2.0, jitter: bool = True):
        self.base_delay = base_delay
        self.max_delay = max_delay
        self.exponential_base = exponential_base
        self.jitter = jitter
        self.provider_backoffs = {}  # Track backoff per provider
        
    def calculate_backoff(self, provider: APIProvider, consecutive_failures: int) -> float:
        """
        Calculate backoff delay for a provider.
        
        Args:
            provider: API provider experiencing rate limiting
            consecutive_failures: Number of consecutive failures
            
        Returns:
            Backoff delay in seconds
        """
        # Exponential backoff calculation
        delay = self.base_delay * (self.exponential_base ** consecutive_failures)
        delay = min(delay, self.max_delay)
        
        # Add jitter to prevent thundering herd
        if self.jitter:
            jitter_factor = 0.1  # 10% jitter
            jitter_range = delay * jitter_factor
            delay += random.uniform(-jitter_range, jitter_range)
        
        # Provider-specific adjustments
        if provider == APIProvider.OPENAI:
            # OpenAI tends to have stricter rate limits
            delay *= 1.2
        elif provider == APIProvider.GOOGLE:
            # Google is usually more forgiving
            delay *= 0.8
        
        return max(0, delay)
    
    def should_backoff(self, provider: APIProvider) -> Tuple[bool, float]:
        """
        Check if provider should be backed off.
        
        Args:
            provider: API provider to check
            
        Returns:
            Tuple of (should_backoff, remaining_backoff_time)
        """
        if provider.value not in self.provider_backoffs:
            return False, 0.0
        
        backoff_until = self.provider_backoffs[provider.value]
        current_time = time.time()
        
        if current_time < backoff_until:
            return True, backoff_until - current_time
        
        # Backoff period expired
        del self.provider_backoffs[provider.value]
        return False, 0.0
    
    def apply_backoff(self, provider: APIProvider, consecutive_failures: int):
        """
        Apply backoff to a provider.
        
        Args:
            provider: API provider to backoff
            consecutive_failures: Number of consecutive failures
        """
        delay = self.calculate_backoff(provider, consecutive_failures)
        backoff_until = time.time() + delay
        self.provider_backoffs[provider.value] = backoff_until
    
    def reset_backoff(self, provider: APIProvider):
        """Reset backoff for a provider after successful request"""
        if provider.value in self.provider_backoffs:
            del self.provider_backoffs[provider.value]


class SlidingWindowLimiter:
    """
    Sliding window rate limiter implementation.
    
    Tracks requests in a sliding time window for more precise rate limiting
    compared to fixed windows. More accurate but higher memory usage.
    """
    
    def __init__(self, requests_per_window: int, window_seconds: float):
        self.requests_per_window = requests_per_window
        self.window_seconds = window_seconds
        self.request_times: List[float] = []
        self.lock = asyncio.Lock()
    
    async def acquire(self, tokens: int = 1) -> bool:
        """
        Acquire tokens using sliding window algorithm.
        
        Args:
            tokens: Number of tokens to acquire
            
        Returns:
            True if tokens acquired, False if rate limited
        """
        async with self.lock:
            current_time = time.time()
            
            # Remove old requests outside the window
            cutoff_time = current_time - self.window_seconds
            self.request_times = [t for t in self.request_times if t > cutoff_time]
            
            # Check if we can make the request
            if len(self.request_times) + tokens <= self.requests_per_window:
                # Add tokens for this request
                for _ in range(tokens):
                    self.request_times.append(current_time)
                return True
            
            return False
    
    def get_status(self) -> Dict[str, Any]:
        """Get current sliding window status"""
        current_time = time.time()
        cutoff_time = current_time - self.window_seconds
        active_requests = len([t for t in self.request_times if t > cutoff_time])
        
        return {
            "requests_in_window": active_requests,
            "requests_per_window": self.requests_per_window,
            "window_seconds": self.window_seconds,
            "capacity_remaining": self.requests_per_window - active_requests,
            "utilization": active_requests / self.requests_per_window
        }


class FixedWindowLimiter:
    """
    Fixed window rate limiter implementation.
    
    Divides time into fixed windows and allows a set number of requests
    per window. Simple and memory efficient but can allow bursts at window boundaries.
    """
    
    def __init__(self, requests_per_window: int, window_seconds: float):
        self.requests_per_window = requests_per_window
        self.window_seconds = window_seconds
        self.current_window_start = 0.0
        self.current_window_count = 0
        self.lock = asyncio.Lock()
    
    async def acquire(self, tokens: int = 1) -> bool:
        """
        Acquire tokens using fixed window algorithm.
        
        Args:
            tokens: Number of tokens to acquire
            
        Returns:
            True if tokens acquired, False if rate limited
        """
        async with self.lock:
            current_time = time.time()
            
            # Check if we need to start a new window
            if current_time >= self.current_window_start + self.window_seconds:
                self.current_window_start = current_time
                self.current_window_count = 0
            
            # Check if we can make the request
            if self.current_window_count + tokens <= self.requests_per_window:
                self.current_window_count += tokens
                return True
            
            return False
    
    def get_status(self) -> Dict[str, Any]:
        """Get current fixed window status"""
        current_time = time.time()
        
        # Check if window has expired
        if current_time >= self.current_window_start + self.window_seconds:
            window_count = 0
            time_until_reset = 0.0
        else:
            window_count = self.current_window_count
            time_until_reset = (self.current_window_start + self.window_seconds) - current_time
        
        return {
            "requests_in_window": window_count,
            "requests_per_window": self.requests_per_window,
            "window_seconds": self.window_seconds,
            "capacity_remaining": self.requests_per_window - window_count,
            "utilization": window_count / self.requests_per_window,
            "time_until_reset": time_until_reset
        }


class MultiProviderRateLimiter:
    """
    Smart rate limiter managing all three API providers optimally.
    
    Features:
    - Per-provider token bucket rate limiting
    - Automatic failover to cheaper providers when rate limited
    - Cost-aware request routing
    - Training-optimized batching
    - Intelligent backoff strategies
    """
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.logger = get_legal_logger("multi_provider_rate_limiter")
        
        # Initialize provider configurations
        self.provider_configs = self._load_provider_configs(config)
        
        # Initialize rate limiters for each provider
        self.limiters = {}
        for provider, provider_config in self.provider_configs.items():
            limiter = TokenBucketLimiter(provider_config)
            limiter.provider = provider
            self.limiters[provider] = limiter
        
        # Rate limit status tracking
        self.provider_status = {}
        for provider in APIProvider:
            self.provider_status[provider] = RateLimitStatus(
                provider=provider,
                requests_remaining=self.provider_configs[provider].requests_per_minute,
                tokens_remaining=self.provider_configs[provider].tokens_per_minute,
                reset_time=time.time() + 60
            )
        
        # Backoff strategy
        self.backoff_strategy = IntelligentBackoffStrategy()
        
        # Fallback chains
        self.fallback_chains = self._setup_fallback_chains()
        
        # Performance tracking
        self.total_requests = 0
        self.successful_requests = 0
        self.failed_requests = 0
        self.fallback_used_count = 0
        
        self.logger.info(f"Initialized multi-provider rate limiter with {len(self.limiters)} providers")
    
    def _load_provider_configs(self, config: Dict[str, Any]) -> Dict[APIProvider, RateLimitConfig]:
        """Load rate limit configurations for all providers"""
        provider_configs = {}
        
        # Default configurations (can be overridden)
        defaults = {
            APIProvider.OPENAI: RateLimitConfig(
                requests_per_minute=500,
                tokens_per_minute=30000,
                cost_per_1k_input_tokens=0.01,
                cost_per_1k_output_tokens=0.03,
                fallback_priority=1
            ),
            APIProvider.ANTHROPIC: RateLimitConfig(
                requests_per_minute=400,
                tokens_per_minute=40000,
                cost_per_1k_input_tokens=0.003,
                cost_per_1k_output_tokens=0.015,
                fallback_priority=2
            ),
            APIProvider.GOOGLE: RateLimitConfig(
                requests_per_minute=300,
                tokens_per_minute=32000,
                cost_per_1k_input_tokens=0.00125,
                cost_per_1k_output_tokens=0.005,
                fallback_priority=3
            )
        }
        
        # Apply configuration overrides
        for provider in APIProvider:
            provider_key = provider.value
            if provider_key in config:
                # Merge with defaults
                provider_config = defaults[provider]
                for key, value in config[provider_key].items():
                    if hasattr(provider_config, key):
                        setattr(provider_config, key, value)
                provider_configs[provider] = provider_config
            else:
                provider_configs[provider] = defaults[provider]
        
        return provider_configs
    
    def _setup_fallback_chains(self) -> Dict[APIProvider, List[APIProvider]]:
        """Setup fallback chains based on cost and availability"""
        # Sort providers by fallback priority (cost-effectiveness)
        sorted_providers = sorted(
            APIProvider, 
            key=lambda p: self.provider_configs[p].fallback_priority
        )
        
        fallback_chains = {}
        for primary in APIProvider:
            # Create chain: primary first, then others by priority
            chain = [primary]
            for provider in sorted_providers:
                if provider != primary:
                    chain.append(provider)
            fallback_chains[primary] = chain
        
        return fallback_chains
    
    async def execute_request(self, provider: APIProvider, 
                            request_data: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """
        Execute API request with rate limiting and cost tracking.
        
        Args:
            provider: Primary API provider to use
            request_data: Request data including estimated tokens
            
        Returns:
            API response data or None if failed
        """
        self.total_requests += 1
        estimated_tokens = request_data.get('estimated_tokens', 1000)
        
        # Get fallback chain starting with primary provider
        fallback_chain = self.fallback_chains.get(provider, [provider])
        
        for attempt_provider in fallback_chain:
            try:
                # Check if provider is backed off
                should_backoff, backoff_time = self.backoff_strategy.should_backoff(attempt_provider)
                if should_backoff:
                    self.logger.debug(f"Provider {attempt_provider.value} is backed off for {backoff_time:.1f}s")
                    continue
                
                # Try to acquire rate limit permission
                limiter = self.limiters[attempt_provider]
                if await limiter.acquire(estimated_tokens):
                    try:
                        # Simulate API request (actual implementation would call real API)
                        result = await self._simulate_api_request(attempt_provider, request_data)
                        
                        # Track successful request
                        self._track_successful_request(attempt_provider, request_data)
                        self.successful_requests += 1
                        
                        if attempt_provider != provider:
                            self.fallback_used_count += 1
                            self.logger.info(f"Fallback successful: {provider.value} -> {attempt_provider.value}")
                        
                        return result
                        
                    except RateLimitExceededError as e:
                        # Handle rate limit from provider
                        self._handle_rate_limit_error(attempt_provider, e)
                        continue
                        
                    except Exception as e:
                        # Handle other API errors
                        self._handle_api_error(attempt_provider, e)
                        continue
                        
                    finally:
                        # Always release the concurrent request slot
                        limiter.release()
                
                else:
                    # Rate limit would be exceeded, try next provider
                    wait_time = limiter.get_wait_time(estimated_tokens)
                    self.logger.debug(f"Rate limit would be exceeded for {attempt_provider.value}, "
                                    f"wait time: {wait_time:.1f}s")
                    continue
            
            except Exception as e:
                self.logger.error(f"Unexpected error with provider {attempt_provider.value}: {e}")
                continue
        
        # All providers failed
        self.failed_requests += 1
        self.logger.error(f"All providers failed for request")
        return None
    
    async def _simulate_api_request(self, provider: APIProvider, 
                                  request_data: Dict[str, Any]) -> Dict[str, Any]:
        """Simulate API request (replace with real API calls in production)"""
        # Simulate network delay
        await asyncio.sleep(random.uniform(0.1, 0.5))
        
        # Simulate occasional rate limit errors
        if random.random() < 0.05:  # 5% chance of rate limit error
            raise RateLimitExceededError(
                f"Rate limit exceeded for {provider.value}",
                provider=provider.value,
                reset_time=time.time() + 60
            )
        
        # Simulate successful response
        return {
            'provider': provider.value,
            'response': f"Simulated response from {provider.value}",
            'tokens_used': request_data.get('estimated_tokens', 1000),
            'success': True
        }
    
    def _track_successful_request(self, provider: APIProvider, request_data: Dict[str, Any]):
        """Track successful request for rate limit and cost monitoring"""
        status = self.provider_status[provider]
        status.total_requests_made += 1
        status.last_request_time = time.time()
        status.consecutive_failures = 0  # Reset failure count
        
        # Reset backoff on success
        self.backoff_strategy.reset_backoff(provider)
        
        # Estimate cost
        tokens_used = request_data.get('estimated_tokens', 1000)
        config = self.provider_configs[provider]
        estimated_cost = (tokens_used / 1000) * config.cost_per_1k_input_tokens
        status.total_cost_incurred += estimated_cost
    
    def _handle_rate_limit_error(self, provider: APIProvider, error: RateLimitExceededError):
        """Handle rate limit error from provider"""
        status = self.provider_status[provider]
        status.is_rate_limited = True
        status.consecutive_failures += 1
        
        # Apply intelligent backoff
        self.backoff_strategy.apply_backoff(provider, status.consecutive_failures)
        
        self.logger.warning(f"Rate limit exceeded for {provider.value}, applying backoff")
    
    def _handle_api_error(self, provider: APIProvider, error: Exception):
        """Handle general API error"""
        status = self.provider_status[provider]
        status.consecutive_failures += 1
        
        # Apply backoff for repeated failures
        if status.consecutive_failures >= 3:
            self.backoff_strategy.apply_backoff(provider, status.consecutive_failures)
            self.logger.warning(f"Multiple failures for {provider.value}, applying backoff")
    
    def get_optimal_provider(self, task_complexity: str = "medium", 
                           content_length: int = 1000) -> APIProvider:
        """
        Choose optimal provider based on rate limits, costs, and task complexity.
        
        Args:
            task_complexity: "simple", "medium", or "complex"
            content_length: Estimated content length in characters
            
        Returns:
            Optimal API provider for the task
        """
        # Estimate tokens from content length
        estimated_tokens = max(100, content_length // 4)  # Rough estimate: 4 chars per token
        
        # For complex legal tasks, prefer higher quality providers
        if task_complexity == "complex":
            preferred_order = [APIProvider.OPENAI, APIProvider.ANTHROPIC, APIProvider.GOOGLE]
        elif task_complexity == "simple":
            # For simple tasks, prefer cheaper providers
            preferred_order = [APIProvider.GOOGLE, APIProvider.ANTHROPIC, APIProvider.OPENAI]
        else:
            # Balanced approach
            preferred_order = [APIProvider.ANTHROPIC, APIProvider.OPENAI, APIProvider.GOOGLE]
        
        # Find first available provider in preferred order
        for provider in preferred_order:
            # Check if provider is backed off
            should_backoff, _ = self.backoff_strategy.should_backoff(provider)
            if should_backoff:
                continue
            
            # Check if provider can handle request
            limiter = self.limiters[provider]
            if limiter.can_make_request(estimated_tokens):
                return provider
        
        # If no preferred provider available, return least loaded
        return self._get_least_loaded_provider()
    
    def _get_least_loaded_provider(self) -> APIProvider:
        """Get provider with most available capacity"""
        best_provider = APIProvider.OPENAI
        best_score = -1
        
        for provider in APIProvider:
            # Skip backed off providers
            should_backoff, _ = self.backoff_strategy.should_backoff(provider)
            if should_backoff:
                continue
            
            limiter = self.limiters[provider]
            status = limiter.get_status()
            
            # Calculate availability score (0-1, higher is better)
            request_availability = status['requests_available'] / status['requests_per_minute']
            token_availability = status['tokens_available'] / status['tokens_per_minute']
            concurrent_availability = 1 - (status['concurrent_requests'] / status['max_concurrent'])
            
            score = (request_availability + token_availability + concurrent_availability) / 3
            
            if score > best_score:
                best_score = score
                best_provider = provider
        
        return best_provider
    
    def get_comprehensive_status(self) -> Dict[str, Any]:
        """Get comprehensive status of all providers and rate limiting"""
        status = {
            'total_requests': self.total_requests,
            'successful_requests': self.successful_requests,
            'failed_requests': self.failed_requests,
            'success_rate': self.successful_requests / max(self.total_requests, 1),
            'fallback_used_count': self.fallback_used_count,
            'fallback_rate': self.fallback_used_count / max(self.total_requests, 1),
            'providers': {}
        }
        
        for provider in APIProvider:
            provider_status = self.provider_status[provider]
            limiter_status = self.limiters[provider].get_status()
            should_backoff, backoff_time = self.backoff_strategy.should_backoff(provider)
            
            status['providers'][provider.value] = {
                'rate_limit_status': limiter_status,
                'total_requests': provider_status.total_requests_made,
                'total_cost': provider_status.total_cost_incurred,
                'consecutive_failures': provider_status.consecutive_failures,
                'is_backed_off': should_backoff,
                'backoff_time_remaining': backoff_time,
                'last_request_time': provider_status.last_request_time
            }
        
        return status
    
    def reset_stats(self):
        """Reset performance statistics"""
        self.total_requests = 0
        self.successful_requests = 0
        self.failed_requests = 0
        self.fallback_used_count = 0
        
        for provider in APIProvider:
            status = self.provider_status[provider]
            status.total_requests_made = 0
            status.total_cost_incurred = 0.0
            status.consecutive_failures = 0
        
        self.logger.info("Rate limiter statistics reset")


# Factory functions for easy setup

def create_production_rate_limiter(config: Optional[Dict[str, Any]] = None) -> MultiProviderRateLimiter:
    """
    Create production-ready rate limiter with optimized settings.
    
    Args:
        config: Optional configuration overrides
        
    Returns:
        Configured MultiProviderRateLimiter
    """
    default_config = {
        'openai': {
            'requests_per_minute': 500,
            'tokens_per_minute': 30000,
            'max_concurrent_requests': 10
        },
        'anthropic': {
            'requests_per_minute': 400,
            'tokens_per_minute': 40000,
            'max_concurrent_requests': 8
        },
        'google': {
            'requests_per_minute': 300,
            'tokens_per_minute': 32000,
            'max_concurrent_requests': 6
        }
    }
    
    if config:
        # Deep merge configurations
        for provider, provider_config in config.items():
            if provider in default_config:
                default_config[provider].update(provider_config)
            else:
                default_config[provider] = provider_config
    
    return MultiProviderRateLimiter(default_config)


def create_development_rate_limiter() -> MultiProviderRateLimiter:
    """Create rate limiter with conservative settings for development"""
    dev_config = {
        'openai': {
            'requests_per_minute': 50,
            'tokens_per_minute': 5000,
            'max_concurrent_requests': 3
        },
        'anthropic': {
            'requests_per_minute': 40,
            'tokens_per_minute': 4000,
            'max_concurrent_requests': 2
        },
        'google': {
            'requests_per_minute': 30,
            'tokens_per_minute': 3000,
            'max_concurrent_requests': 2
        }
    }
    
    return MultiProviderRateLimiter(dev_config)


def create_aggressive_rate_limiter() -> MultiProviderRateLimiter:
    """Create rate limiter with aggressive settings for maximum throughput"""
    aggressive_config = {
        'openai': {
            'requests_per_minute': 1000,
            'tokens_per_minute': 60000,
            'max_concurrent_requests': 20
        },
        'anthropic': {
            'requests_per_minute': 800,
            'tokens_per_minute': 80000,
            'max_concurrent_requests': 15
        },
        'google': {
            'requests_per_minute': 600,
            'tokens_per_minute': 64000,
            'max_concurrent_requests': 12
        }
    }
    
    return MultiProviderRateLimiter(aggressive_config)


# Context manager for rate limiter management
class ManagedRateLimiter:
    """Context manager for automatic rate limiter lifecycle management"""
    
    def __init__(self, config_type: str = "production", custom_config: Optional[Dict] = None):
        self.config_type = config_type
        self.custom_config = custom_config
        self.rate_limiter = None
    
    def __enter__(self) -> MultiProviderRateLimiter:
        if self.config_type == "production":
            self.rate_limiter = create_production_rate_limiter(self.custom_config)
        elif self.config_type == "development":
            self.rate_limiter = create_development_rate_limiter()
        elif self.config_type == "aggressive":
            self.rate_limiter = create_aggressive_rate_limiter()
        else:
            raise ValueError(f"Unknown config type: {self.config_type}")
        
        return self.rate_limiter
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        if self.rate_limiter:
            # Log final statistics
            status = self.rate_limiter.get_comprehensive_status()
            logger = get_legal_logger("rate_limiter_manager")
            logger.info(f"Rate limiter session complete: {status['success_rate']:.1%} success rate, "
                       f"{status['fallback_rate']:.1%} fallback rate")


# Utility functions for rate limit optimization

def estimate_tokens_from_content(content: str) -> int:
    """
    Estimate token count from content length.
    
    Args:
        content: Text content to estimate
        
    Returns:
        Estimated token count
    """
    # Rough estimation: 4 characters per token for English text
    return max(100, len(content) // 4)


def get_task_complexity(task_type: LegalTaskType, content_length: int) -> str:
    """
    Determine task complexity for optimal provider selection.
    
    Args:
        task_type: Legal task type
        content_length: Length of content to process
        
    Returns:
        Complexity level: "simple", "medium", or "complex"
    """
    if task_type == LegalTaskType.JUDICIAL_REASONING:
        return "complex"
    elif task_type == LegalTaskType.PRECEDENT_ANALYSIS:
        return "complex"
    elif task_type == LegalTaskType.OPINION_GENERATION:
        return "medium"
    elif content_length > 5000:
        return "complex"
    elif content_length > 1000:
        return "medium"
    else:
        return "simple"


def calculate_optimal_batch_size(provider: APIProvider, available_tokens: int) -> int:
    """
    Calculate optimal batch size for provider.
    
    Args:
        provider: API provider
        available_tokens: Currently available tokens
        
    Returns:
        Optimal batch size for efficient processing
    """
    # Provider-specific batch size optimization
    base_batch_sizes = {
        APIProvider.OPENAI: 10,
        APIProvider.ANTHROPIC: 8,
        APIProvider.GOOGLE: 12
    }
    
    base_size = base_batch_sizes.get(provider, 10)
    
    # Adjust based on available tokens
    if available_tokens < 5000:
        return max(1, base_size // 4)
    elif available_tokens < 15000:
        return max(1, base_size // 2)
    else:
        return base_size
