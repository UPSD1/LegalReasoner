"""
Enhanced Logging System for Multi-Task Legal Reward System

This module provides comprehensive logging capabilities specifically designed for
the legal reward system, including:

- Cost tracking across all API providers
- Performance monitoring for evaluation pipelines  
- Structured logging with JSON formatting
- Integration with exception framework
- Jurisdiction inference logging
- Cache performance tracking
- Production-ready monitoring support

Key Features:
- Multi-provider API cost tracking with budget alerts
- Performance metrics collection and analysis
- Structured JSON logging for production monitoring
- Integration with legal system exceptions
- Jurisdiction compliance logging
- Cache hit/miss tracking with cost savings
- Configurable log levels and handlers
- Async logging support for high-throughput scenarios
"""

import logging
import logging.handlers
import json
import time
import threading
from typing import Dict, Any, Optional, List, Union
from datetime import datetime, timedelta
from dataclasses import dataclass, field
from collections import defaultdict, deque
from pathlib import Path
import asyncio
from concurrent.futures import ThreadPoolExecutor

# Import core components
from ..core import (
    LegalRewardSystemError, APIProvider, LegalTaskType, 
    USJurisdiction, LogLevel, ERROR_CODES
)


@dataclass
class APIRequestLog:
    """Log entry for API requests with cost and performance tracking"""
    timestamp: float
    provider: str
    operation: str
    cost: float
    tokens_used: int
    response_time: float
    success: bool
    error_message: Optional[str] = None
    request_id: Optional[str] = None
    task_type: Optional[str] = None
    jurisdiction: Optional[str] = None
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization"""
        return {
            'timestamp': self.timestamp,
            'datetime': datetime.fromtimestamp(self.timestamp).isoformat(),
            'provider': self.provider,
            'operation': self.operation,
            'cost': self.cost,
            'tokens_used': self.tokens_used,
            'response_time': self.response_time,
            'success': self.success,
            'error_message': self.error_message,
            'request_id': self.request_id,
            'task_type': self.task_type,
            'jurisdiction': self.jurisdiction
        }


@dataclass
class PerformanceMetric:
    """Performance metric tracking entry"""
    component: str
    operation: str
    timestamp: float
    duration: float
    success: bool
    additional_data: Dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization"""
        return {
            'component': self.component,
            'operation': self.operation,
            'timestamp': self.timestamp,
            'datetime': datetime.fromtimestamp(self.timestamp).isoformat(),
            'duration': self.duration,
            'success': self.success,
            'additional_data': self.additional_data
        }


class CostTracker:
    """
    Track API costs across providers with budget management and alerting.
    
    Features:
    - Real-time cost tracking per provider
    - Daily/monthly budget monitoring
    - Cost projection and alerts
    - Historical cost analysis
    - Cost optimization recommendations
    """
    
    def __init__(self, monthly_budget: float = 1000.0):
        self.monthly_budget = monthly_budget
        self.api_requests: List[APIRequestLog] = []
        self.provider_costs = defaultdict(float)
        self.daily_costs = defaultdict(float)
        self.monthly_cost = 0.0
        self.lock = threading.Lock()
        
        # Budget alerting
        self.budget_alerts_sent = set()
        self.alert_thresholds = [0.5, 0.75, 0.9, 0.95]  # 50%, 75%, 90%, 95%
        
    def track_api_request(self, log_entry: APIRequestLog):
        """Track an API request for cost analysis"""
        with self.lock:
            self.api_requests.append(log_entry)
            self.provider_costs[log_entry.provider] += log_entry.cost
            
            # Track daily costs
            date_key = datetime.fromtimestamp(log_entry.timestamp).strftime('%Y-%m-%d')
            self.daily_costs[date_key] += log_entry.cost
            
            # Update monthly cost
            current_month = datetime.now().strftime('%Y-%m')
            request_month = datetime.fromtimestamp(log_entry.timestamp).strftime('%Y-%m')
            if request_month == current_month:
                self.monthly_cost += log_entry.cost
            
            # Check budget alerts
            self._check_budget_alerts()
    
    def _check_budget_alerts(self):
        """Check if budget alerts should be sent"""
        budget_utilization = self.monthly_cost / self.monthly_budget
        
        for threshold in self.alert_thresholds:
            if budget_utilization >= threshold and threshold not in self.budget_alerts_sent:
                self.budget_alerts_sent.add(threshold)
    
    def get_cost_summary(self) -> Dict[str, Any]:
        """Get comprehensive cost summary"""
        with self.lock:
            total_requests = len(self.api_requests)
            successful_requests = sum(1 for req in self.api_requests if req.success)
            
            # Calculate averages
            avg_cost_per_request = self.monthly_cost / max(total_requests, 1)
            avg_tokens_per_request = sum(req.tokens_used for req in self.api_requests) / max(total_requests, 1)
            
            # Cost by provider
            provider_breakdown = dict(self.provider_costs)
            
            # Recent trend (last 7 days)
            recent_costs = []
            cutoff_time = time.time() - (7 * 24 * 3600)  # 7 days ago
            for req in self.api_requests:
                if req.timestamp >= cutoff_time:
                    recent_costs.append(req.cost)
            
            daily_avg_recent = sum(recent_costs) / 7 if recent_costs else 0
            
            return {
                'monthly_budget': self.monthly_budget,
                'monthly_cost': self.monthly_cost,
                'budget_utilization': self.monthly_cost / self.monthly_budget,
                'budget_remaining': self.monthly_budget - self.monthly_cost,
                'total_requests': total_requests,
                'successful_requests': successful_requests,
                'success_rate': successful_requests / max(total_requests, 1),
                'avg_cost_per_request': avg_cost_per_request,
                'avg_tokens_per_request': avg_tokens_per_request,
                'cost_by_provider': provider_breakdown,
                'daily_avg_recent': daily_avg_recent,
                'projected_monthly_cost': daily_avg_recent * 30,
                'alerts_triggered': len(self.budget_alerts_sent),
                'last_updated': datetime.now().isoformat()
            }
    
    def get_cost_optimization_suggestions(self) -> List[str]:
        """Get cost optimization suggestions based on usage patterns"""
        suggestions = []
        cost_summary = self.get_cost_summary()
        
        # High budget utilization
        if cost_summary['budget_utilization'] > 0.8:
            suggestions.append("Consider implementing more aggressive caching to reduce API calls")
            suggestions.append("Review rate limiting settings to optimize provider usage")
        
        # Provider cost analysis
        provider_costs = cost_summary['cost_by_provider']
        if provider_costs:
            most_expensive = max(provider_costs.items(), key=lambda x: x[1])
            suggestions.append(f"Consider using cheaper alternatives to {most_expensive[0]} for simple tasks")
        
        # Success rate issues
        if cost_summary['success_rate'] < 0.95:
            suggestions.append("Investigate API failures to avoid wasted costs on failed requests")
        
        return suggestions


class PerformanceTracker:
    """
    Track system performance metrics with analysis and optimization insights.
    
    Features:
    - Component-level performance tracking
    - Operation timing and success rates
    - Performance trend analysis
    - Bottleneck identification
    - Performance optimization recommendations
    """
    
    def __init__(self, max_metrics: int = 10000):
        self.max_metrics = max_metrics
        self.metrics: deque[PerformanceMetric] = deque(maxlen=max_metrics)
        self.component_stats = defaultdict(lambda: {
            'total_operations': 0,
            'total_time': 0.0,
            'successful_operations': 0,
            'failed_operations': 0
        })
        self.lock = threading.Lock()
    
    def track_operation(self, component: str, operation: str, duration: float, 
                       success: bool, **additional_data):
        """Track a system operation for performance analysis"""
        metric = PerformanceMetric(
            component=component,
            operation=operation,
            timestamp=time.time(),
            duration=duration,
            success=success,
            additional_data=additional_data
        )
        
        with self.lock:
            self.metrics.append(metric)
            
            # Update component stats
            stats = self.component_stats[component]
            stats['total_operations'] += 1
            stats['total_time'] += duration
            if success:
                stats['successful_operations'] += 1
            else:
                stats['failed_operations'] += 1
    
    def get_performance_summary(self) -> Dict[str, Any]:
        """Get comprehensive performance summary"""
        with self.lock:
            if not self.metrics:
                return {'total_operations': 0, 'message': 'No performance data available'}
            
            # Overall stats
            total_operations = len(self.metrics)
            successful_ops = sum(1 for m in self.metrics if m.success)
            total_time = sum(m.duration for m in self.metrics)
            avg_duration = total_time / total_operations
            
            # Component breakdown
            component_breakdown = {}
            for component, stats in self.component_stats.items():
                component_breakdown[component] = {
                    'total_operations': stats['total_operations'],
                    'success_rate': stats['successful_operations'] / max(stats['total_operations'], 1),
                    'avg_duration': stats['total_time'] / max(stats['total_operations'], 1),
                    'total_time': stats['total_time']
                }
            
            # Recent performance (last hour)
            cutoff_time = time.time() - 3600  # 1 hour ago
            recent_metrics = [m for m in self.metrics if m.timestamp >= cutoff_time]
            recent_avg_duration = sum(m.duration for m in recent_metrics) / max(len(recent_metrics), 1)
            
            # Identify slowest operations
            sorted_metrics = sorted(self.metrics, key=lambda x: x.duration, reverse=True)
            slowest_operations = [
                {
                    'component': m.component,
                    'operation': m.operation,
                    'duration': m.duration,
                    'timestamp': m.timestamp
                }
                for m in sorted_metrics[:5]
            ]
            
            return {
                'total_operations': total_operations,
                'successful_operations': successful_ops,
                'success_rate': successful_ops / total_operations,
                'avg_duration': avg_duration,
                'recent_avg_duration': recent_avg_duration,
                'component_breakdown': component_breakdown,
                'slowest_operations': slowest_operations,
                'last_updated': datetime.now().isoformat()
            }
    
    def get_bottlenecks(self) -> List[Dict[str, Any]]:
        """Identify performance bottlenecks"""
        performance_summary = self.get_performance_summary()
        bottlenecks = []
        
        # Slow components
        for component, stats in performance_summary['component_breakdown'].items():
            if stats['avg_duration'] > 2.0:  # Slower than 2 seconds
                bottlenecks.append({
                    'type': 'slow_component',
                    'component': component,
                    'avg_duration': stats['avg_duration'],
                    'suggestion': f"Optimize {component} performance - average duration {stats['avg_duration']:.2f}s"
                })
            
            # Low success rates
            if stats['success_rate'] < 0.9:
                bottlenecks.append({
                    'type': 'low_success_rate',
                    'component': component,
                    'success_rate': stats['success_rate'],
                    'suggestion': f"Investigate {component} failures - success rate {stats['success_rate']:.1%}"
                })
        
        return bottlenecks


class LegalRewardJSONFormatter(logging.Formatter):
    """
    Custom JSON formatter for structured logging with legal system context.
    
    Provides structured JSON output with legal-specific fields including
    task types, jurisdictions, cost information, and performance metrics.
    """
    
    def format(self, record: logging.LogRecord) -> str:
        """Format log record as structured JSON"""
        log_data = {
            'timestamp': record.created,
            'datetime': datetime.fromtimestamp(record.created).isoformat(),
            'level': record.levelname,
            'logger': record.name,
            'message': record.getMessage(),
            'module': record.module,
            'function': record.funcName,
            'line': record.lineno
        }
        
        # Add exception information if present
        if record.exc_info:
            log_data['exception'] = {
                'type': record.exc_info[0].__name__,
                'message': str(record.exc_info[1]),
                'traceback': self.formatException(record.exc_info)
            }
        
        # Add legal system specific fields from extra data
        legal_fields = [
            'task_type', 'jurisdiction', 'legal_domain', 'data_id',
            'api_provider', 'cost', 'tokens_used', 'response_time',
            'cache_hit', 'cache_savings', 'evaluation_score', 'confidence',
            'jurisdiction_inferred', 'ensemble_type', 'hybrid_evaluation'
        ]
        
        for field in legal_fields:
            if hasattr(record, field):
                log_data[field] = getattr(record, field)
        
        # Add any additional extra data
        if hasattr(record, 'extra_data'):
            log_data['extra_data'] = record.extra_data
        
        return json.dumps(log_data, default=str, ensure_ascii=False)


class LegalRewardLogger:
    """
    Enhanced logger for the legal reward system with comprehensive tracking.
    
    Features:
    - Cost tracking across all API providers
    - Performance monitoring for all operations
    - Structured JSON logging for production
    - Integration with legal system exceptions
    - Jurisdiction inference logging
    - Cache performance tracking
    - Configurable output (console, file, remote)
    - Async logging support for high throughput
    """
    
    def __init__(self, name: str, config: Optional[Dict[str, Any]] = None):
        self.name = name
        self.config = config or self._get_default_config()
        
        # Initialize components
        self.logger = logging.getLogger(name)
        self.cost_tracker = CostTracker(self.config.get('monthly_budget', 1000.0))
        self.performance_tracker = PerformanceTracker()
        
        # Setup logging configuration
        self._setup_logging()
        
        # Async logging support
        self.async_executor = ThreadPoolExecutor(max_workers=2, thread_name_prefix="legal_logger")
        
        # Cache performance tracking
        self.cache_hits = 0
        self.cache_misses = 0
        self.total_cache_savings = 0.0
    
    def _get_default_config(self) -> Dict[str, Any]:
        """Get default logging configuration"""
        return {
            'level': 'INFO',
            'console_output': True,
            'file_output': True,
            'json_format': True,
            'log_api_costs': True,
            'log_cache_performance': True,
            'log_jurisdiction_inference': True,
            'monthly_budget': 1000.0,
            'log_file': 'legal_reward_system.log',
            'max_file_size': 100 * 1024 * 1024,  # 100MB
            'backup_count': 5
        }
    
    def _setup_logging(self):
        """Setup logging handlers and formatters"""
        self.logger.setLevel(getattr(logging, self.config['level']))
        
        # Clear existing handlers
        self.logger.handlers.clear()
        
        # Console handler
        if self.config.get('console_output', True):
            console_handler = logging.StreamHandler()
            if self.config.get('json_format', True):
                console_handler.setFormatter(LegalRewardJSONFormatter())
            else:
                console_handler.setFormatter(
                    logging.Formatter(
                        '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
                    )
                )
            self.logger.addHandler(console_handler)
        
        # File handler
        if self.config.get('file_output', True):
            log_file = Path(self.config.get('log_file', 'legal_reward_system.log'))
            log_file.parent.mkdir(parents=True, exist_ok=True)
            
            file_handler = logging.handlers.RotatingFileHandler(
                log_file,
                maxBytes=self.config.get('max_file_size', 100 * 1024 * 1024),
                backupCount=self.config.get('backup_count', 5)
            )
            
            if self.config.get('json_format', True):
                file_handler.setFormatter(LegalRewardJSONFormatter())
            else:
                file_handler.setFormatter(
                    logging.Formatter(
                        '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
                    )
                )
            
            self.logger.addHandler(file_handler)
    
    def log_api_request(self, provider: str, operation: str, cost: float, 
                       tokens_used: int, response_time: float, success: bool,
                       error_message: Optional[str] = None, 
                       task_type: Optional[str] = None,
                       jurisdiction: Optional[str] = None,
                       request_id: Optional[str] = None):
        """Log API request with comprehensive cost tracking"""
        
        # Create API request log
        api_log = APIRequestLog(
            timestamp=time.time(),
            provider=provider,
            operation=operation,
            cost=cost,
            tokens_used=tokens_used,
            response_time=response_time,
            success=success,
            error_message=error_message,
            request_id=request_id,
            task_type=task_type,
            jurisdiction=jurisdiction
        )
        
        # Track for cost analysis
        self.cost_tracker.track_api_request(api_log)
        
        # Log the request
        log_level = logging.INFO if success else logging.ERROR
        self.logger.log(
            log_level,
            f"API request to {provider}: {operation} ({'success' if success else 'failed'})",
            extra={
                'api_provider': provider,
                'operation': operation,
                'cost': cost,
                'tokens_used': tokens_used,
                'response_time': response_time,
                'success': success,
                'error_message': error_message,
                'task_type': task_type,
                'jurisdiction': jurisdiction,
                'request_id': request_id
            }
        )
    
    def log_cache_hit(self, cache_key: str, estimated_savings: float, 
                     operation: str = "cache_lookup"):
        """Log cache hit with cost savings tracking"""
        self.cache_hits += 1
        self.total_cache_savings += estimated_savings
        
        if self.config.get('log_cache_performance', True):
            self.logger.info(
                f"Cache hit: {operation} (savings: ${estimated_savings:.4f})",
                extra={
                    'cache_hit': True,
                    'cache_key': cache_key[:50] + "..." if len(cache_key) > 50 else cache_key,
                    'cache_savings': estimated_savings,
                    'operation': operation,
                    'total_cache_savings': self.total_cache_savings
                }
            )
    
    def log_cache_miss(self, cache_key: str, operation: str = "cache_lookup"):
        """Log cache miss for performance analysis"""
        self.cache_misses += 1
        
        if self.config.get('log_cache_performance', True):
            self.logger.debug(
                f"Cache miss: {operation}",
                extra={
                    'cache_hit': False,
                    'cache_key': cache_key[:50] + "..." if len(cache_key) > 50 else cache_key,
                    'operation': operation
                }
            )
    
    def log_evaluation_result(self, data_id: str, task_type: str, 
                            ensemble_score: float, confidence: float,
                            evaluation_time: float, evaluation_method: str,
                            jurisdiction: Optional[str] = None,
                            ensemble_type: Optional[str] = None):
        """Log evaluation result with performance metrics"""
        
        # Track performance
        self.performance_tracker.track_operation(
            component="evaluation",
            operation=f"{task_type}_evaluation",
            duration=evaluation_time,
            success=True,
            ensemble_score=ensemble_score,
            confidence=confidence,
            evaluation_method=evaluation_method
        )
        
        self.logger.info(
            f"Evaluation completed: {task_type} (score: {ensemble_score:.2f}, confidence: {confidence:.2f})",
            extra={
                'data_id': data_id,
                'task_type': task_type,
                'evaluation_score': ensemble_score,
                'confidence': confidence,
                'evaluation_time': evaluation_time,
                'evaluation_method': evaluation_method,
                'jurisdiction': jurisdiction,
                'ensemble_type': ensemble_type
            }
        )
    
    def log_jurisdiction_inference(self, query: str, inferred_jurisdiction: str, 
                                 confidence: float, reasoning: str,
                                 should_ask_user: bool = False,
                                 requires_disclaimer: bool = False):
        """Log jurisdiction inference results"""
        
        if self.config.get('log_jurisdiction_inference', True):
            self.logger.info(
                f"Jurisdiction inferred: {inferred_jurisdiction} (confidence: {confidence:.2f})",
                extra={
                    'query_length': len(query),
                    'jurisdiction': inferred_jurisdiction,
                    'jurisdiction_inferred': True,
                    'confidence': confidence,
                    'reasoning': reasoning,
                    'should_ask_user': should_ask_user,
                    'requires_disclaimer': requires_disclaimer
                }
            )
    
    def log_hybrid_evaluation(self, data_id: str, specialized_score: Optional[float],
                            general_chat_score: Optional[float], 
                            jurisdiction_compliance_score: float,
                            hybrid_raw_score: float, evaluation_method: str,
                            total_time: float):
        """Log hybrid evaluation results"""
        
        self.logger.info(
            f"Hybrid evaluation: {evaluation_method} (final: {hybrid_raw_score:.2f})",
            extra={
                'data_id': data_id,
                'hybrid_evaluation': True,
                'specialized_score': specialized_score,
                'general_chat_score': general_chat_score,
                'jurisdiction_compliance_score': jurisdiction_compliance_score,
                'hybrid_raw_score': hybrid_raw_score,
                'evaluation_method': evaluation_method,
                'evaluation_time': total_time
            }
        )
    
    def log_exception(self, exception: LegalRewardSystemError, 
                     context: Optional[Dict[str, Any]] = None):
        """Log legal system exception with structured information"""
        
        log_data = exception.to_dict()
        if context:
            log_data['additional_context'] = context
        
        # Log at appropriate level based on exception severity
        log_level = getattr(logging, exception.severity, logging.ERROR)
        
        self.logger.log(
            log_level,
            f"{exception.__class__.__name__}: {exception.message}",
            extra=log_data,
            exc_info=(type(exception), exception, exception.__traceback__)
        )
        
        # Track performance impact
        if exception.error_context:
            self.performance_tracker.track_operation(
                component=exception.error_context.component,
                operation=exception.error_context.operation,
                duration=exception.error_context.performance_impact,
                success=False,
                error_type=exception.__class__.__name__,
                error_code=exception.error_code
            )
    
    def log_system_startup(self, component: str, config: Dict[str, Any]):
        """Log system component startup"""
        self.logger.info(
            f"System component started: {component}",
            extra={
                'component': component,
                'startup': True,
                'config': config
            }
        )
    
    def log_system_shutdown(self, component: str, stats: Optional[Dict[str, Any]] = None):
        """Log system component shutdown with final stats"""
        self.logger.info(
            f"System component shutdown: {component}",
            extra={
                'component': component,
                'shutdown': True,
                'final_stats': stats
            }
        )
    
    def get_cost_summary(self) -> Dict[str, Any]:
        """Get comprehensive cost tracking summary"""
        return self.cost_tracker.get_cost_summary()
    
    def get_performance_summary(self) -> Dict[str, Any]:
        """Get comprehensive performance tracking summary"""
        performance_data = self.performance_tracker.get_performance_summary()
        
        # Add cache performance
        total_cache_requests = self.cache_hits + self.cache_misses
        cache_hit_rate = self.cache_hits / max(total_cache_requests, 1)
        
        performance_data.update({
            'cache_performance': {
                'total_requests': total_cache_requests,
                'cache_hits': self.cache_hits,
                'cache_misses': self.cache_misses,
                'hit_rate': cache_hit_rate,
                'total_savings': self.total_cache_savings
            }
        })
        
        return performance_data
    
    def get_monitoring_dashboard_data(self) -> Dict[str, Any]:
        """Get comprehensive data for monitoring dashboards"""
        return {
            'cost_tracking': self.get_cost_summary(),
            'performance_metrics': self.get_performance_summary(),
            'cost_optimization_suggestions': self.cost_tracker.get_cost_optimization_suggestions(),
            'performance_bottlenecks': self.performance_tracker.get_bottlenecks(),
            'logger_info': {
                'name': self.name,
                'level': self.logger.level,
                'handlers': len(self.logger.handlers),
                'config': self.config
            }
        }
    
    async def log_async(self, level: int, message: str, **kwargs):
        """Async logging for high-throughput scenarios"""
        loop = asyncio.get_event_loop()
        await loop.run_in_executor(
            self.async_executor,
            lambda: self.logger.log(level, message, extra=kwargs)
        )
    
    def cleanup(self):
        """Cleanup resources"""
        self.async_executor.shutdown(wait=True)
        for handler in self.logger.handlers:
            handler.close()
        self.logger.handlers.clear()
        
    def debug(self, message: str, **extra):
        """Standard debug logging - delegates to Python logger"""
        self.logger.debug(message, extra=extra)
    
    def info(self, message: str, **extra):
        """Standard info logging - delegates to Python logger"""
        self.logger.info(message, extra=extra)
    
    def warning(self, message: str, **extra):
        """Standard warning logging - delegates to Python logger"""
        self.logger.warning(message, extra=extra)
    
    def error(self, message: str, **extra):
        """Standard error logging - delegates to Python logger"""
        self.logger.error(message, extra=extra)
    
    def critical(self, message: str, **extra):
        """Standard critical logging - delegates to Python logger"""
        self.logger.critical(message, extra=extra)
    
    def exception(self, message: str, **extra):
        """Standard exception logging with traceback - delegates to Python logger"""
        self.logger.exception(message, extra=extra)


# Global logger registry for reuse across the system
_logger_registry: Dict[str, LegalRewardLogger] = {}
_registry_lock = threading.Lock()


def get_legal_logger(name: str, config: Optional[Dict[str, Any]] = None) -> LegalRewardLogger:
    """
    Get or create a legal reward system logger.
    
    Args:
        name: Logger name (usually module/component name)
        config: Optional configuration dictionary
        
    Returns:
        LegalRewardLogger instance
    """
    with _registry_lock:
        if name not in _logger_registry:
            _logger_registry[name] = LegalRewardLogger(name, config)
        return _logger_registry[name]


def setup_system_logging(config: Dict[str, Any]) -> Dict[str, LegalRewardLogger]:
    """
    Setup logging for all system components.
    
    Args:
        config: System-wide logging configuration
        
    Returns:
        Dictionary of loggers for each component
    """
    components = [
        'reward_router', 'judge_evaluation', 'api_client', 'cache',
        'jurisdiction_inference', 'hybrid_evaluation', 'verl_integration'
    ]
    
    loggers = {}
    for component in components:
        loggers[component] = get_legal_logger(component, config)
    
    return loggers


def get_system_logging_summary() -> Dict[str, Any]:
    """Get summary of all system loggers"""
    with _registry_lock:
        return {
            'total_loggers': len(_logger_registry),
            'logger_names': list(_logger_registry.keys()),
            'aggregate_cost_summary': {
                'total_monthly_cost': sum(
                    logger.get_cost_summary()['monthly_cost'] 
                    for logger in _logger_registry.values()
                ),
                'total_requests': sum(
                    logger.get_cost_summary()['total_requests']
                    for logger in _logger_registry.values()
                )
            },
            'aggregate_cache_performance': {
                'total_cache_savings': sum(
                    logger.total_cache_savings 
                    for logger in _logger_registry.values()
                ),
                'total_cache_hits': sum(
                    logger.cache_hits 
                    for logger in _logger_registry.values()
                )
            }
        }


# Performance monitoring context manager
class LoggedOperation:
    """Context manager for automatic operation logging"""
    
    def __init__(self, logger: LegalRewardLogger, component: str, operation: str, **extra):
        self.logger = logger
        self.component = component
        self.operation = operation
        self.extra = extra
        self.start_time = None
    
    def __enter__(self):
        self.start_time = time.time()
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        duration = time.time() - self.start_time
        success = exc_type is None
        
        self.logger.performance_tracker.track_operation(
            component=self.component,
            operation=self.operation,
            duration=duration,
            success=success,
            **self.extra
        )
        
        if not success and isinstance(exc_val, LegalRewardSystemError):
            self.logger.log_exception(exc_val)
