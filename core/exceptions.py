"""
Custom Exceptions for Multi-Task Legal Reward System

This module defines a comprehensive exception hierarchy for the legal reward system,
providing structured error handling with contextual information for debugging,
monitoring, and graceful degradation in production environments.

Exception Hierarchy:
- LegalRewardSystemError (base)
  ├── JurisdictionInferenceError
  ├── JudgeEvaluationError  
  ├── APIClientError
  │   ├── RateLimitExceededError
  │   ├── APIProviderError
  │   └── APIResponseError
  ├── CacheError
  ├── ConfigurationError
  ├── HybridEvaluationError
  ├── RoutingError
  ├── VERLIntegrationError
  └── SystemValidationError

Features:
- Structured error context for debugging
- Serialization support for logging systems
- Cost tracking for API-related errors
- Performance impact assessment
- Graceful degradation guidance
"""

import time
import traceback
from typing import Dict, Any, Optional, List, Union
from datetime import datetime
from dataclasses import dataclass, field


@dataclass
class ErrorContext:
    """
    Structured error context for comprehensive debugging information.
    
    Provides detailed context about where, when, and why an error occurred,
    along with system state information for debugging and monitoring.
    """
    component: str  # System component where error occurred
    operation: str  # Operation being performed when error occurred
    timestamp: float = field(default_factory=time.time)
    user_data_id: Optional[str] = None  # Associated data point ID
    task_type: Optional[str] = None  # Legal task type if applicable
    jurisdiction: Optional[str] = None  # Jurisdiction context if applicable
    api_provider: Optional[str] = None  # API provider if applicable
    cost_impact: float = 0.0  # Estimated cost impact of error
    performance_impact: float = 0.0  # Performance impact in seconds
    retry_count: int = 0  # Number of retries attempted
    additional_context: Dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert error context to dictionary for logging"""
        return {
            'component': self.component,
            'operation': self.operation,
            'timestamp': self.timestamp,
            'datetime': datetime.fromtimestamp(self.timestamp).isoformat(),
            'user_data_id': self.user_data_id,
            'task_type': self.task_type,
            'jurisdiction': self.jurisdiction,
            'api_provider': self.api_provider,
            'cost_impact': self.cost_impact,
            'performance_impact': self.performance_impact,
            'retry_count': self.retry_count,
            'additional_context': self.additional_context
        }
    
    def __str__(self) -> str:
        """Human-readable string representation"""
        parts = [f"{self.component}.{self.operation}"]
        if self.user_data_id:
            parts.append(f"data_id={self.user_data_id}")
        if self.task_type:
            parts.append(f"task={self.task_type}")
        if self.jurisdiction:
            parts.append(f"jurisdiction={self.jurisdiction}")
        if self.api_provider:
            parts.append(f"provider={self.api_provider}")
        if self.retry_count > 0:
            parts.append(f"retries={self.retry_count}")
        return " | ".join(parts)


class LegalRewardSystemError(Exception):
    """
    Base exception for the legal reward system.
    
    All system-specific exceptions inherit from this base class,
    providing consistent error handling, logging, and debugging
    capabilities across the entire system.
    
    Attributes:
        message: Human-readable error description
        error_context: Structured context information
        original_exception: Original exception if this wraps another error
        error_code: System-specific error code for categorization
        severity: Error severity level for monitoring
        is_recoverable: Whether the error can be recovered from
        suggested_action: Recommended action for handling this error
    """
    
    def __init__(self, 
                 message: str,
                 error_context: Optional[ErrorContext] = None,
                 original_exception: Optional[Exception] = None,
                 error_code: Optional[str] = None,
                 severity: str = "ERROR",
                 is_recoverable: bool = True,
                 suggested_action: Optional[str] = None):
        
        super().__init__(message)
        self.message = message
        self.error_context = error_context or ErrorContext("unknown", "unknown")
        self.original_exception = original_exception
        self.error_code = error_code or self.__class__.__name__
        self.severity = severity
        self.is_recoverable = is_recoverable
        self.suggested_action = suggested_action
        self.traceback_str = traceback.format_exc()
        
    def to_dict(self) -> Dict[str, Any]:
        """
        Convert exception to dictionary for structured logging.
        
        Returns:
            Dictionary containing all exception information for logging systems
        """
        return {
            'error_type': self.__class__.__name__,
            'message': self.message,
            'error_code': self.error_code,
            'severity': self.severity,
            'is_recoverable': self.is_recoverable,
            'suggested_action': self.suggested_action,
            'error_context': self.error_context.to_dict() if self.error_context else None,
            'original_exception': {
                'type': type(self.original_exception).__name__,
                'message': str(self.original_exception)
            } if self.original_exception else None,
            'traceback': self.traceback_str
        }
    
    def get_monitoring_tags(self) -> Dict[str, str]:
        """
        Get tags for monitoring systems (Datadog, Prometheus, etc.).
        
        Returns:
            Dictionary of key-value pairs for monitoring tag
        """
        tags = {
            'error_type': self.__class__.__name__,
            'error_code': self.error_code,
            'severity': self.severity,
            'component': self.error_context.component if self.error_context else "unknown",
            'recoverable': str(self.is_recoverable).lower()
        }
        
        if self.error_context:
            if self.error_context.task_type:
                tags['task_type'] = self.error_context.task_type
            if self.error_context.jurisdiction:
                tags['jurisdiction'] = self.error_context.jurisdiction
            if self.error_context.api_provider:
                tags['api_provider'] = self.error_context.api_provider
        
        return tags
    
    def __str__(self) -> str:
        """Enhanced string representation with context"""
        parts = [f"{self.__class__.__name__}: {self.message}"]
        if self.error_context:
            parts.append(f"Context: {self.error_context}")
        if self.suggested_action:
            parts.append(f"Suggested Action: {self.suggested_action}")
        return " | ".join(parts)


class JurisdictionInferenceError(LegalRewardSystemError):
    """
    Error in US jurisdiction inference process.
    
    Raised when the jurisdiction inference engine cannot determine
    the appropriate US jurisdiction for a legal query or when
    jurisdiction validation fails.
    """
    
    def __init__(self, message: str, 
                 query: Optional[str] = None,
                 inferred_jurisdiction: Optional[str] = None,
                 confidence_score: Optional[float] = None,
                 **kwargs):
        
        # Build enhanced error message
        enhanced_message = f"Jurisdiction inference failed: {message}"
        if inferred_jurisdiction:
            enhanced_message += f" (inferred: {inferred_jurisdiction}"
            if confidence_score is not None:
                enhanced_message += f", confidence: {confidence_score:.2f}"
            enhanced_message += ")"
        
        # Set up error context
        if 'error_context' not in kwargs:
            context = ErrorContext(
                component="jurisdiction_inference",
                operation="infer_jurisdiction",
                additional_context={
                    'query_length': len(query) if query else 0,
                    'inferred_jurisdiction': inferred_jurisdiction,
                    'confidence_score': confidence_score
                }
            )
            kwargs['error_context'] = context
        
        super().__init__(
            enhanced_message,
            suggested_action="Check query content for jurisdiction indicators or provide explicit jurisdiction",
            **kwargs
        )


class JudgeEvaluationError(LegalRewardSystemError):
    """
    Error in judge ensemble evaluation process.
    
    Raised when individual judges or judge ensembles fail to evaluate
    legal content, including API failures, timeout errors, or
    invalid evaluation results.
    """
    
    def __init__(self, message: str,
                 judge_name: Optional[str] = None,
                 ensemble_name: Optional[str] = None,
                 evaluation_stage: Optional[str] = None,
                 **kwargs):
        
        # Build enhanced error message
        enhanced_message = f"Judge evaluation failed: {message}"
        if judge_name:
            enhanced_message += f" (judge: {judge_name})"
        if ensemble_name:
            enhanced_message += f" (ensemble: {ensemble_name})"
        if evaluation_stage:
            enhanced_message += f" (stage: {evaluation_stage})"
        
        # Set up error context
        if 'error_context' not in kwargs:
            context = ErrorContext(
                component="judge_evaluation",
                operation=evaluation_stage or "evaluate",
                additional_context={
                    'judge_name': judge_name,
                    'ensemble_name': ensemble_name,
                    'evaluation_stage': evaluation_stage
                }
            )
            kwargs['error_context'] = context
        
        super().__init__(
            enhanced_message,
            suggested_action="Check judge configuration and API connectivity",
            **kwargs
        )


class APIClientError(LegalRewardSystemError):
    """
    Base error for API client operations.
    
    Covers general API client failures including configuration errors,
    authentication issues, and network connectivity problems.
    """
    
    def __init__(self, message: str,
                 provider: Optional[str] = None,
                 status_code: Optional[int] = None,
                 request_id: Optional[str] = None,
                 **kwargs):
        
        # Build enhanced error message
        enhanced_message = f"API client error: {message}"
        if provider:
            enhanced_message += f" (provider: {provider})"
        if status_code:
            enhanced_message += f" (status: {status_code})"
        if request_id:
            enhanced_message += f" (request_id: {request_id})"
        
        # Set up error context
        if 'error_context' not in kwargs:
            context = ErrorContext(
                component="api_client",
                operation="api_request",
                api_provider=provider,
                additional_context={
                    'status_code': status_code,
                    'request_id': request_id
                }
            )
            kwargs['error_context'] = context
        
        super().__init__(
            enhanced_message,
            suggested_action="Check API credentials and network connectivity",
            **kwargs
        )


class RateLimitExceededError(APIClientError):
    """
    API rate limit exceeded error.
    
    Raised when API providers return rate limit errors. Includes
    information about rate limits, reset times, and fallback options.
    """
    
    def __init__(self, message: str,
                 provider: Optional[str] = None,
                 limit_type: Optional[str] = None,
                 reset_time: Optional[float] = None,
                 requests_remaining: Optional[int] = None,
                 **kwargs):
        
        # Build enhanced error message
        enhanced_message = f"Rate limit exceeded: {message}"
        if provider:
            enhanced_message += f" (provider: {provider})"
        if limit_type:
            enhanced_message += f" (limit_type: {limit_type})"
        if reset_time:
            reset_datetime = datetime.fromtimestamp(reset_time)
            enhanced_message += f" (resets: {reset_datetime.strftime('%H:%M:%S')})"
        if requests_remaining is not None:
            enhanced_message += f" (remaining: {requests_remaining})"
        
        # Set up error context
        if 'error_context' not in kwargs:
            context = ErrorContext(
                component="api_client",
                operation="rate_limit_check",
                api_provider=provider,
                additional_context={
                    'limit_type': limit_type,
                    'reset_time': reset_time,
                    'requests_remaining': requests_remaining
                }
            )
            kwargs['error_context'] = context
        
        kwargs.pop('is_recoverable', None)
        kwargs.pop('suggested_action', None)

        super().__init__(
            enhanced_message,
            provider=provider,
            suggested_action="Wait for rate limit reset or use fallback provider",
            is_recoverable=True,
            **kwargs
        )


class APIProviderError(APIClientError):
    """
    Specific API provider error (OpenAI, Anthropic, Google).
    
    Raised when a specific API provider returns an error response
    or fails to provide a valid evaluation result.
    """
    
    def __init__(self, message: str,
                 provider: str,
                 error_type: Optional[str] = None,
                 **kwargs):
        
        enhanced_message = f"{provider} API error: {message}"
        if error_type:
            enhanced_message += f" (type: {error_type})"
        
        super().__init__(
            enhanced_message,
            provider=provider,
            suggested_action=f"Try fallback provider or check {provider} service status",
            **kwargs
        )


class APIResponseError(APIClientError):
    """
    Invalid API response error.
    
    Raised when API providers return responses that cannot be parsed
    or don't contain expected evaluation data.
    """
    
    def __init__(self, message: str,
                 provider: Optional[str] = None,
                 response_data: Optional[Dict] = None,
                 **kwargs):
        
        enhanced_message = f"Invalid API response: {message}"
        if provider:
            enhanced_message += f" (provider: {provider})"
        
        # Set up error context with response data
        if 'error_context' not in kwargs:
            context = ErrorContext(
                component="api_client",
                operation="parse_response",
                api_provider=provider,
                additional_context={
                    'response_data_present': response_data is not None,
                    'response_keys': list(response_data.keys()) if response_data else None
                }
            )
            kwargs['error_context'] = context
        
        super().__init__(
            enhanced_message,
            provider=provider,
            suggested_action="Check API response format and parsing logic",
            **kwargs
        )


class CacheError(LegalRewardSystemError):
    """
    Error in caching operations.
    
    Raised when cache operations fail, including cache misses,
    serialization errors, or cache storage issues.
    """
    
    def __init__(self, message: str,
                 cache_key: Optional[str] = None,
                 operation: Optional[str] = None,
                 **kwargs):
        
        enhanced_message = f"Cache error: {message}"
        if cache_key:
            enhanced_message += f" (key: {cache_key[:50]}...)" if len(cache_key) > 50 else f" (key: {cache_key})"
        if operation:
            enhanced_message += f" (operation: {operation})"
        
        # Set up error context
        if 'error_context' not in kwargs:
            context = ErrorContext(
                component="cache",
                operation=operation or "cache_operation",
                additional_context={
                    'cache_key_length': len(cache_key) if cache_key else 0,
                    'cache_operation': operation
                }
            )
            kwargs['error_context'] = context
        
        super().__init__(
            enhanced_message,
            suggested_action="Check cache configuration and storage availability",
            **kwargs
        )


class ConfigurationError(LegalRewardSystemError):
    """
    Error in system configuration.
    
    Raised when configuration files are invalid, missing required
    settings, or contain incompatible configuration values.
    """
    
    def __init__(self, message: str,
                 config_file: Optional[str] = None,
                 config_key: Optional[str] = None,
                 **kwargs):
        
        enhanced_message = f"Configuration error: {message}"
        if config_file:
            enhanced_message += f" (file: {config_file})"
        if config_key:
            enhanced_message += f" (key: {config_key})"
        
        # Set up error context
        if 'error_context' not in kwargs:
            context = ErrorContext(
                component="configuration",
                operation="load_config",
                additional_context={
                    'config_file': config_file,
                    'config_key': config_key
                }
            )
            kwargs['error_context'] = context
        
        kwargs.pop('is_recoverable', None)
        kwargs.pop('severity', None)
        kwargs.pop('suggested_action', None)
        
        super().__init__(
            enhanced_message,
            suggested_action="Check configuration file format and required settings",
            is_recoverable=False,  # Config errors usually require manual intervention
            severity="CRITICAL",
            **kwargs
        )


class HybridEvaluationError(LegalRewardSystemError):
    """
    Error in hybrid evaluation process.
    
    Raised when hybrid evaluation fails to combine specialized and
    general chat evaluations, or when evaluation weighting fails.
    """
    
    def __init__(self, message: str,
                 evaluation_stage: Optional[str] = None,
                 specialized_available: Optional[bool] = None,
                 general_chat_available: Optional[bool] = None,
                 **kwargs):
        
        enhanced_message = f"Hybrid evaluation error: {message}"
        if evaluation_stage:
            enhanced_message += f" (stage: {evaluation_stage})"
        
        # Set up error context
        if 'error_context' not in kwargs:
            context = ErrorContext(
                component="hybrid_evaluation",
                operation=evaluation_stage or "hybrid_evaluate",
                additional_context={
                    'specialized_available': specialized_available,
                    'general_chat_available': general_chat_available,
                    'evaluation_stage': evaluation_stage
                }
            )
            kwargs['error_context'] = context
        
        super().__init__(
            enhanced_message,
            suggested_action="Check judge ensemble availability and configuration",
            **kwargs
        )


class RoutingError(LegalRewardSystemError):
    """
    Error in reward routing process.
    
    Raised when the reward router cannot determine appropriate
    judge ensembles or fails to route evaluation requests.
    """
    
    def __init__(self, message: str,
                 task_type: Optional[str] = None,
                 available_ensembles: Optional[List[str]] = None,
                 **kwargs):
        
        enhanced_message = f"Routing error: {message}"
        if task_type:
            enhanced_message += f" (task_type: {task_type})"
        if available_ensembles:
            enhanced_message += f" (available: {', '.join(available_ensembles)})"
        
        # Set up error context
        if 'error_context' not in kwargs:
            context = ErrorContext(
                component="reward_router",
                operation="route_evaluation",
                task_type=task_type,
                additional_context={
                    'available_ensembles': available_ensembles,
                    'ensemble_count': len(available_ensembles) if available_ensembles else 0
                }
            )
            kwargs['error_context'] = context
        
        super().__init__(
            enhanced_message,
            suggested_action="Check judge ensemble registration and task type configuration",
            **kwargs
        )


class VERLIntegrationError(LegalRewardSystemError):
    """
    Error in VERL integration process.
    
    Raised when VERL data format conversion fails or when the
    system cannot provide valid reward scores to VERL.
    """
    
    def __init__(self, message: str,
                 data_format: Optional[str] = None,
                 batch_size: Optional[int] = None,
                 **kwargs):
        
        enhanced_message = f"VERL integration error: {message}"
        if data_format:
            enhanced_message += f" (format: {data_format})"
        if batch_size:
            enhanced_message += f" (batch_size: {batch_size})"
        
        # Set up error context
        if 'error_context' not in kwargs:
            context = ErrorContext(
                component="verl_integration",
                operation="convert_data",
                additional_context={
                    'data_format': data_format,
                    'batch_size': batch_size
                }
            )
            kwargs['error_context'] = context
        
        super().__init__(
            enhanced_message,
            suggested_action="Check VERL data format and conversion logic",
            **kwargs
        )


class SystemValidationError(LegalRewardSystemError):
    """
    Error in system validation process.
    
    Raised when system validation checks fail, indicating that
    the system is not properly configured or ready for operation.
    """
    
    def __init__(self, message: str,
                 validation_type: Optional[str] = None,
                 failed_checks: Optional[List[str]] = None,
                 **kwargs):
        
        enhanced_message = f"System validation error: {message}"
        if validation_type:
            enhanced_message += f" (type: {validation_type})"
        if failed_checks:
            enhanced_message += f" (failed: {', '.join(failed_checks)})"
        
        # Set up error context
        if 'error_context' not in kwargs:
            context = ErrorContext(
                component="system_validation",
                operation="validate_system",
                additional_context={
                    'validation_type': validation_type,
                    'failed_checks': failed_checks,
                    'failure_count': len(failed_checks) if failed_checks else 0
                }
            )
            kwargs['error_context'] = context
        
        kwargs.pop('is_recoverable', None)
        kwargs.pop('severity', None)
        kwargs.pop('suggested_action', None)
        
        super().__init__(
            enhanced_message,
            suggested_action="Complete system setup and configuration",
            is_recoverable=False,
            severity="CRITICAL",
            **kwargs
        )


# Utility functions for exception handling

def wrap_exception(original_exception: Exception, 
                   system_error_class: type,
                   message: str,
                   **kwargs) -> LegalRewardSystemError:
    """
    Wrap an external exception in a system-specific exception.
    
    Args:
        original_exception: The original exception to wrap
        system_error_class: Legal reward system exception class to use
        message: Additional context message
        **kwargs: Additional arguments for the system exception
        
    Returns:
        System-specific exception wrapping the original
    """
    return system_error_class(
        message,
        original_exception=original_exception,
        **kwargs
    )


def create_error_context(component: str, operation: str, **kwargs) -> ErrorContext:
    """
    Create an error context with common fields populated.
    
    Args:
        component: System component name
        operation: Operation being performed
        **kwargs: Additional context fields
        
    Returns:
        ErrorContext instance
    """
    return ErrorContext(
        component=component,
        operation=operation,
        **kwargs
    )


def handle_api_error(exception: Exception, 
                     provider: str, 
                     operation: str) -> APIClientError:
    """
    Convert generic API errors to structured API client errors.
    
    Args:
        exception: Original API exception
        provider: API provider name
        operation: Operation that failed
        
    Returns:
        Structured APIClientError
    """
    # Check for rate limit errors
    if "rate limit" in str(exception).lower():
        return RateLimitExceededError(
            str(exception),
            provider=provider,
            error_context=create_error_context("api_client", operation, api_provider=provider),
            original_exception=exception
        )
    
    # Check for authentication errors
    elif "auth" in str(exception).lower() or "401" in str(exception):
        return APIProviderError(
            str(exception),
            provider=provider,
            error_type="authentication",
            error_context=create_error_context("api_client", operation, api_provider=provider),
            original_exception=exception
        )
    
    # Generic API error
    else:
        return APIClientError(
            str(exception),
            provider=provider,
            error_context=create_error_context("api_client", operation, api_provider=provider),
            original_exception=exception
        )


def log_exception(exception: LegalRewardSystemError, logger) -> None:
    """
    Log an exception with structured information.
    
    Args:
        exception: Legal reward system exception to log
        logger: Logger instance to use
    """
    if exception.severity == "CRITICAL":
        logger.critical(exception.message, extra=exception.to_dict())
    elif exception.severity == "ERROR":
        logger.error(exception.message, extra=exception.to_dict())
    else:
        logger.warning(exception.message, extra=exception.to_dict())


def get_recovery_suggestions(exception: LegalRewardSystemError) -> List[str]:
    """
    Get recovery suggestions for an exception.
    
    Args:
        exception: Exception to analyze
        
    Returns:
        List of recovery suggestions
    """
    suggestions = []
    
    if exception.suggested_action:
        suggestions.append(exception.suggested_action)
    
    # Add general suggestions based on exception type
    if isinstance(exception, APIClientError):
        suggestions.extend([
            "Check API key configuration",
            "Verify network connectivity",
            "Try using a fallback API provider"
        ])
    
    elif isinstance(exception, CacheError):
        suggestions.extend([
            "Clear cache and try again",
            "Check cache directory permissions",
            "Verify cache configuration"
        ])
    
    elif isinstance(exception, ConfigurationError):
        suggestions.extend([
            "Review configuration file format",
            "Check for missing required settings",
            "Validate configuration against schema"
        ])
    
    return suggestions

def clean_exception_kwargs(**kwargs) -> dict:
    """
    Clean kwargs to avoid parameter conflicts in exception constructors.
    
    Removes common parameters that are often set explicitly to avoid
    'multiple values for keyword argument' errors.
    """
    cleaned = kwargs.copy()
    conflict_params = [
        'is_recoverable', 'severity', 'suggested_action', 
        'error_code', 'message'
    ]
    for param in conflict_params:
        cleaned.pop(param, None)
    return cleaned

# Exception type mapping for easy access
EXCEPTION_TYPES = {
    'base': LegalRewardSystemError,
    'jurisdiction': JurisdictionInferenceError,
    'judge': JudgeEvaluationError,
    'api': APIClientError,
    'rate_limit': RateLimitExceededError,
    'api_provider': APIProviderError,
    'api_response': APIResponseError,
    'cache': CacheError,
    'config': ConfigurationError,
    'hybrid': HybridEvaluationError,
    'routing': RoutingError,
    'verl': VERLIntegrationError,
    'validation': SystemValidationError
}


# Common error codes for monitoring and alerting
ERROR_CODES = {
    'JURISDICTION_INFERENCE_FAILED': 'JIF001',
    'JUDGE_EVALUATION_TIMEOUT': 'JET001',
    'API_RATE_LIMIT_EXCEEDED': 'ARL001',
    'API_AUTHENTICATION_FAILED': 'AAF001',
    'CACHE_STORAGE_FULL': 'CSF001',
    'CONFIG_MISSING_REQUIRED': 'CMR001',
    'HYBRID_EVALUATION_FAILED': 'HEF001',
    'ROUTING_NO_ENSEMBLE': 'RNE001',
    'VERL_DATA_FORMAT_INVALID': 'VDF001',
    'SYSTEM_VALIDATION_FAILED': 'SVF001'
}