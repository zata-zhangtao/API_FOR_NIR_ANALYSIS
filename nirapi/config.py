"""
Configuration management for nirapi package.

This module handles database connections and other configuration settings
using environment variables for security.
"""

import os
from typing import Optional

__all__ = [
    'DatabaseConfig',
    'EmailConfig',
    'db_config',
    'email_config',
    'get_database_connection_params'
]


class DatabaseConfig:
    """Database configuration with environment variable support."""
    
    def __init__(self):
        # Database connection settings with environment variable fallbacks
        self.host = os.getenv('NIRAPI_DB_HOST', 'localhost')
        self.port = int(os.getenv('NIRAPI_DB_PORT', '3306'))
        self.user = os.getenv('NIRAPI_DB_USER', 'root')
        self.password = os.getenv('NIRAPI_DB_PASSWORD', '')
        self.charset = os.getenv('NIRAPI_DB_CHARSET', 'utf8mb4')
        
        # Legacy support for old IP constants (deprecated)
        self._guangyin_ip = os.getenv('GUANGYIN_DATABASE_IP', '192.168.110.150')
        self._guangyin_port = int(os.getenv('GUANGYIN_DATABASE_PORT', '53306'))
    
    @property
    def guangyin_ip(self) -> str:
        """Legacy property for Guangyin database IP."""
        import warnings
        warnings.warn(
            "Direct IP access is deprecated. Use environment variables instead.",
            DeprecationWarning,
            stacklevel=2
        )
        return self._guangyin_ip
    
    @property
    def guangyin_port(self) -> int:
        """Legacy property for Guangyin database port."""
        import warnings
        warnings.warn(
            "Direct port access is deprecated. Use environment variables instead.",
            DeprecationWarning,
            stacklevel=2
        )
        return self._guangyin_port


class EmailConfig:
    """Email configuration for notification services."""
    
    def __init__(self):
        self.smtp_host = os.getenv('NIRAPI_SMTP_HOST', 'smtp.163.com')
        self.smtp_port = int(os.getenv('NIRAPI_SMTP_PORT', '465'))
        self.email_user = os.getenv('NIRAPI_EMAIL_USER', '')
        self.email_password = os.getenv('NIRAPI_EMAIL_PASSWORD', '')


# Global configuration instances
db_config = DatabaseConfig()
email_config = EmailConfig()


def get_database_connection_params(
    database: str,
    host: Optional[str] = None,
    port: Optional[int] = None,
    user: Optional[str] = None,
    password: Optional[str] = None,
    charset: Optional[str] = None
) -> dict:
    """
    Get database connection parameters with fallbacks to configuration.
    
    Args:
        database: Database name
        host: Database host (falls back to config)
        port: Database port (falls back to config)
        user: Database user (falls back to config)
        password: Database password (falls back to config)
        charset: Database charset (falls back to config)
    
    Returns:
        Dictionary of connection parameters
    """
    return {
        'host': host or db_config.host,
        'port': port or db_config.port,
        'user': user or db_config.user,
        'password': password or db_config.password,
        'database': database,
        'charset': charset or db_config.charset
    }