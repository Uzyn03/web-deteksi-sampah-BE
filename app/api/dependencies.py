from fastapi import Depends, HTTPException, Request
from fastapi.security import HTTPBearer
from typing import Optional
import time
from collections import defaultdict
from app.config import settings
from app.utils.logger import get_logger

logger = get_logger()

# Simple in-memory rate limiter
rate_limiter = defaultdict(list)
security = HTTPBearer(auto_error=False)

def get_client_ip(request: Request) -> str:
    """Get client IP address"""
    forwarded = request.headers.get("X-Forwarded-For")
    if forwarded:
        return forwarded.split(',')[0].strip()
    return request.client.host

def rate_limit_check(request: Request) -> bool:
    """Simple rate limiting check"""
    client_ip = get_client_ip(request)
    current_time = time.time()
    
    # Clean old entries (older than 1 minute)
    cutoff_time = current_time - 60
    rate_limiter[client_ip] = [
        timestamp for timestamp in rate_limiter[client_ip] 
        if timestamp > cutoff_time
    ]
    
    # Check rate limit
    if len(rate_limiter[client_ip]) >= settings.rate_limit_per_minute:
        return False
    
    # Add current request
    rate_limiter[client_ip].append(current_time)
    return True

def check_rate_limit(request: Request):
    """Rate limiting dependency"""
    if not rate_limit_check(request):
        logger.warning(f"Rate limit exceeded for IP: {get_client_ip(request)}")
        raise HTTPException(
            status_code=429, 
            detail="Rate limit exceeded. Please try again later."
        )

# Optional authentication dependency (implement if needed)
def get_current_user(token: Optional[str] = Depends(security)):
    """Get current user (placeholder for authentication)"""
    # Implement your authentication logic here
    # For now, just return None (no authentication required)
    return None