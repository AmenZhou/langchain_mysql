"""
Admin API Endpoints

Administrative endpoints for controlling application configuration,
including PII filtering toggle, chart generation settings, and system status.
"""

from typing import Dict, Any
from fastapi import APIRouter, HTTPException, status
from pydantic import BaseModel
import logging

from config import get_config, is_pii_filtering_enabled, is_chart_generation_enabled

logger = logging.getLogger(__name__)

# Create router for admin endpoints
router = APIRouter(prefix="/admin", tags=["admin"])


class ConfigToggleRequest(BaseModel):
    """Request model for toggling configuration settings."""
    enabled: bool


class ConfigResponse(BaseModel):
    """Response model for configuration operations."""
    success: bool
    message: str
    current_state: bool


class SystemStatusResponse(BaseModel):
    """Response model for system status."""
    status: str
    configuration: Dict[str, Any]
    features: Dict[str, bool]


@router.get("/status", response_model=SystemStatusResponse)
async def get_system_status():
    """
    Get current system status and configuration.
    
    Returns information about enabled features, configuration settings,
    and overall system health.
    """
    try:
        config = get_config()
        
        return SystemStatusResponse(
            status="healthy",
            configuration=config.get_summary(),
            features={
                "pii_filtering": is_pii_filtering_enabled(),
                "chart_generation": is_chart_generation_enabled(),
                "debug_mode": config.DEBUG_MODE,
                "development_mode": config.is_development_mode()
            }
        )
    except Exception as e:
        logger.error(f"Error getting system status: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to get system status: {str(e)}"
        )


@router.post("/pii-filtering/toggle", response_model=ConfigResponse)
async def toggle_pii_filtering():
    """
    Toggle PII filtering on/off.
    
    This endpoint allows runtime control of PII filtering without
    restarting the application.
    """
    try:
        config = get_config()
        new_state = config.toggle_pii_filtering()
        
        status_msg = "enabled" if new_state else "disabled"
        logger.info(f"PII filtering toggled to: {status_msg}")
        
        return ConfigResponse(
            success=True,
            message=f"PII filtering has been {status_msg}",
            current_state=new_state
        )
    except Exception as e:
        logger.error(f"Error toggling PII filtering: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to toggle PII filtering: {str(e)}"
        )


@router.post("/pii-filtering/enable", response_model=ConfigResponse)
async def enable_pii_filtering():
    """Enable PII filtering."""
    try:
        config = get_config()
        config.enable_pii_filtering()
        
        return ConfigResponse(
            success=True,
            message="PII filtering has been enabled",
            current_state=True
        )
    except Exception as e:
        logger.error(f"Error enabling PII filtering: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to enable PII filtering: {str(e)}"
        )


@router.post("/pii-filtering/disable", response_model=ConfigResponse)
async def disable_pii_filtering():
    """Disable PII filtering."""
    try:
        config = get_config()
        config.disable_pii_filtering()
        
        return ConfigResponse(
            success=True,
            message="PII filtering has been disabled",
            current_state=False
        )
    except Exception as e:
        logger.error(f"Error disabling PII filtering: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to disable PII filtering: {str(e)}"
        )


@router.get("/pii-filtering/status")
async def get_pii_filtering_status():
    """Get current PII filtering status."""
    return {
        "enabled": is_pii_filtering_enabled(),
        "message": f"PII filtering is {'enabled' if is_pii_filtering_enabled() else 'disabled'}"
    }


@router.post("/chart-generation/toggle", response_model=ConfigResponse)
async def toggle_chart_generation():
    """Toggle chart generation on/off."""
    try:
        config = get_config()
        current_state = config.ENABLE_CHART_GENERATION
        new_state = not current_state
        
        if new_state:
            config.enable_chart_generation()
        else:
            config.disable_chart_generation()
        
        status_msg = "enabled" if new_state else "disabled"
        
        return ConfigResponse(
            success=True,
            message=f"Chart generation has been {status_msg}",
            current_state=new_state
        )
    except Exception as e:
        logger.error(f"Error toggling chart generation: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to toggle chart generation: {str(e)}"
        )


@router.get("/chart-generation/status")
async def get_chart_generation_status():
    """Get current chart generation status."""
    return {
        "enabled": is_chart_generation_enabled(),
        "message": f"Chart generation is {'enabled' if is_chart_generation_enabled() else 'disabled'}"
    }


@router.get("/config")
async def get_configuration():
    """Get current application configuration."""
    try:
        config = get_config()
        return {
            "success": True,
            "configuration": config.get_summary()
        }
    except Exception as e:
        logger.error(f"Error getting configuration: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to get configuration: {str(e)}"
        ) 