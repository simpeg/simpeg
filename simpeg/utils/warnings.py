"""
Custom warnings that can be used across SimPEG.
"""

__all__ = ["PerformanceWarning"]


class PerformanceWarning(Warning):
    """
    Warning raised when there is a possible performance impact.
    """
