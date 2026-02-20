"""
Custom warnings that can be used across SimPEG.
"""

__all__ = ["BreakingChangeWarning", "PerformanceWarning"]


class BreakingChangeWarning(Warning):
    """
    Warning to let users know about a breaking change that was introduced.
    """


class PerformanceWarning(Warning):
    """
    Warning raised when there is a possible performance impact.
    """
