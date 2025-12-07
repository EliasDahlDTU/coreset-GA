"""
Configuration package for coreset-GA project.

Usage:
    # Import everything from default config (backward compatible)
    from config import K_VALUES, GA_POPULATION_SIZE, etc.
    
    # Or import the default module
    from config import default
    
    # Or import specific configs (when you add more)
    from config import default, experimental
"""

# Re-export everything from default for backward compatibility
# This allows: from config import K_VALUES, GA_POPULATION_SIZE, etc.
from config.default import *

