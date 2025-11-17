"""
Adaptive threshold configuration for OWL-ViT detections.

Based on empirical analysis of garden_large dataset, different object
categories require different confidence thresholds for reliable detection.
"""

# Category-specific thresholds OPTIMIZED FOR PERFORMANCE
# Goal: 5-10 high-quality detections per frame instead of 50-100 low-quality ones
# Strategy: Use P95 (95th percentile) instead of P90 to be more selective
CATEGORY_THRESHOLDS = {
    # Buildings/structures - use P95 for better quality
    "house": 0.089,      # P95: 0.089 (was 0.043) - much more selective
    "building": 0.038,   # P95: 0.038 (was 0.029)
    "shed": 0.040,       # P95: 0.040 (was 0.011)
    "door": 0.049,       # P95: 0.049 (was 0.025)
    "window": 0.020,     # P95: 0.020 (was 0.016)
    "roof": 0.041,       # P95: 0.041 (was 0.026)
    
    # Natural landmarks - use P95
    "tree": 0.065,       # P95: 0.065 (was 0.038) - 2x more selective
    
    # Vegetation - use P95 (these were causing explosion)
    "bush": 0.019,       # P95: 0.019 (was 0.011)
    "grass": 0.012,      # P95: 0.012 (was 0.009)
    "flowers": 0.017,    # P95: 0.017 (was 0.012)
    "plant": 0.011,      # P95: 0.011 (was 0.008)
}

# Default threshold for unlisted categories
DEFAULT_THRESHOLD = 0.12

def get_threshold(label: str) -> float:
    """Get adaptive threshold for a given label."""
    return CATEGORY_THRESHOLDS.get(label.lower(), DEFAULT_THRESHOLD)


# Query groupings for focused detection
# Based on empirical analysis showing which categories actually get detected

# HIGH CONFIDENCE: Max score > 0.10, reasonable P90
HIGH_CONFIDENCE_QUERIES = [
    "house",      # Max: 0.280, P90: 0.043 ✓
    "tree",       # Max: 0.132, P90: 0.038 ✓
    "window",     # Max: 0.105, P90: 0.016
    "roof",       # Max: 0.099, P90: 0.026
    "door",       # Max: 0.095, P90: 0.025
    "shed",       # Max: 0.095, P90: 0.011
]

# MEDIUM CONFIDENCE: Max score 0.05-0.10
MEDIUM_CONFIDENCE_QUERIES = [
    "bush",       # Max: 0.068, P90: 0.011
    "grass",      # Max: 0.060, P90: 0.009
    "fountain",   # Max: 0.054, P90: 0.008
    "flowers",    # Max: 0.052, P90: 0.012
    "building",   # Max: 0.047, P90: 0.029
    "plant",      # Max: 0.047, P90: 0.008
]

# LOW CONFIDENCE: Max score 0.03-0.05 (noisy, use with caution)
LOW_CONFIDENCE_QUERIES = [
    "lamp",       # Max: 0.044, P90: 0.009
    "shrub",      # Max: 0.039, P90: 0.026
    "leaves",     # Max: 0.037, P90: 0.009
    "wall",       # Max: 0.034, P90: 0.006
    "rock",       # Max: 0.033, P90: 0.020
    "fence",      # Max: 0.033, P90: 0.015
    "hedge",      # Max: 0.032, P90: 0.014
    "branch",     # Max: 0.030, P90: 0.010
]

# UNRELIABLE: Max score < 0.03 (mostly noise, not recommended)
UNRELIABLE_QUERIES = [
    "puddle", "snow", "mud", "stone", "path", "walkway", "post",
    "bench", "statue", "planter", "gate", "ice", "foliage", "pot"
]

# Recommended query sets for different use cases
STRUCTURAL_QUERIES = [
    "house", "building", "shed", "door", "window", "roof"
]

VEGETATION_QUERIES = [
    "tree", "bush", "shrub", "grass", "flowers", "plant", "leaves"
]

SEASONAL_QUERIES = [
    # WARNING: These have very low detection rates in this dataset
    "snow",       # Max: 0.021, P90: 0.001 - mostly noise
    "puddle",     # Max: 0.026, P90: 0.015 - occasionally works
    "mud",        # Max: 0.021, P90: 0.005 - mostly noise
]

SMALL_OBJECT_QUERIES = [
    "lamp", "rock", "fence"
]

# RECOMMENDED: Best performing queries for garden scenes
# PERFORMANCE OPTIMIZED: Reduced to top 6 most reliable categories
# This should give 5-15 detections per frame instead of 50-100
RECOMMENDED_GARDEN_QUERIES = [
    "house",      # Max: 0.280, very reliable
    "tree",       # Max: 0.132, second most reliable
    "window",     # Max: 0.105, good for building details
    "door",       # Max: 0.095, good for building details  
    "bush",       # Max: 0.068, best vegetation detector
    "grass",      # Max: 0.060, ground coverage
]

# Alternative: If you want 10-20 detections, use this expanded set
EXPANDED_GARDEN_QUERIES = HIGH_CONFIDENCE_QUERIES + [
    "bush", "grass", "flowers", "plant"  # Add vegetation
]

# ULTRA FAST: Top 3 most reliable categories only (3-8 detections per frame)
ULTRA_FAST_QUERIES = [
    "house",   # Max: 0.280 - best detector
    "tree",    # Max: 0.132 - second best
    "window",  # Max: 0.105 - building details
]
