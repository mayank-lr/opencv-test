# Contour Cleaning - Remove Repetitive Zigzag Patterns

This document explains how to clean contour data that has repetitive parallel coordinates creating a zigzag/noise pattern.

## Problem

When you have contour data that looks like this:
```
[[243, 206], [241, 204], [241, 183], [242, 182], [242, 160], [241, 159], ...
```

Notice how the x-coordinates jump between 241, 242, 243 repeatedly - this creates the zigzag noise pattern.

## Solution

I've created two functions in `clean_contour.py`:

### 1. `clean_repetitive_contour_simple()` - Recommended

This function removes points that create back-and-forth patterns in the x-direction with a threshold.

**Parameters:**
- `points`: Your contour array (N, 2) with [x, y] coordinates
- `x_threshold`: Maximum x-coordinate variation to consider as repetitive (default=2)

**How it works:**
- Uses sliding window logic
- Detects direction changes in x-coordinate
- Removes points that create small back-and-forth movements (+1, -1 pattern)

### 2. `clean_repetitive_contour()` - Advanced

This function focuses on a specific median region and cleans based on proximity to that median.

**Parameters:**
- `points`: Your contour array
- `median_region`: The approximate x-coordinate center (e.g., 241)
- `threshold`: Maximum distance from median to process (default=2)
- `window_size`: Size of sliding window (default=3)

## Usage Example

```python
import numpy as np
from clean_contour import clean_repetitive_contour_simple

# Your contour data
contour = np.array(data['contours'][7], dtype=np.int32)

# Clean it
cleaned_contour = clean_repetitive_contour_simple(contour, x_threshold=2)

# Result: Reduced from many points to just the essential trajectory
print(f"Original: {len(contour)} points")
print(f"Cleaned: {len(cleaned_contour)} points")
```

## Test Results

Original data (20 points with zigzag):
```
[[243, 206], [241, 204], [241, 183], [242, 182], ... [242, 128]]
```

Cleaned data (7 points, smooth line):
```
[[243, 206], [242, 160], [242, 128], [242, 102], [242, 85], [242, 111], [242, 128]]
```

**Reduction: 65% fewer points while maintaining the essential contour shape!**

## Next Steps

1. Load your full contour data
2. Apply the cleaning function
3. Adjust `x_threshold` if needed (increase for more aggressive cleaning)
4. Visualize the before/after to verify results

## Function Reference

See `clean_contour.py` for full implementation details.
