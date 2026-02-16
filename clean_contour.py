import numpy as np

def clean_repetitive_contour(points, median_region=241, threshold=2, window_size=3):
    """
    Clean repetitive zigzag patterns in contour data.
    
    Parameters:
    -----------
    points : ndarray
        Array of shape (N, 2) containing [x, y] coordinates
    median_region : int or float
        The approximate x-coordinate median/center around which to clean
    threshold : int or float
        Maximum allowed x-coordinate difference from median for points to be considered part of the parallel lines
    window_size : int
        Size of sliding window to detect direction changes
    
    Returns:
    --------
    cleaned_points : ndarray
        Cleaned array of points with repetitive zigzag removed
    """
    
    # Step 1: Sort points by x-coordinate
    sorted_indices = np.argsort(points[:, 0])
    sorted_points = points[sorted_indices]
    
    # Step 2: Identify points near the median region (where zigzag occurs)
    x_coords = sorted_points[:, 0]
    near_median = np.abs(x_coords - median_region) <= threshold
    
    # Step 3: Process points using sliding window
    cleaned = []
    i = 0
    
    while i < len(sorted_points):
        # Check if current point is in the problematic region
        if near_median[i]:
            # Look ahead to find all consecutive points in this region
            j = i
            while j < len(sorted_points) and near_median[j]:
                j += 1
            
            # Extract this segment
            segment = sorted_points[i:j]
            
            # For this segment, keep only the median trajectory
            # Group by direction and keep representative points
            if len(segment) > 0:
                # Calculate median x for this segment
                median_x = np.median(segment[:, 0])
                
                # Keep the point closest to median x for each unique y region
                unique_ys = []
                for point in segment:
                    # Check if this y-value is significantly different from existing ones
                    is_new = True
                    for existing_y in unique_ys:
                        if abs(point[1] - existing_y) < 5:  # tolerance for y-values
                            is_new = False
                            break
                    
                    if is_new:
                        unique_ys.append(point[1])
                        # Add point with median x-coordinate
                        cleaned.append([median_x, point[1]])
            
            i = j
        else:
            # Keep points outside the problematic region as-is
            cleaned.append(sorted_points[i])
            i += 1
    
    return np.array(cleaned, dtype=points.dtype)


def clean_repetitive_contour_simple(points, x_threshold=2):
    """
    Simplified version: removes points that create back-and-forth patterns in x-direction.
    
    Parameters:
    -----------
    points : ndarray
        Array of shape (N, 2) containing [x, y] coordinates
    x_threshold : int or float
        Maximum x-coordinate variation to consider as "parallel/repetitive"
    
    Returns:
    --------
    cleaned_points : ndarray
        Cleaned array of points
    """
    if len(points) < 3:
        return points
    
    cleaned = [points[0]]
    
    for i in range(1, len(points) - 1):
        prev_point = cleaned[-1]
        curr_point = points[i]
        next_point = points[i + 1]
        
        # Calculate direction changes
        dx1 = curr_point[0] - prev_point[0]
        dx2 = next_point[0] - curr_point[0]
        
        # Check if this creates a back-and-forth pattern with small x-variation
        if abs(dx1) <= x_threshold and abs(dx2) <= x_threshold and np.sign(dx1) != np.sign(dx2):
            # Skip this point - it's part of a zigzag
            continue
        
        cleaned.append(curr_point)
    
    # Always keep the last point
    cleaned.append(points[-1])
    
    return np.array(cleaned, dtype=points.dtype)


# Example usage:
if __name__ == "__main__":
    # Example data (from your notebook)
    sample_data = np.array([
        [243, 206], [241, 204], [241, 183], [242, 182],
        [242, 160], [241, 159], [241, 129], [242, 128],
        [242, 111], [241, 110], [241, 103], [242, 102],
        [242, 85], [243, 84], [242, 85], [242, 102],
        [241, 103], [241, 110], [242, 111], [242, 128]
    ])
    
    print("Original points:")
    print(sample_data)
    print(f"\nNumber of points: {len(sample_data)}")
    
    cleaned = clean_repetitive_contour_simple(sample_data, x_threshold=2)
    
    print("\nCleaned points:")
    print(cleaned)
    print(f"\nNumber of points after cleaning: {len(cleaned)}")
