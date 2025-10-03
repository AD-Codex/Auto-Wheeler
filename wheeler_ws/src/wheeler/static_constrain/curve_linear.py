import numpy as np
import matplotlib.pyplot as plt

# Define the cubic curve
def cubic_curve(x):
    a, b, c, d, e, f = 1, -2, 3, -1, 2, 0  # Example coefficients
    return a*x**5 + b*x**4 + c*x**3 + d*x**2 + e*x + f

# Adaptive segmentation algorithm
def adaptive_segmentation(x, y, tolerance, side):
    segments = []
    start_idx = 0

    while start_idx < len(x) - 1:
        # Start with the current point
        for end_idx in range(start_idx + 1, len(x)):
            # Fit a line between start_idx and end_idx
            x1, x2 = x[start_idx], x[end_idx]
            y1, y2 = y[start_idx], y[end_idx]
            m = (y2 - y1) / (x2 - x1)
            c = y1 - m * x1

            # Compute the error for points between start_idx and end_idx
            errors = np.abs(y[start_idx:end_idx+1] - (m * x[start_idx:end_idx+1] + c))

            # Check if the maximum error exceeds the tolerance
            if np.max(errors) > tolerance:
                # Finalize the segment at the previous point
                end_idx -= 1
                x1, x2 = x[start_idx], x[end_idx]
                y1, y2 = y[start_idx], y[end_idx]
                m = (y2 - y1) / (x2 - x1)
                c = y1 - m * x1 - tolerance*side
                segments.append((x1, x2, m, c))
                start_idx = end_idx
                break
        else:
            # If we reach the last point, finalize the segment
            x1, x2 = x[start_idx], x[-1]
            y1, y2 = y[start_idx], y[-1]
            m = (y2 - y1) / (x2 - x1)
            c = y1 - m * x1 - tolerance*side
            segments.append((x1, x2, m, c))
            break

    return segments

# Generate sample points along the curve
x_points = np.linspace(-2, 2, 200)
y_points = cubic_curve(x_points)
y_points2 = cubic_curve(x_points) - 100

# Set the error tolerance
tolerance = 5 # Adjust for desired precision

# Get the linear segments
segments = adaptive_segmentation(x_points, y_points, tolerance, 1)
segments2 = adaptive_segmentation(x_points, y_points2, tolerance, -1)

# Plot the original curve
x_smooth = np.linspace(-2, 2, 1000)
y_smooth = cubic_curve(x_smooth)
y_smooth2 = cubic_curve(x_smooth) -100
plt.plot(x_smooth, y_smooth, label="Original Curve", color="blue")
plt.plot(x_smooth, y_smooth2, label="Original Curve2", color="blue")

# Plot the piecewise linear approximation
for x1, x2, m, c in segments:
    plt.plot([x1, x2], [m*x1 + c, m*x2 + c], label=f"Segment: {x1:.2f} to {x2:.2f}", color="red")

for x1, x2, m, c in segments2:
    plt.plot([x1, x2], [m*x1 + c, m*x2 + c], label=f"Segment: {x1:.2f} to {x2:.2f}", color="red")


# Display the plot
plt.scatter(x_points, y_points, color="black", label="Sample Points", s=10)
plt.legend()
plt.title("Optimized Piecewise Linear Approximation")
plt.xlabel("X")
plt.ylabel("Y")
plt.show()

# Print the equations
print("Linear Equations for Segments1:")
for i, (x1, x2, m, c) in enumerate(segments):
    print(f"Segment1 {i+1}: y = {m:.3f}x + {c:.3f}, for x in [{x1:.3f}, {x2:.3f}]")

print("Linear Equations for Segments2:")
for i, (x1, x2, m, c) in enumerate(segments2):
    print(f"Segment2 {i+1}: y = {m:.3f}x + {c:.3f}, for x in [{x1:.3f}, {x2:.3f}]")