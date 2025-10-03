import cv2
import numpy as np
import cv2


# line seperate fn ( according to x axis)
def separate_line_masks(binary_mask):
    contour_areaLimit = 500

    binary_mask = binary_mask.astype(np.uint8)
    _, binary_mask = cv2.threshold( binary_mask, 127, 255, cv2.THRESH_BINARY)
    contours, _ = cv2.findContours( binary_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    separated_lines = []
    for i, contour in enumerate(contours):
        # filter noise
        if cv2.contourArea(contour) > contour_areaLimit:
            line_image = np.zeros_like(binary_mask)
            cv2.drawContours( line_image, [contour], -1, 255, thickness=cv2.FILLED)

            separated_lines.append(line_image)

    return separated_lines


def image_to_camera_corrds( u, v, depth):
    fx = 959.5588989257812
    fy = 959.5588989257812
    cx = 631.89208984375
    cy = 3376.25347900390625

    if depth <= 0:
        return None
    
    x = (u - cx) * depth / fx
    y = (v - cy) * depth / fy
    z = depth

    return (x, y, z)



lines = cv2.imread( 'line_separate.jpg')
depth_map = cv2.imread( 'depth_pgm.pgm', cv2.IMREAD_UNCHANGED)
depth_map_m = depth_map.astype(np.float32)/1000

lines = cv2.cvtColor( lines, cv2.COLOR_BGR2GRAY)

point_list = []

separated_lines = separate_line_masks(lines)
for i, line_img in enumerate(separated_lines):
    skeleton = cv2.ximgproc.thinning(line_img)

    points = cv2.findNonZero(skeleton)
    point_list.append( points.reshape(-1,2))
    print(i , " " , points[0][0])

    cv2.imshow("Line "+str(i), line_img)
    cv2.imshow('skeleton '+str(i), skeleton)

for i , point in enumerate(point_list[0]):
    print("depth ", depth_map_m[point[1], point[0]])
    # x , y, z = image_to_camera_corrds( point[0], point[1], )

cv2.waitKey(0)
cv2.destroyAllWindows()
