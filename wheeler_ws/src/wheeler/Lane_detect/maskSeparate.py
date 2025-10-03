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


lines = cv2.imread( 'line_separate.jpg')
lines = cv2.cvtColor( lines, cv2.COLOR_BGR2GRAY)

separated_lines = separate_line_masks(lines)
for i, line_img in enumerate(separated_lines):
    skeleton = cv2.ximgproc.thinning(line_img)
    points = cv2.findNonZero(skeleton)
    print(i , " " , points[0][0])

    cv2.imshow("Line "+str(i), line_img)
    cv2.imshow('skeleton '+str(i), skeleton)

cv2.waitKey(0)
cv2.destroyAllWindows()
