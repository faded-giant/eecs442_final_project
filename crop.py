import cv2
import numpy as np
import os

# Prints the corner points found by cammy edge detection
DEBUG = False

input_folder = './sample_pokemon_scans/'
# input_folder = './in/'
intermediate_output_folder = './out_temp/'
final_output_folder = './out/'

# <Parameters>

# Coarse crop percentages
# These are the percentages of the edges to be croppped
# They should be ok as is. Increasing them seems to lead to cropping that is too tight
# and removes the edges.
LEFT_PCT = 0.055
RIGHT_PCT = 0.07
TOP_PCT = 0.25
BOTTOM_PCT = 0.08

# Kernel sizes for blurring card middle / edges
# Good to keep these are large as possible to ensure maximum gradients at edges
MIDDLE_BLUR = (55,55)
EGDE_BLUR = (41, 41)

# Part of middle to blur
# This is a ratio/percentage of the height/width
# These values should be ok as they are.
RATIO = 1.25
# Additional offset to compensate for the card being taller than
# it is wide. This is subtracted from the ratio.
# The larger this offset, the larger the part of the top/bottom that are blurred.
RATIO_HEIGHT_OFFSET = 0.05

# Part of edge to blur
# This is a percentage of the width or height of the card
EDGE_PCT = 0.03
# This is the factor to blur the top and bottom edges
# i.e. the (EDGE_PCT * EDGE_HEIGHT_FACTOR)% of the top and bottom are blurred
EDGE_HEIGHT_FACTOR = 1.5

# Thresholds for canny edge detection
# These seem to work quite well as they are
THRESHOLD1 = 50
THRESHOLD2 = 150
KERNEL_SIZE = 9

# </Parameters>

def coarse_crop_image(image, left_pct=LEFT_PCT, right_pct=RIGHT_PCT, top_pct=TOP_PCT, bottom_pct=BOTTOM_PCT, color: bool = False):
    if color:
        h, w, _ = image.shape
    else:
        h, w = image.shape

    left = int(left_pct * w)
    right = int(w - right_pct * w)
    top = int(top_pct * h)
    bottom = int(h - bottom_pct * h)

    return image[top:bottom, left:right]

def blur_image_middle(image, ratio=RATIO, ratio_height_offset=RATIO_HEIGHT_OFFSET, blur_size=MIDDLE_BLUR):
    
    h, w = image.shape

    center_x, center_y = w // 2, h // 2
    region_width, region_height = w // ratio, h // (ratio - ratio_height_offset)
    x1 = center_x - region_width // 2
    y1 = center_y - region_height // 2
    x2 = center_x + region_width // 2
    y2 = center_y + region_height // 2

    x1, y1 = max(0, x1), max(0, y1)
    x2, y2 = min(w, x2), min(h, y2)
    x1 = int(x1)
    y1 = int(y1)
    x2 = int(x2)
    y2 = int(y2)

    # Extract the middle region
    middle_region = image[y1:y2, x1:x2]
    
    # Apply Gaussian blur to the middle region
    blurred_middle = cv2.GaussianBlur(middle_region, blur_size, 0)
    
    # Replace the original middle region with the blurred one
    image[y1:y2, x1:x2] = blurred_middle

    return image

def blur_edges(image, blur_size=EGDE_BLUR, edge_pct=EDGE_PCT, edge_height_factor=EDGE_HEIGHT_FACTOR):

    # Get the dimensions of the image
    height, width = image.shape[:2]
    
    # Calculate the border width (1% of the image dimensions)
    tb_border_width = int(edge_pct * min(width, height))
    side_border_width = int(edge_pct * edge_height_factor * min(width, height))

    # Define the regions to blur: the 1% borders (edges)
    # Top border
    top_edge = image[0:tb_border_width, :]
    # Bottom border
    bottom_edge = image[height-tb_border_width:, :]
    # Left border
    left_edge = image[:, 0:side_border_width]
    # Right border
    right_edge = image[:, width-side_border_width:]
    
    # Apply Gaussian blur to the edge regions
    top_blurred = cv2.GaussianBlur(top_edge, blur_size, 0)
    bottom_blurred = cv2.GaussianBlur(bottom_edge, blur_size, 0)
    left_blurred = cv2.GaussianBlur(left_edge, blur_size, 0)
    right_blurred = cv2.GaussianBlur(right_edge, blur_size, 0)
    
    # Place the blurred edges back into the image
    image[0:tb_border_width, :] = top_blurred
    image[height-tb_border_width:, :] = bottom_blurred
    image[:, 0:side_border_width] = left_blurred
    image[:, width-side_border_width:] = right_blurred

    return image

def canny(image, threshold1=THRESHOLD1, threshold2=THRESHOLD2, kernel_size=KERNEL_SIZE):

    # Apply Canny edge detection with current threshold values
    edges = cv2.Canny(image, threshold1, threshold2)

    # Use kernel size for morphological closing (ensure it's odd and > 1)
    kernel = np.ones((kernel_size, kernel_size), np.uint8) if kernel_size > 1 else np.ones((3, 3), np.uint8)
    closed_edges = cv2.morphologyEx(edges, cv2.MORPH_CLOSE, kernel)

    # Find contours in the closed edge image
    contours, _ = cv2.findContours(closed_edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Find the contour with the largest area
    largest_contour = max(contours, key=cv2.contourArea)

    # Get the minimum area rectangle for the largest contour
    rect = cv2.minAreaRect(largest_contour)
    box = cv2.boxPoints(rect)  # Get the four vertices of the rectangle
    box = np.intp(box)

    # Get the width and height of the bounding box
    width = int(rect[1][0])
    height = int(rect[1][1])

    # Order the box points in a consistent order (top-left, top-right, bottom-right, bottom-left)
    # This is important for the perspective transformation
    rect_points = box.reshape(4, 2)
    rect_points = sorted(rect_points, key=lambda x: (x[1], x[0]))  # Sort by y and x coordinates

    if DEBUG:
        print (rect_points)

    # Determine the new corner order for perspective transform
    top_left, top_right = sorted(rect_points[:2], key=lambda x: x[0])
    bottom_left, bottom_right = sorted(rect_points[2:], key=lambda x: x[0])

    # Create an ordered set of points for perspective transformation
    src_pts = np.float32([top_left, top_right, bottom_right, bottom_left])

    # Define the destination rectangle (straight box)
    dst_pts = np.float32([[0, 0], [width, 0], [width, height], [0, height]])

    # Get the perspective transform matrix
    matrix = cv2.getPerspectiveTransform(src_pts, dst_pts)

    # Apply the perspective transformation (warp the image)
    cropped_image = cv2.warpPerspective(image, matrix, (width, height))

    # Get points to crop the color image
    x = rect_points[0][0]
    y = rect_points[0][1]
    h = rect_points[3][0]
    w = rect_points[3][1]

    return cropped_image, x, y, h, w


def main():
    num_failed = 0
    num_img = 0

    for img in os.listdir(input_folder):
        
        # if num_img > 20:
        #     exit()

        # This try-except block is just to ignore any images that empty output
        try:
            num_img += 1
            print(f"Processing {num_img}: {img}")
            original_image = cv2.imread(input_folder + img)

            image = original_image.copy()

            # Normalize iamge
            image = cv2.normalize(image, None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX)

            # Convert to grayscale
            image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            image = cv2.medianBlur(image, ksize=7)

            # Coarse crop
            image = coarse_crop_image(image)

            # Blur middle and edges
            image = blur_image_middle(image)
            image = blur_edges(image)

            # For some reason, this next part only works if the image is saved
            cv2.imwrite(intermediate_output_folder + img, image)
            image = cv2.imread(intermediate_output_folder + img)
            image, x, y, w, h = canny(image)
            # Overwrite temporary image
            cv2.imwrite(intermediate_output_folder + img, image)

            # Crop the original color images
            color_image = original_image.copy()
            color_image = coarse_crop_image(color_image, color=True)
            color_image = color_image[y:y+h, x:x+w]
            cv2.imwrite(final_output_folder + 'cropped_' + img, color_image)
        
        except Exception:
            print(f"Failed to process {num_img}: {img}, ignoring...")
            num_failed += 1
            pass

    print(f"Successfully processed {num_img-num_failed}/{num_img} images.")

if __name__ == '__main__':
    main()