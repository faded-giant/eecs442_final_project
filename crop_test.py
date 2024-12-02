# import cv2
# import numpy as np
# from PIL import Image

# # Pokemon card aspect ratio is 1:√2

# def crop_image_with_minimum_size(image_path, output_path, min_width=100, min_height=100, threshold1=100, threshold2=200):
#     # Load the image
#     image = cv2.imread(image_path)
#     if image is None:
#         print("Error: Image not found.")
#         return
    
#     # Convert to grayscale
#     gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

#     # Apply Canny Edge Detection
#     edges = cv2.Canny(gray, threshold1, threshold2)

#     # Find contours
#     contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

#     # If contours are found, process the bounding box
#     if contours:
#         # Get the largest contour by area
#         largest_contour = max(contours, key=cv2.contourArea)
        
#         # Get the bounding rectangle of the largest contour
#         x, y, w, h = cv2.boundingRect(largest_contour)

#         # Ensure the bounding box meets the minimum size
#         if w < min_width or h < min_height:
#             print(f"Bounding box too small (w: {w}, h: {h}). Applying padding.")
#             # Pad the image to meet the minimum size
#             padded_width = max(w, min_width)
#             padded_height = max(h, min_height)
            
#             # Calculate padding needed
#             pad_x = max(0, (padded_width - w) // 2)
#             pad_y = max(0, (padded_height - h) // 2)
            
#             # Create padding
#             padded_image = cv2.copyMakeBorder(
#                 image[y:y+h, x:x+w],
#                 top=pad_y, bottom=pad_y,
#                 left=pad_x, right=pad_x,
#                 borderType=cv2.BORDER_CONSTANT,
#                 value=(0, 0, 0)  # Padding color (black)
#             )
            
#             cv2.imwrite(output_path, padded_image)
#             print(f"Image padded and saved to {output_path}")
#         else:
#             # Crop the image based on the bounding box
#             cropped_image = image[y:y+h, x:x+w]
#             cv2.imwrite(output_path, cropped_image)
#             print(f"Image cropped and saved to {output_path}")
#     else:
#         print("No edges found. Skipping cropping.")

# image_path = 'in/front.jpg'  # Replace with your image path
# output_path = 'out/front.jpg'  # Replace with your desired output path
# image = Image.open('in/front.jpg')
# w, h = image.size

# # Crop 5% off the left and right
# # 25% off the top (for the label) and 5% off the bottom
# l_crop = int(0.08*w)
# r_crop = int(w - l_crop)
# t_crop = int(0.25*h)
# b_crop = int(h - 0.05*h)
# image = image.crop((l_crop, t_crop, r_crop, b_crop))
# image = np.array(image)
# image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
# image = cv2.medianBlur(image, ksize=7)
# cv2.imwrite('test.jpg', image)

# image_path = 'test.jpg'
# output_path = 'out.jpg'
# # print(f"{w}, {h}")
# mw = int(w*0.7)
# mh = int(mw*np.sqrt(2))
# print(f"{mw}, {mh}")

# crop_image_with_minimum_size(image_path, output_path, threshold1=0, threshold2=255)

import cv2
import numpy as np

# Callback function for trackbars
def nothing(x):
    pass

# Load the image
image = cv2.imread('out/front3.jpg')

# Convert the image to grayscale
# gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
gray = image

# Create a window for trackbars
cv2.namedWindow("Contour Detection")

# Create trackbars for Canny thresholds and kernel size
cv2.createTrackbar("Threshold1", "Contour Detection", 50, 255, nothing)
cv2.createTrackbar("Threshold2", "Contour Detection", 150, 255, nothing)
cv2.createTrackbar("Kernel Size", "Contour Detection", 5, 20, nothing)  # Kernel size for morphological operations
first = True
while True:
    # Get current positions of the trackbars
    threshold1 = cv2.getTrackbarPos("Threshold1", "Contour Detection")
    threshold2 = cv2.getTrackbarPos("Threshold2", "Contour Detection")
    kernel_size = cv2.getTrackbarPos("Kernel Size", "Contour Detection")

    # Apply Canny edge detection with current threshold values
    edges = cv2.Canny(gray, threshold1, threshold2)

    # Use kernel size for morphological closing (ensure it's odd and > 1)
    kernel = np.ones((kernel_size, kernel_size), np.uint8) if kernel_size > 1 else np.ones((3, 3), np.uint8)
    closed_edges = cv2.morphologyEx(edges, cv2.MORPH_CLOSE, kernel)

    # Find contours in the closed edge image
    contours, _ = cv2.findContours(closed_edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # If no contours are found, skip the iteration
    if len(contours) == 0:
        continue

    # Find the contour with the largest area
    largest_contour = max(contours, key=cv2.contourArea)

    # Make a copy of the original image to draw the largest contour
    image_with_largest_contour = image.copy()

    # Get the minimum area rectangle for the largest contour
    rect = cv2.minAreaRect(largest_contour)
    box = cv2.boxPoints(rect)  # Get the four vertices of the rectangle
    if first:
        print(f"thresholds: {threshold1}, {threshold2}")
        print(f"kernel_size: {kernel_size}")
        print(box)
        first = False
    box = np.int0(box)  # Convert points to integer

    # Draw the rectangle on the image
    cv2.drawContours(image_with_largest_contour, [box], 0, (0, 255, 0), 2)

    # Display the image with the rectangle
    cv2.imshow('Contour Detection - Rectangular Approximation', image_with_largest_contour)
    cv2.imshow('Edges', closed_edges)

    # Wait for user input to exit
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Clean up and close all windows
cv2.destroyAllWindows()
