import cv2

# Load the image
image_path = "C:/Users/rodul/Downloads/val_batch0_labels.jpg"
image = cv2.imread(image_path)

# Display the image
cv2.imshow('Select ROI', image)

# Prompt the user to draw a bounding box
rect = cv2.selectROI('Select ROI', image, fromCenter=False, showCrosshair=True)
cv2.destroyWindow('Select ROI')  # Close the window after ROI selection

# Extract the portion of the image within the ROI
x, y, w, h = rect
roi = image[y:y + h, x:x + w]

# Get the width and height of the ROI
roi_width = w
roi_height = h

print(f"Width of the ROI: {roi_width} pixels")
print(f"Height of the ROI: {roi_height} pixels")
