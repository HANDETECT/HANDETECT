import cv2
import numpy as np
import matplotlib.pyplot as plt

# Function to display images
def display_image(image):
    plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
    plt.axis('off')
    plt.show()

# Function to extract curve data and calculate curvature
def extract_curve_data(image_path):
    # Load the image
    image = cv2.imread(image_path)
    if image is None:
        print("Error loading image!")
        return np.zeros((0,))  # Return empty curvature

    # Display the original image
    display_image(image)

    # Convert to grayscale
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Check if the image is empty or monochromatic
    if gray.size == 0 or np.all(gray == 255) or np.all(gray == 0):
        print("Image is empty or monochromatic. Skipping curvature calculation.")
        return np.zeros((0,))  # Return empty curvature

    # Apply Gaussian blur
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)

    # Binarize the image
    _, binary = cv2.threshold(blurred, 127, 255, cv2.THRESH_BINARY_INV)
    # Alternative thresholding
    # binary = cv2.adaptiveThreshold(blurred, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
    #                                cv2.THRESH_BINARY, 11, 2)

    # Display the binary image
    display_image(binary)

    # Find contours
    contours, _ = cv2.findContours(binary, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

    # Log contour information
    print(f"Binary image shape: {binary.shape}, unique values: {np.unique(binary)}")
    print(f"Number of contours found: {len(contours)}")

    if not contours:
        print("No contours found! Curve data extraction failed. Returning empty curvature.")
        return np.zeros((0,))  # Return empty curvature

    # Calculate curvature based on contours (placeholder logic)
    curvature = []  # Replace this with your curvature calculation logic

    # Example of processing contours (modify as needed)
    for contour in contours:
        # Here you would calculate curvature based on contour points
        # For demonstration, just getting the contour length
        curvature.append(cv2.arcLength(contour, closed=True))

    # Convert curvature list to numpy array
    curvature = np.array(curvature)
    print(f"Curvature calculated: {curvature}")

    # Return the curvature array or any other required processing
    return curvature

# Example usage
image_path = 'C:\\Users\\User\\Desktop\\SpiralSense\\data\\train\\Parkinsons Disease\\Parkinsons Disease_train_1.jpg'
curvature = extract_curve_data(image_path)

# Further processing can be done with the curvature data as needed
