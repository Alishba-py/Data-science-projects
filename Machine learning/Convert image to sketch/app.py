import cv2
import numpy as np

def convert_to_sketch(image_path, output_path):
    # Read the image
    image = cv2.imread(image_path)
    
    # Convert to grayscale
    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    
    # Invert the grayscale image
    inverted_gray_image = 255 - gray_image
    
    # Apply Gaussian blur
    blurred = cv2.GaussianBlur(inverted_gray_image, (21, 21), 0)
    
    # Invert the blurred image
    inverted_blurred = 255 - blurred
    
    # Create the pencil sketch
    sketch = cv2.divide(gray_image, inverted_blurred, scale=256.0)
    
    # Save the sketch image
    cv2.imwrite(output_path, sketch)

# Example usage
convert_to_sketch('2.jpg', 'output_sketch.jpg')