from rembg import remove
from PIL import Image
import os

# Specify the input image path
input_path = '2.jpeg'  # Change this to your specific image file name
output_path = 'output.jpg'  # Desired output file name

# Check if the file exists before attempting to open it
if not os.path.isfile(input_path):
    print(f"File not found: {input_path}")
else:
    # Load the image and remove the background
    image = Image.open(input_path).convert('RGBA')
    output_image = remove(image)

    # Convert the output image to RGB before saving as JPEG
    output_image = output_image.convert('RGB')

    # Save the output image in the desired format
    output_image.save(output_path, format='JPEG')  # Saving as JPEG
    print(f"Background removed from {input_path} and saved as {output_path}")