import barcode
from barcode.writer import ImageWriter

def generate_barcode(data, filename="barcode"):
    try:
        # Select the barcode format (e.g., Code128, EAN13)
        barcode_class = barcode.get_barcode_class("code128")
        
        # Create the barcode
        barcode_obj = barcode_class(data, writer=ImageWriter())
        
        # Save the barcode to a file
        file_path = barcode_obj.save(filename)
        
        print(f"Barcode saved to: {file_path}")
    except Exception as e:
        print(f"Error: {e}")

# Example usage
data_to_encode = "123456789012"
generate_barcode(data_to_encode, filename="my_barcode")
