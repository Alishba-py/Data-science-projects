
import qrcode

def generate_qr_code(link, output_file):
    """
    Generate a QR code for a given link and save it as an image file.

    Parameters:
    - link (str): The URL or text to encode in the QR code.
    - output_file (str): The file path where the QR code image will be saved.
    """
    # Create a QR Code object
    qr = qrcode.QRCode(
        version=1,  # Controls the size of the QR Code (1 = small, 40 = large)
        error_correction=qrcode.constants.ERROR_CORRECT_L,  # Error correction level
        box_size=10,  # Size of each QR code box
        border=4,  # Border size
    )
    
    # Add data to the QR code
    qr.add_data(link)
    qr.make(fit=True)
    
    # Generate the QR code image
    img = qr.make_image(fill_color="black", back_color="white")
    
    # Save the image to a file
    img.save(output_file)
    print(f"QR code saved as {output_file}. Scan it to visit the link!")

# Main program
if __name__ == "__main__":
    # Link to encode in the QR code
    profile_link = ""  # Add your  profile link here
    
    # File name to save the QR code image
    output_filename = "qr.png"
    
    # Generate the QR code
    generate_qr_code(profile_link, output_filename)
