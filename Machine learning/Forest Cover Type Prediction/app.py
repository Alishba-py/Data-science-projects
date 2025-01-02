import streamlit as st
import pickle
import numpy as np
from PIL import Image

# Load the model
rfc = pickle.load(open('rfc.pkl', 'rb'))

# Create the web app
st.title('Forest Cover Type Prediction')

# Display an introductory image
image = Image.open('img_2.png')  # Replace with your intro image path
st.image(image, caption='Forest Cover Type Prediction', use_container_width=True)

# User input for features
user_input = st.text_input('Input Features (comma-separated)')

expected_features = 55  # Replace with the actual number of features used during training

if user_input:
    # Process user input
    user_input = user_input.split(',')
    
    if len(user_input) != expected_features:
        st.error(f"Please provide exactly {expected_features} features.")
    else:
        features = np.array([user_input], dtype=np.float64)  # Convert to NumPy array

        # Make a prediction
        output = rfc.predict(features).reshape(1, -1)

        # Create the cover type dictionary with available images
        cover_type_dict = {
            2: {"name": "Lodgepole Pine", "image": "img_2.png"},
            3: {"name": "Ponderosa Pine", "image": "img_3.png"},
            6: {"name": "Douglas-fir", "image": "img_6.png"},
            7: {"name": "Krummholz", "image": "img_7.png"}
        }

        # Convert the output to integer
        predicted_cover_type = int(output[0])
        cover_type_info = cover_type_dict.get(predicted_cover_type)

        if cover_type_info is not None:
            cover_type_name = cover_type_info["name"]
            cover_type_image_path = cover_type_info["image"]

            # Display the cover type card
            col1, col2 = st.columns([2, 3])

            with col1:
                st.write("Predicted Cover Type:")
                st.write(f"<h1 style='font-size: 40px; font-weight: bold;'>{cover_type_name}</h1>", unsafe_allow_html=True)

            with col2:
                cover_type_image = Image.open(cover_type_image_path)
                st.image(cover_type_image, caption=cover_type_name, use_container_width=True)
        else:
            st.write("Unable to make a prediction")