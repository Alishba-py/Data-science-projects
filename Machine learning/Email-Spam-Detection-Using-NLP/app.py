import pickle
import streamlit as st

# Load the model and vectorizer
model = pickle.load(open("spam.pkl", "rb"))
vectorizer = pickle.load(open("vectorizer.pkl", "rb"))
# Title of the app
st.title("Email Spam Classification")

# Input text area for the email content
email_text = st.text_area("Enter the email content:")

# Button to classify the email
if st.button("Predict"):
    # Transform the input text using the vectorizer
    email_vector = vectorizer.transform([email_text])
    
    # Predict using the loaded model
    prediction = model.predict(email_vector)
    
    # Display the result
    if prediction[0] == 1:
        st.success("This email is classified as **Spam**.")
    else:
        st.success("This email is classified as **Not Spam**.")

