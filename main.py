import streamlit as st
from tensorflow.keras.preprocessing import image
import numpy as np
import tensorflow as tf
from PIL import Image
import pandas as pd
import io

# Load your pre-trained model
model = tf.keras.models.load_model('apple_classifier.h5')

# Define image size and class labels
img_width, img_height = 150, 150
class_labels = ['ripe', 'rotten', 'unripe']

# Custom CSS for card styling
st.markdown("""
    <style>
        .card {
            border-radius: 10px;
            padding: 20px;
            box-shadow: 0 4px 8px rgba(0, 0, 0, 0.1);
            transition: 0.3s;
            margin: 15px 0;
            border: 2px solid #f0f0f0;
        }
        .card:hover {
            box-shadow: 0 8px 16px rgba(0, 0, 0, 0.2);
            border: 2px solid #ff6347;
        }
        .card-header {
            font-size: 24px;
            font-weight: bold;
            color: #ff6347;
        }
        .card-body {
            font-size: 18px;
        }
    </style>
""", unsafe_allow_html=True)

# Set up the Streamlit page title and header
st.title('🍏 Apple Classifier')
st.sidebar.title("Navigation")
options = ["Home", "Apple History", "Nutritional Benefits", "Vitamins", "Health Benefits", "Feedback"]
choice = st.sidebar.selectbox("Select a page", options)

if choice == "Home":
    # Home Page: Apple Classifier
    st.header('Upload an image to classify the state of an apple')

    # Class Descriptions
    st.markdown("""
    ### Class Descriptions:
    - **Ripe:** Apples that are ready to eat.
    - **Rotten:** Apples that have begun to decay and are not safe to consume.
    - **Unripe:** Apples that need more time to mature before they are edible.
    """)

    # Image upload interface (accepts only one image)
    uploaded_file = st.file_uploader("Choose an image...", type="jpg", key="upload_image")

    # Initialize results storage
    if 'results' not in st.session_state:
        st.session_state.results = []
    if 'current_prediction' not in st.session_state:
        st.session_state.current_prediction = None

    if uploaded_file is not None:
        # Button to show image
        if st.button("Show Image"):
            # Load and display the uploaded image
            img = Image.open(uploaded_file)
            st.image(img, caption='Uploaded Image', use_column_width=True)

        # Preprocess the image for the model
        img_resized = Image.open(uploaded_file).resize((img_width, img_height))
        img_array = image.img_to_array(img_resized)
        img_array = np.expand_dims(img_array, axis=0)
        img_array /= 255.0

        # Button to show prediction
        if st.button("Show Prediction"):
            # Make a prediction
            predictions = model.predict(img_array)
            predicted_class_index = np.argmax(predictions, axis=1)[0]
            predicted_class = class_labels[predicted_class_index]
            confidence = np.max(predictions)  # Get the confidence score

            # Store the result (current prediction)
            st.session_state.current_prediction = (uploaded_file.name, predicted_class, confidence)
            
            # Add the result to the session results list
            st.session_state.results.append({
                'image': uploaded_file.name,
                'prediction': predicted_class,
                'confidence': confidence
            })

            # Display the prediction with confidence in a styled card
            if st.session_state.current_prediction:
                img_name, pred_class, conf = st.session_state.current_prediction
                st.markdown(f"""
                <div class="card">
                    <div class="card-header">Prediction Result</div>
                    <div class="card-body">
                        <p><strong>Image:</strong> {img_name}</p>
                        <p><strong>Prediction:</strong> {pred_class}</p>
                        <p><strong>Confidence:</strong> {conf:.2f}</p>
                    </div>
                </div>
                """, unsafe_allow_html=True)

        # Button to download all results as a CSV file
        if st.session_state.results:
            results_df = pd.DataFrame(st.session_state.results)
            csv = results_df.to_csv(index=False)
            st.download_button(
                label="Download All Results",
                data=csv,
                file_name='apple_predictions.csv',
                mime='text/csv'
            )

    # # Show a reset button to clear the current upload (retain results)
    # if uploaded_file is not None:
    #     if st.button("Reset Image", key="reset_button"):
    #         # Clear uploaded file and reset prediction
    #         st.session_state.upload_image = None
    #         st.session_state.current_prediction = None
    #         st.experimental_rerun()

elif choice == "Apple History":
    # Apple History Page
    st.header("History of Apples")
    st.markdown("""
    Apples are one of the oldest cultivated fruits, with origins tracing back to the 
    Tien Shan mountains of Central Asia. They have been grown for thousands of years 
    in Asia and Europe and were brought to North America by European colonists. 
    The Malus domestica species has undergone extensive breeding, leading to a vast 
    variety of apple cultivars available today.
    """)

elif choice == "Nutritional Benefits":
    # Nutritional Benefits Page
    st.header("Nutritional Benefits of Apples")
    st.markdown("""
    Apples are low in calories and rich in essential nutrients. A medium apple contains:
    - **Calories:** 95
    - **Fiber:** 4 grams
    - **Vitamin C:** 14% of the Daily Value (DV)
    - **Potassium:** 6% of the DV
    - **Vitamin K:** 5% of the DV
    Apples also contain various antioxidants, which can help combat oxidative stress.
    """)

elif choice == "Vitamins":
    # Vitamins Page
    st.header("Vitamins in Apples")
    st.markdown("""
    Apples are a good source of vitamins, particularly:
    - **Vitamin C:** Important for the immune system and skin health.
    - **Vitamin A:** Essential for vision and skin health.
    - **B vitamins:** Including B6 and riboflavin, which play a role in energy metabolism.
    Eating apples contributes to meeting daily vitamin requirements.
    """)

elif choice == "Health Benefits":
    # Health Benefits Page
    st.header("Why 'An Apple a Day Keeps the Doctor Away'")
    st.markdown("""
    The saying suggests that eating apples regularly can help maintain health and prevent illness. 
    The health benefits of apples include:
    - **Heart Health:** Rich in soluble fiber, which can lower cholesterol levels.
    - **Weight Management:** Low in calories and high in fiber, making them filling.
    - **Digestive Health:** The fiber content promotes healthy digestion.
    - **Reduced Risk of Chronic Diseases:** Antioxidants in apples can reduce the risk of diseases like diabetes and cancer.
    """)

elif choice == "Feedback":
    # Feedback Page
    st.header("User Feedback")
    
    # Feedback form
    st.subheader("We'd love to hear from you!")
    name = st.text_input("Your Name")
    rating = st.slider("Rate your experience:", 1, 5)
    feedback = st.text_area("Share your feedback:")
    
    if st.button("Submit Feedback"):
        if name and feedback:
            # Save feedback to session state
            if 'feedback' not in st.session_state:
                st.session_state.feedback = []
            st.session_state.feedback.append({
                "name": name,
                "rating": rating,
                "feedback": feedback
            })
            st.success("Thank you for your feedback!")
        else:
            st.error("Please fill out your name and feedback.")
    
    # Display previous feedback
    if 'feedback' in st.session_state and st.session_state.feedback:
        st.subheader("Previous Feedback:")
        for entry in st.session_state.feedback:
            st.markdown(f"**{entry['name']}** rated the app {entry['rating']} stars.")
            st.markdown(f"_\"{entry['feedback']}\"_")
