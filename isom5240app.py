from transformers import pipeline
from PIL import Image
import streamlit as st

# Streamlit UI
st.header("Title: Age Classification using ViT")

# Load the age classification pipeline
def imgclassification():
    age_classifier = pipeline("image-classification",
                               model="prithivMLmods/Age-Classification-SigLIP2")

    image_name = "middleagedMan.jpg"
    image_name = Image.open(image_name).convert("RGB")

    # Classify age
    age_predictions = age_classifier(image_name)
    st.write(age_predictions)
    age_predictions = sorted(age_predictions, key=lambda x: x['score'], reverse=True)

    return age_predictions[0]['label']
  
# Display results

def main():
    predicted_age = imgclassifcation()  
    st.write("Predicted Age Range:")
    st.write(f"Age range: {predicted_age}")
    st.write("Done")

if __name__ == "__main__":
    main()
