from transformers import pipeline
from PIL import Image
import streamlit as st

def AgeClassifier(imgFilename):
    age_classifier = pipeline("image-classification",
                               model="prithivMLmods/Age-Classification-SigLIP2")

    image_name = imgFilename
    image_name = Image.open(image_name).convert("RGB")

    # Classify age
    age_predictions = age_classifier(image_name)

    return age_predictions

# Load the age classification pipeline
def main():

    # Streamlit UI
    st.header("Title: Age Classification using ViT")

    age_predictions = AgeClassifier("middleageMan.jpg")
    
    st.write(age_predictions)
    age_predictions = sorted(age_predictions, key=lambda x: x['score'], reverse=True)
  
    # Display results
    st.write("Predicted Age Range:")
    st.write(f"Age range: {age_predictions[0]['label']}")
    st.write("Done")

if __name__ == "__main__":
    main()
