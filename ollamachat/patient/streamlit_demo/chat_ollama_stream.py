import streamlit as st
import json
import ollama

class OllaChat:
    def __init__(self):
        self.model = 'llama3.2:3b' 

    def chat_llama(self, user_input):
        st.write("Loading model...")
        prompt = f"""
        You are a medical data extraction assistant. Your task is to analyze free-text patient information and extract specific details. For each input, provide a Python dictionary containing the following fields:

        1. `name` - Full name of the patient.
        2. `gender` - Gender of the patient (e.g., male, female, other).
        3. `age` - Age of the patient (integer in years).
        4. `weight` - Weight of the patient in kilograms (kg).
        5. `height` - Height of the patient in centimeters (cm).
        6. `BMI` - Body Mass Index (BMI) calculated using the formula: BMI = weight (kg) / (height (m))^2.
        7. `chief_medical_complaint` - The primary medical complaint or reason for the patient's visit.

        Ensure the output is a well-formed Python dictionary containing **only** these seven fields. Fill in missing fields with `null` if the information is unavailable in the input text.
        Below is the user_input to be considered to extract the data:
        {user_input}
        """

        response = ollama.chat(model=self.model, format='json', messages=[
            {
                'role': 'user',
                'content': prompt,
            },
        ])
        
        result = response['message']['content']
        result_json = json.loads(result)  
        return result_json


def main():
    st.title("Medical Data Extraction from Free Text")
    
    user_input = st.text_area("Enter patient information (free text):", height=250)

    if st.button("Extract Information"):
        if user_input.strip():
            st.write("Processing data...")
            oc = OllaChat()
            result = oc.chat_llama(user_input=user_input)

            if result:
                st.subheader("Extracted Information:")
                st.json(result)  
            else:
                st.warning("No information extracted. Please try again.")
        else:
            st.warning("Please provide the patient information.")

if __name__ == "__main__":
    main()
