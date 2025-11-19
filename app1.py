import streamlit as st
import re
import joblib

# ---------------------------
# Load Model + TFIDF
# ---------------------------
model = joblib.load('knn2.pkl')
tfidf = joblib.load('tfidf.pkl')

# ---------------------------
# Correct Label Mapping
# ---------------------------
label_mapping = {
    6: 'Data Science',
    12: 'HR',
    0: 'Advocate',
    1: 'Arts',
    24: 'Web Designing',
    16: 'Mechanical Engineer',
    22: 'Sales',
    14: 'Health and fitness',
    5: 'Civil Engineer',
    15: 'Java Developer',
    4: 'Business Analyst',
    21: 'SAP Developer',
    2: 'Automation Testing',
    11: 'Electrical Engineering',
    18: 'Operations Manager',
    20: 'Python Developer',
    8: 'DevOps Engineer',
    17: 'Network Security Engineer',
    19: 'PMO',
    7: 'Database',
    13: 'Hadoop',
    10: 'ETL Developer',
    9: 'DotNet Developer',
    3: 'Blockchain',
    23: 'Testing'
}  # <- Make sure this closing brace is present

# ---------------------------
# Text Preprocessing Function
# ---------------------------
def preprocess_text(text):
    """
    Preprocess resume text:
    - Lowercase
    - Remove special characters/numbers
    - Remove extra spaces
    """
    text = text.lower()
    text = re.sub(r'[^a-z\s]', '', text)
    text = re.sub(r'\s+', ' ', text).strip()
    return text

# ---------------------------
# Streamlit UI
# ---------------------------
st.title("Real Time Resume Screening Web App")
st.write("Paste your resume text below:")

resume_text = st.text_area("Resume Text", height=250)

if st.button("Predict Job Category"):
    if resume_text.strip() == "":
        st.warning("Please enter resume text before predicting.")
    else:
        # Preprocess
        clean_text = preprocess_text(resume_text)
        # Vectorize
        vector = tfidf.transform([clean_text])
        # Predict
        prediction = model.predict(vector)
        # Map label to job name
        job_name = label_mapping.get(prediction[0], "Unknown")
        # Show result
        st.success(f"Predicted Job Category: **{job_name}**")
