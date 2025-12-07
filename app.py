import streamlit as st
import numpy as np
import joblib

# Load model
model = joblib.load('score_model.pkl')

st.title("Student Score Prediction")

# Collect inputs
gender = st.selectbox("Gender", ["Male", "Female"])
attendance_rate = st.slider("Attendance Rate (%)", 0, 100)
study_hours_per_week = st.slider("Study Hours per Week", 0, 40)
previous_grade = st.slider("Previous Grade", 0, 100)
extra_curricular_activities = st.checkbox("Extracurricular Activities")
parental_support = st.selectbox("Parental Support", ["Low", "Medium", "High"])
study_hours = st.slider("Study Hours (Normalized)", 0.0, 1.0)
attendance_percent = st.slider("Attendance (%) (Normalized)", 0.0, 1.0)
online_classes = st.checkbox("Online Classes Taken")

# Encode inputs

gender = 0 if gender == "Male" else 1
support_map = {"Low": 0, "Medium": 1, "High": 2}
parental_support = support_map[parental_support]
online_classes = int(online_classes)

# Prepare input array
input_data = np.array([[gender, attendance_rate, study_hours_per_week, previous_grade, extra_curricular_activities,
                        parental_support, study_hours, attendance_percent, online_classes,]])

# Predict
if st.button("Predict Final Grade"):
    score = model.predict(input_data)[0]
    st.success(f"Predicted Final Grade: {score:.2f}")
    #streamlit run c:/Users/daveb/OneDrive/Desktop/projects/studentscoreprediction/app.py  (running line for code)