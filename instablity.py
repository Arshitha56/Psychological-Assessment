import streamlit as st
import numpy as np
from sklearn.ensemble import RandomForestClassifier

# Train the model (this is just an example using dummy data)
def train_model():
    # Example dataset with 8 features (age, gender, etc.)
    X_train = np.array([
        [23, 1, 1, 1, 1, 0, 0, 1],
        [45, 0, 0, 0, 1, 1, 1, 0],
        [34, 1, 1, 1, 1, 1, 0, 1],
        [56, 0, 0, 0, 1, 0, 1, 0],
        [40, 1, 0, 1, 0, 0, 0, 1],
        [29, 0, 1, 0, 1, 1, 1, 1],
        [60, 1, 0, 1, 1, 0, 1, 0],
        [50, 0, 1, 0, 0, 0, 1, 1]
    ])
    # Labels for the training set (1 means needs treatment, 0 means no treatment)
    y_train = np.array([1, 0, 1, 0, 0, 1, 0, 1])
    
    # Train a RandomForest model (you can use any model you prefer)
    model = RandomForestClassifier(n_estimators=100)
    model.fit(X_train, y_train)
    
    return model

# Function to make predictions with the trained model
def predict(model, features):
    final = [np.array(features)]
    prediction = model.predict_proba(final)
    output = '{0:.{1}f}'.format(prediction[0][1], 2)
    return float(output)

# Home Page
def home():
    st.title("Psychological State Assessment using Machine Learning")
    
    # Input form for features (using only 8 features)
    st.header("Please provide the following details:")
    
    age = st.number_input("Age", min_value=0, max_value=100, value=25)
    gender = st.selectbox("Gender", ["Male", "Female", "Other"])
    family_history = st.selectbox("Family History of Mental Illness", ["Yes", "No"])
    benefits = st.selectbox("Has health benefits?", ["Yes", "No"])
    care_options = st.selectbox("Has access to care options?", ["Yes", "No"])
    anonymity = st.selectbox("Does the person value anonymity?", ["Yes", "No"])
    leave = st.selectbox("Has the person taken leave for mental health?", ["Yes", "No"])
    work_interfere = st.selectbox("Does mental health interfere with work?", ["Yes", "No"])

    # Convert categorical inputs to numerical values
    gender = 1 if gender == "Male" else (0 if gender == "Female" else 2)
    family_history = 1 if family_history == "Yes" else 0
    benefits = 1 if benefits == "Yes" else 0
    care_options = 1 if care_options == "Yes" else 0
    anonymity = 1 if anonymity == "Yes" else 0
    leave = 1 if leave == "Yes" else 0
    work_interfere = 1 if work_interfere == "Yes" else 0

    # Create a feature array (only using the 8 features)
    features = [age, gender, family_history, benefits, care_options, anonymity, leave, work_interfere]
    
    # Train the model (this would typically be done once, not in every prediction)
    model = train_model()
    
    # Make prediction
    if st.button("Predict"):
        output = predict(model, features)
        
        if output > 0.5:
            st.write(f"You need treatment. Probability of mental illness is {output:.2f}")
            # Suggestion for treatment if probability is high
            st.subheader("Suggestions for Treatment:")
            st.write("""
                - Consider consulting a mental health professional such as a therapist or counselor.
                - Take time off from work if necessary and focus on self-care.
                - Reach out to supportive family and friends.
                - Practice mindfulness and relaxation techniques.
            """)
        else:
            st.write(f"You do not need treatment. Probability of mental illness is {output:.2f}")
            # Preventive suggestions if probability is low
            st.subheader("Preventive Measures and Self-Care Suggestions:")
            st.write("""
                - Maintain a healthy work-life balance.
                - Regular physical exercise to improve mental well-being.
                - Stay connected with loved ones and engage in social activities.
                - Practice stress-relief techniques like meditation or yoga.
            """)

# About Page
def about():
    st.title("About")
    st.write("""
    This is a mental health treatment prediction app. It uses a trained machine learning model to predict the likelihood of someone needing treatment for mental illness based on a set of personal features.
    
    **Features considered:**
    - Age
    - Gender
    - Family History of Mental Illness
    - Health Benefits
    - Access to Care Options
    - Value of Anonymity
    - History of Taking Leave
    - Impact on Work
    
    This model was trained on a dataset and can help in identifying individuals who may require mental health treatment. 
    """)

# Streamlit sidebar for navigation
st.sidebar.title("Navigation")
page = st.sidebar.radio("Go to", ["Home", "About"])

# Display the selected page
if page == "Home":
    home()
elif page == "About":
    about()
