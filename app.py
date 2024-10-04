import os
import pickle
import pandas as pd
import streamlit as st
from sklearn.linear_model import LinearRegression
from sklearn.naive_bayes import MultinomialNB
from sklearn.tree import DecisionTreeClassifier

# Function to load models from Pickle files
def load_model(model_name):
    with open(f'{model_name}', 'rb') as f:  # Remove 'STREAMLITTOO/' prefix
        return pickle.load(f)

# Load models
linear_regression_model = load_model('linear_regression.pkl')
logistic_regression_model = load_model('logistic_regression.pkl')
naive_bayes_model = load_model('naive_bayes.pkl')
apriori_model = load_model('apriori_car_data.pkl')
decision_tree_model = load_model('decision_tree_diabetes.pkl')

# UI with a sidebar to select models
st.sidebar.title("Model Selection")
model_option = st.sidebar.selectbox("Choose a model:", 
                                      ("Linear Regression", 
                                       "Logistic Regression", 
                                       "Naive Bayes", 
                                       "Apriori", 
                                       "Decision Tree"))

# Function to display evaluation metrics
def display_metrics(metrics):
    st.write("### Evaluation Metrics")
    for key, value in metrics.items():
        st.write(f"{key}: {value}")

# Linear Regression Model Fragment
if model_option == "Linear Regression":
    st.fragment()
    st.title("Linear Regression Model")
    st.write("Enter the features:")
    
    # User input for features
    area = st.number_input("Area")
    bedrooms = st.number_input("Bedrooms")
    bathrooms = st.number_input("Bathrooms")
    stories = st.number_input("Stories")
    mainroad = st.selectbox("Mainroad (0 or 1)", [0, 1])
    guestroom = st.selectbox("Guestroom (0 or 1)", [0, 1])
    basement = st.selectbox("Basement (0 or 1)", [0, 1])
    hotwaterheating = st.selectbox("Hot Water Heating (0 or 1)", [0, 1])
    airconditioning = st.selectbox("Air Conditioning (0 or 1)", [0, 1])
    parking = st.selectbox("Parking (0 or 1)", [0, 1])
    prefarea = st.selectbox("Preferred Area (0 or 1)", [0, 1])
    furnishingstatus = st.selectbox("Furnishing Status (0 or 1)", [0, 1])

    # Prediction button
    if st.button("Predict"):
        features = pd.DataFrame([[area, bedrooms, bathrooms, stories, mainroad, 
                                   guestroom, basement, hotwaterheating, 
                                   airconditioning, parking, prefarea, 
                                   furnishingstatus]], 
                                columns=['area', 'bedrooms', 'bathrooms', 'stories', 
                                         'mainroad', 'guestroom', 'basement', 
                                         'hotwaterheating', 'airconditioning', 
                                         'parking', 'prefarea', 'furnishingstatus'])
        price = linear_regression_model['model'].predict(features)
        st.write(f"Predicted Price: {price[0]}")
        display_metrics({'MSE': linear_regression_model['mse'], 'R²': linear_regression_model['r2']})

# Logistic Regression Model Fragment
elif model_option == "Logistic Regression":
    st.fragment()
    st.title("Logistic Regression Model")
    st.write("Enter the features:")
    
    sepal_length = st.number_input("Sepal Length")
    sepal_width = st.number_input("Sepal Width")
    petal_length = st.number_input("Petal Length")
    petal_width = st.number_input("Petal Width")

    # Prediction button
    if st.button("Predict"):
        features = pd.DataFrame([[sepal_length, sepal_width, petal_length, petal_width]], 
                                columns=['sepal_length', 'sepal_width', 
                                         'petal_length', 'petal_width'])
        species = logistic_regression_model['model'].predict(features)
        st.write(f"Predicted Species: {species[0]}")
        # Note: Add evaluation metrics if available

# Naive Bayes Model Fragment
elif model_option == "Naive Bayes":
    st.fragment()
    st.title("Naive Bayes Model")
    st.write("Enter the features:")
    
    # Assuming your features are similar to the email dataset
    features = []
    for col in ['the', 'to', 'ect', 'and', 'for', 'of', 'a', 'you', 'hou', 'connevey', 'jay', 
                'valued', 'lay', 'infrastructure', 'military', 'allowing', 'ff', 'dry']:
        features.append(st.number_input(f"{col}", 0))

    # Prediction button
    if st.button("Predict"):
        prediction = naive_bayes_model['model'].predict([features])
        st.write(f"Prediction: {prediction[0]}")
        # Note: Add evaluation metrics if available

# Apriori Model Fragment
elif model_option == "Apriori":
    st.fragment()
    st.title("Apriori Model")
    st.write("Displaying Frequent Itemsets and Association Rules")
    
    frequent_itemsets = apriori_model['frequent_itemsets']
    association_rules = apriori_model['association_rules']
    
    st.write("### Frequent Itemsets")
    st.write(frequent_itemsets)
    
    st.write("### Association Rules")
    st.write(association_rules)

# Decision Tree Model Fragment
elif model_option == "Decision Tree":
    st.fragment()
    st.title("Decision Tree Model")
    st.write("Enter the features:")
    
    pregnancies = st.number_input("Pregnancies")
    glucose = st.number_input("Glucose")
    blood_pressure = st.number_input("Blood Pressure")
    skin_thickness = st.number_input("Skin Thickness")
    insulin = st.number_input("Insulin")
    bmi = st.number_input("BMI")
    diabetes_pedigree_function = st.number_input("Diabetes Pedigree Function")
    age = st.number_input("Age")

    # Prediction button
    if st.button("Predict"):
        features = pd.DataFrame([[pregnancies, glucose, blood_pressure, skin_thickness, 
                                   insulin, bmi, diabetes_pedigree_function, age]], 
                                columns=['Pregnancies', 'Glucose', 'BloodPressure', 
                                         'SkinThickness', 'Insulin', 'BMI', 
                                         'DiabetesPedigreeFunction', 'Age'])
        outcome = decision_tree_model['model'].predict(features)
        st.write(f"Predicted Outcome: {outcome[0]}")
        # Note: Add evaluation metrics if available

# Footer
st.write("---")
st.write("Made with ❤️ using Streamlit and Shad-CN-UI")
