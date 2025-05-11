import streamlit as st
import pandas as pd
import pickle

with open('best_model.pkl', 'rb') as f:
    model = pickle.load(f)

with open('encoder.pkl', 'rb') as f:
    encoder = pickle.load(f)


def predict(Age, Workclass, Final_Weight, Education, EducationNum, Marital_Status, Occupation, Relationship,
            Race, Gender, Capital_Gain, capital_loss, Hours_per_Week, Native_Country):
    
    # Ubah Gender ke numerik terlebih dahulu
    Gender_num = 0 if Gender == 'Male' else 1

    # Masukkan Gender yang sudah di-encode
    input_df = pd.DataFrame({
        'Age': [Age],
        'Workclass': [Workclass],
        'Final Weight': [Final_Weight],
        'Education': [Education],
        'EducationNum': [EducationNum],
        'Marital Status': [Marital_Status],
        'Occupation': [Occupation],
        'Relationship': [Relationship],
        'Race': [Race],
        'Capital Gain': [Capital_Gain],
        'Capital Loss': [capital_loss],
        'Hours per Week': [Hours_per_Week],
        'Native Country': [Native_Country],
        'Gender': [Gender_num]  # Sudah dalam bentuk angka
    })

    # Kolom numerik & kategorikal
    cat_cols = ['Workclass', 'Education', 'Marital Status', 'Occupation', 'Relationship', 'Race', 'Native Country']
    num_cols = ['Age', 'Final Weight', 'EducationNum', 'Capital Gain', 'Capital Loss', 'Hours per Week', 'Gender']

    # Transformasi
    input_cat = encoder.transform(input_df[cat_cols]).toarray()
    input_num = input_df[num_cols].astype(float).values

    import numpy as np
    input_encoded = np.concatenate([input_num, input_cat], axis=1)

    if input_encoded.shape[1] != model.n_features_in_:
        raise ValueError(f"Jumlah fitur input ({input_encoded.shape[1]}) tidak sesuai dengan jumlah fitur yang diharapkan oleh model ({model.n_features_in_})")

    prediction = model.predict(input_encoded)
    return '>50K' if prediction == 1 else '<=50K'


def run_ml_app():
    st.title("Income Classification App 💰")
    st.write("Predict whether someone has income >50K or <=50K based on demographic data")

    
    Age = st.slider("Age", 17, 90, 30)
    Workclass = st.selectbox("Workclass", ['Private', 'Self-emp-not-inc', 'Self-emp-inc',
                                           'Federal-gov', 'Local-gov', 'State-gov', 'Without-pay', 'Never-worked'])
    Final_Weight = st.number_input("Final Weight (fnlwgt)", min_value=10000, max_value=1000000, value=50000)
    Education = st.selectbox("Education", ['Bachelors', 'HS-grad', '11th', 'Masters', '9th', 'Some-college',
                                           'Assoc-acdm', 'Assoc-voc', '7th-8th', 'Doctorate', 'Prof-school',
                                           '5th-6th', '10th', '1st-4th', 'Preschool', '12th'])
    EducationNum = st.slider("Education Number", 1, 16, 10)
    Marital_Status = st.selectbox("Marital Status", ['Married-civ-spouse', 'Divorced', 'Never-married',
                                                     'Separated', 'Widowed', 'Married-spouse-absent', 'Married-AF-spouse'])
    Occupation = st.selectbox("Occupation", ['Tech-support', 'Craft-repair', 'Other-service', 'Sales',
                                             'Exec-managerial', 'Prof-specialty', 'Handlers-cleaners',
                                             'Machine-op-inspct', 'Adm-clerical', 'Farming-fishing',
                                             'Transport-moving', 'Priv-house-serv', 'Protective-serv', 'Armed-Forces'])
    Relationship = st.selectbox("Relationship", ['Wife', 'Own-child', 'Husband', 'Not-in-family',
                                                 'Other-relative', 'Unmarried'])
    Race = st.selectbox("Race", ['White', 'Asian-Pac-Islander', 'Amer-Indian-Eskimo', 'Other', 'Black'])
    Gender = st.selectbox("Gender", ['Male', 'Female'])
    Capital_Gain = st.number_input("Capital Gain", 0, 99999, 0)
    capital_loss = st.number_input("Capital Loss", 0, 99999, 0)
    Hours_per_Week = st.slider("Hours per Week", 1, 100, 40)
    Native_Country = st.selectbox("Native Country", ['United-States', 'Cambodia', 'England', 'Puerto-Rico', 'Canada',
                                                     'Germany', 'Outlying-US(Guam-USVI-etc)', 'India', 'Japan', 'Greece',
                                                     'South', 'China', 'Cuba', 'Iran', 'Honduras', 'Philippines',
                                                     'Italy', 'Poland', 'Jamaica', 'Vietnam', 'Mexico', 'Portugal',
                                                     'Ireland', 'France', 'Dominican-Republic', 'Laos', 'Ecuador',
                                                     'Taiwan', 'Haiti', 'Columbia', 'Hungary', 'Guatemala', 'Nicaragua',
                                                     'Scotland', 'Thailand', 'Yugoslavia', 'El-Salvador', 'Trinadad&Tobago',
                                                     'Peru', 'Hong', 'Holand-Netherlands'])

    button = st.button("Predict")

    if button:
        result = predict(Age, Workclass, Final_Weight, Education,EducationNum, Marital_Status, Occupation, Relationship,
                         Race, Gender, Capital_Gain, capital_loss, Hours_per_Week, Native_Country)
        
        if result == ">50K":
            st.success("Prediction: Income >50K")
        else:
            st.warning("Prediction: Income <=50K")

# Menjalankan aplikasi
if __name__ == '__main__':
    run_ml_app()