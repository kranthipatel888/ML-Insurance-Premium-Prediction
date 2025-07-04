from joblib import load
import pandas as pd

model_young=load("artifacts/model_young.joblib")
model_rest=load("artifacts/model_rest.joblib")
scaler_young=load("artifacts/scaler_young.joblib")
scaler_rest=load("artifacts/scaler_rest.joblib")

def calculate_medical_risk_score(medical_history_string):

    risk_score = {
        'diabetes': 6,
        'high blood pressure': 6,
        'no disease': 0,
        'thyroid': 5,
        'heart disease': 8,
        'none': 0 # Ensure 'none' explicitly maps to 0
    }

    total_score = 0

    # Handle None or empty strings by treating them as 'none'
    if medical_history_string is None or medical_history_string.strip() == '':
        processed_string = 'none'
    else:
        # Convert to lowercase and strip whitespace for consistent matching
        processed_string = medical_history_string.strip().lower()

    # Split the string into individual conditions
    # This will return a list of conditions (e.g., ['diabetes', 'high blood pressure'])
    conditions = processed_string.split(" & ")

    # Iterate through each condition and add its score
    for condition in conditions:
        # .get(key, default_value) allows safe lookup: returns the score if found,
        # otherwise returns 0 if the condition is not in risk_score.
        score = risk_score.get(condition, 0)
        total_score += score

    return total_score


def input_preprocessing(input_dict):
    Expected_columns=['Age', 'Number Of Dependants', 'BMI_Category',
       'Income_Lakhs', 'Insurance_Plan', 'Genetical_Risk', 'Gender_Male',
       'Region_Northwest', 'Region_Southeast', 'Region_Southwest',
       'Smoking_Status_Occasional', 'Smoking_Status_Regular',
       'Employment_Status_Salaried', 'Employment_Status_Self-Employed',
       'risk_norm']

    BMI_Category_encoding = {'Normal': 2, 'Obesity': 4, 'Overweight': 3, 'Underweight': 1}
    Insurance_Plan_encoding = {'Bronze': 1, 'Silver': 2, 'Gold': 3}

    df=pd.DataFrame(0,columns=Expected_columns,index=[0])

    df["Income_Level"]=0

    for key,value in input_dict.items():
        if key=='BMI Category':
            df["BMI_Category"]=BMI_Category_encoding[value]
        elif key=='Insurance Plan':
            df["Insurance_Plan"]=Insurance_Plan_encoding[value]
        elif key=='Age':
            df["Age"]=value
        elif key=="Gender":
            if value=="Male":
                df["Gender_Male"] = 1
        elif key=='Region':
            if value=="Northwest":
                df['Region_Northwest']=1
            if value=="Southeast":
                df['Region_Southeast']=1
            if value=="Southwest":
                df['Region_Southwest']=1
        elif key=='Smoking Status':
            if value=="Regular":
                df['Smoking_Status_Regular']=1
            if value=="Occasional":
                df['Smoking_Status_Occasional']=1
        elif key=="Employment Status":
            if value=='Salaried':
                df['Employment_Status_Salaried']=1
            if value=='Self':
                df["Employment_Status_Self"]=1
        elif key=="Genetical Risk":
            df["Genetical_Risk"]=value
        elif key=='Number Of Dependants':
            df["Number Of Dependants"]=value
        elif key=="Income in Lakhs":
            df["Income_Lakhs"]=value

    df["risk_norm"]=calculate_medical_risk_score(input_dict['Medical History'])
    df=handle_scaling(input_dict["Age"],df)
    return df

def handle_scaling(age,df):
    if age>25:
        scaler_object=scaler_rest
    else:
        scaler_object=scaler_young
    cols_to_scale=scaler_object["cols_to_scale"]
    scaler=scaler_object["scaler"]
    df[cols_to_scale]=scaler.transform(df[cols_to_scale])

    df.drop('Income_Level',axis=1,inplace=True)
    return df

def predict(input_dict):
    input_df=input_preprocessing(input_dict)
    if input_dict["Age"]>25:
        prediction=model_rest.predict(input_df)
    else:
        prediction = model_young.predict(input_df)

    return int(prediction[0])