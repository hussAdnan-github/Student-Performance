from pycaret.classification import *
import pandas as pd

data = pd.read_csv('studentData.csv')

# Set up the PyCaret environment

clf1 = setup(data, target='GradeClass', session_id=123 , 
                numeric_features=[
                    'Age', 'StudyTimeWeekly', 'Absences' 
                ] , 
                categorical_features=[
                    'Gender', 'Ethnicity', 'ParentalEducation',
                    'Tutoring' , 'ParentalSupport', 'Extracurricular', 'Sports',
                    'Music' , 'Volunteering' 
                ] , 
                ignore_features=[
                    'StudentID' , 'GPA'
                ])


# Compare different classification models

best_model = compare_models()

# save the best model
save_model(best_model, 'best_student_grade_model')

print("Model trained and saved successfully!")

create_api(best_model, 'student_grade_api')