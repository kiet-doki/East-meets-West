import pandas as pd
import pickle

# Load the trained model
with open('matching_model.pkl', 'rb') as model_file:
    model = pickle.load(model_file)

# Define matching function
def match_students(prospective_student, current_students):
    # Predict probabilities of being matched for each current student
    match_probabilities = model.predict_proba(current_students)

    # Sort current students based on match probabilities
    sorted_indices = match_probabilities[:, 1].argsort()[::-1]
    matched_students = current_students.iloc[sorted_indices[:10]]  # Select top 10 matches

    return matched_students
