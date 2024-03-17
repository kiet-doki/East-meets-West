from flask import Flask, request, jsonify
import pandas as pd
from data_processing import preprocess_data
from matching_algorithm import match_students

app = Flask(__name__)

# Load data
prospective_students_data = pd.read_csv('data/prospective_students.csv')
current_students_data = pd.read_csv('data/current_students.csv')

@app.route('/match', methods=['POST'])
def match():
    data = request.get_json()
    prospective_student = pd.DataFrame(data)
    prospective_student_processed = preprocess_data(prospective_student)
    matches = match_students(prospective_student_processed, current_students_data)
    return jsonify(matches.to_dict(orient='records'))

if __name__ == '__main__':
    app.run(debug=True)
