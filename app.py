import pickle
import numpy as np
from flask import Flask, request, render_template_string

app = Flask(__name__)

# Load the trained logistic regression model
with open('linear.pkl', 'rb') as f:
    model = pickle.load(f)

# HTML & CSS Template combined into a single responsive layout
HTML_TEMPLATE = """
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Student Performance Risk Predictor</title>
    <style>
        :root {
            --primary: #4f46e5;
            --primary-hover: #4338ca;
            --bg-color: #f8fafc;
            --card-bg: #ffffff;
            --text-color: #1e293b;
            --border-color: #cbd5e1;
        }
        body {
            font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
            background-color: var(--bg-color);
            color: var(--text-color);
            margin: 0;
            padding: 20px;
            display: flex;
            justify-content: center;
            align-items: center;
            min-height: 100vh;
        }
        .container {
            background: var(--card-bg);
            padding: 30px 40px;
            border-radius: 12px;
            box-shadow: 0 4px 20px rgba(0, 0, 0, 0.08);
            width: 100%;
            max-width: 650px;
        }
        h2 {
            text-align: center;
            color: var(--primary);
            margin-bottom: 25px;
        }
        .form-grid {
            display: grid;
            grid-template-columns: 1fr 1fr;
            gap: 15px;
        }
        .form-group {
            margin-bottom: 12px;
        }
        .form-group.full-width {
            grid-column: span 2;
        }
        label {
            display: block;
            margin-bottom: 5px;
            font-weight: 600;
            font-size: 14px;
        }
        input, select {
            width: 100%;
            padding: 10px;
            border: 1px solid var(--border-color);
            border-radius: 6px;
            font-size: 14px;
            box-sizing: border-box;
            transition: border-color 0.2s;
        }
        input:focus, select:focus {
            outline: none;
            border-color: var(--primary);
            box-shadow: 0 0 0 3px rgba(79, 70, 229, 0.1);
        }
        button {
            width: 100%;
            padding: 12px;
            background-color: var(--primary);
            color: white;
            border: none;
            border-radius: 6px;
            font-size: 16px;
            font-weight: 600;
            cursor: pointer;
            margin-top: 15px;
            transition: background-color 0.2s;
        }
        button:hover {
            background-color: var(--primary-hover);
        }
        .result-card {
            margin-top: 25px;
            padding: 15px;
            border-radius: 8px;
            text-align: center;
            font-size: 18px;
            font-weight: bold;
        }
        .Safe { background-color: #d1fae5; color: #065f46; border: 1px solid #10b981; }
        .At-Risk { background-color: #fef3c7; color: #92400e; border: 1px solid #f59e0b; }
        .High-Risk { background-color: #fee2e2; color: #991b1b; border: 1px solid #ef4444; }
        @media (max-width: 600px) {
            .form-grid { grid-template-columns: 1fr; }
            .form-group.full-width { grid-column: span 1; }
        }
    </style>
</head>
<body>
    <div class="container">
        <h2>Student Risk Assessment</h2>
        <form method="POST">
            <div class="form-grid">
                <div class="form-group">
                    <label for="attendance">Attendance (%)</label>
                    <input type="number" step="0.1" id="attendance" name="attendance" required min="0" max="100" placeholder="e.g. 85">
                </div>
                <div class="form-group">
                    <label for="study_hours">Study Hours (per week)</label>
                    <input type="number" step="0.1" id="study_hours" name="study_hours" required min="0" placeholder="e.g. 15">
                </div>
                <div class="form-group">
                    <label for="past_failures">Past Failures</label>
                    <input type="number" id="past_failures" name="past_failures" required min="0" placeholder="e.g. 0">
                </div>
                <div class="form-group">
                    <label for="assignments_completed_pct">Assignments Completed (%)</label>
                    <input type="number" step="0.1" id="assignments_completed_pct" name="assignments_completed_pct" required min="0" max="100" placeholder="e.g. 90">
                </div>
                
                <!-- Categorical Fields -->
                <div class="form-group">
                    <label for="parental_education">Parental Education</label>
                    <select id="parental_education" name="parental_education" required>
                        <option value="0">High School</option>
                        <option value="1">Associate / Diploma</option>
                        <option value="2">Bachelor's Degree</option>
                        <option value="3">Master's / Higher</option>
                    </select>
                </div>
                <div class="form-group">
                    <label for="family_income">Family Income Level</label>
                    <select id="family_income" name="family_income" required>
                        <option value="0">Low</option>
                        <option value="1">Medium</option>
                        <option value="2">High</option>
                    </select>
                </div>
                <div class="form-group">
                    <label for="extracurricular">Extracurricular Activities</label>
                    <select id="extracurricular" name="extracurricular" required>
                        <option value="0">No</option>
                        <option value="1">Yes</option>
                    </select>
                </div>
                <div class="form-group">
                    <label for="internet_access">Internet Access at Home</label>
                    <select id="internet_access" name="internet_access" required>
                        <option value="0">No</option>
                        <option value="1">Yes</option>
                    </select>
                </div>

                <div class="form-group">
                    <label for="previous_grade">Previous Grade (%)</label>
                    <input type="number" step="0.1" id="previous_grade" name="previous_grade" required min="0" max="100" placeholder="e.g. 78">
                </div>
                <div class="form-group">
                    <label for="final_score">Current Final Score (%)</label>
                    <input type="number" step="0.1" id="final_score" name="final_score" required min="0" max="100" placeholder="e.g. 82">
                </div>
            </div>
            
            <button type="submit">Predict Status</button>
        </form>

        {% if prediction %}
        <div class="result-card {{ prediction }}">
            Predicted Status: {{ prediction }}
        </div>
        {% endif %}
    </div>
</body>
</html>
"""

@app.route('/', methods=['GET', 'POST'])
def home():
    prediction = None
    if request.method == 'POST':
        try:
            # Extract features in the exact order expected by the model
            features = [
                float(request.form['attendance']),
                float(request.form['study_hours']),
                float(request.form['past_failures']),
                float(request.form['assignments_completed_pct']),
                float(request.form['parental_education']),
                float(request.form['family_income']),
                float(request.form['extracurricular']),
                float(request.form['internet_access']),
                float(request.form['previous_grade']),
                float(request.form['final_score'])
            ]
            
            # Reshape into a 2D array for prediction
            input_data = np.array([features])
            prediction = model.predict(input_data)[0]
        except Exception as e:
            prediction = f"Error in processing input: {str(e)}"
            
    return render_template_string(HTML_TEMPLATE, prediction=prediction)

if __name__ == '__main__':
    app.run(debug=True)
