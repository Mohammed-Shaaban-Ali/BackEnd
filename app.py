import os
from flask import Flask, request, jsonify, send_file
from PIL import Image
from io import BytesIO
import joblib
import pandas as pd
import traceback

app = Flask(__name__)

# Helper function to load models
def load_model(filename):
    model_path = os.path.join(os.path.dirname(__file__), 'models', filename)
    return joblib.load(model_path)

# Load the trained models
linear_model = load_model('linear_regression_model.pkl')
rf_model = load_model('random_forest_regressor_model.pkl')
dt_model = load_model('decision_tree_regressor_model.pkl')
svr_model = load_model('svr_model.pkl')

# Load the LabelEncoders
labelencoder = load_model('label_encoders.pkl')  # Dictionary of LabelEncoders

# Define the endpoint for predictions
@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Get the data from the request
        data = request.json
        
        # Convert incoming JSON data to a pandas DataFrame
        student_df = pd.DataFrame([data])
        
        # Transform categorical features in the student data using the same LabelEncoder
        for column in student_df.select_dtypes(include=['object']).columns:
            if column in labelencoder:
                student_df[column] = labelencoder[column].transform(student_df[column])
            else:
                return jsonify({"error": f"LabelEncoder for column '{column}' not found."}), 400
        
        # Predict TOTAL_MARK for the new student using all models
        total_mark_prediction_linear = linear_model.predict(student_df)[0]
        total_mark_prediction_rf = rf_model.predict(student_df)[0]
        total_mark_prediction_dt = dt_model.predict(student_df)[0]
        total_mark_prediction_svr = svr_model.predict(student_df)[0]

        # Return the predictions as JSON
        predictions = {
            "linear_regression": float(total_mark_prediction_linear),
            "random_forest_regressor": float(total_mark_prediction_rf),
            "decision_tree_regressor": float(total_mark_prediction_dt),
            "svr": float(total_mark_prediction_svr)
        }

        return jsonify({"predictions": predictions})

    except Exception as e:
        return jsonify({"error": str(e), "trace": traceback.format_exc()}), 500

@app.route('/process_image', methods=['POST'])
def process_image():
    try:
        # Get the image from the request
        if 'file' not in request.files:
            return jsonify({"error": "No file part"}), 400
        
        file = request.files['file']
        if file.filename == '':
            return jsonify({"error": "No selected file"}), 400
        
        # Open the image file
        image = Image.open(file.stream).convert('RGB')

        # Convert the image to grayscale
        grayscale_image = image.convert('L')

        # Save the grayscale image to a BytesIO object
        img_byte_arr = BytesIO()
        grayscale_image.save(img_byte_arr, format='PNG')
        img_byte_arr.seek(0)

        return send_file(img_byte_arr, mimetype='image/png', as_attachment=True, download_name='grayscale_image.png')

    except Exception as e:
        return jsonify({"error": str(e), "trace": traceback.format_exc()}), 500

@app.route('/')
def welcome():
    return "Welcome to my Flask API!"

@app.errorhandler(500)
def internal_server_error(error):
    return jsonify({"error": "Internal Server Error", "details": str(error), "trace": traceback.format_exc()}), 500

# Note: Remove the if __name__ == '__main__': block for Vercel deployment