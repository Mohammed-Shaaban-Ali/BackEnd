from flask import Flask, request, jsonify, send_file
from PIL import Image
from io import BytesIO
import joblib
import pandas as pd

app = Flask(__name__)

# Load the trained models
linear_model = joblib.load('linear_regression_model.pkl')
rf_model = joblib.load('random_forest_regressor_model.pkl')
dt_model = joblib.load('decision_tree_regressor_model.pkl')
svr_model = joblib.load('svr_model.pkl')

# Load the LabelEncoders
labelencoder = joblib.load('label_encoders.pkl')  # Dictionary of LabelEncoders

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
        total_mark_prediction_linear = linear_model.predict(student_df.drop(columns=[]))[0]
        total_mark_prediction_rf = rf_model.predict(student_df.drop(columns=[]))[0]
        total_mark_prediction_dt = dt_model.predict(student_df.drop(columns=[]))[0]
        total_mark_prediction_svr = svr_model.predict(student_df.drop(columns=[]))[0]

        # Return the predictions as JSON
        predictions = {
            "linear_regression": total_mark_prediction_linear,
            "random_forest_regressor": total_mark_prediction_rf,
            "decision_tree_regressor": total_mark_prediction_dt,
            "svr": total_mark_prediction_svr
        }

        return jsonify({"predictions": predictions})

    except Exception as e:
        return jsonify({"error": str(e)}), 500


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
        return jsonify({"error": str(e)}), 500

@app.route('/')
def welcome():
    return "Welcome to my Flask API!"



if __name__ == '__main__':
    try:
        app.run(debug=True, use_reloader=False, port=5001)
    except SystemExit:
        pass  # This prevents Spyder from exiting when Flask tries to restart