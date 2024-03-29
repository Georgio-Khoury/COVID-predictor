from flask import Flask, render_template, request, jsonify
from flask_cors import CORS
import pandas as pd
from sklearn.preprocessing import LabelEncoder
import joblib
app = Flask(__name__)
CORS(app)
# Load the trained model
model = joblib.load('trained_model.pkl')

@app.route('/')
def index():
    return "hello"

@app.route('/predict', methods=['POST'])
def predict():
   
    data = request.get_json()

   
    input_data = [data]

    
    input_df = pd.DataFrame(input_data)

  

   
    prediction = model.predict(input_df)


    prediction_list = prediction.tolist()
    return jsonify({'prediction': prediction_list})


if __name__ == '__main__':
    app.run(debug=True)
