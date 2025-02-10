from flask import Flask, request, jsonify, render_template
import pandas as pd
import numpy as np
import joblib
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.preprocessing import StandardScaler, LabelEncoder

# Custom LabelEncoder Transformer
class LabelEncoderTransformer(BaseEstimator, TransformerMixin):
    def __init__(self):
        self.encoders = {}

    def fit(self, X, y=None):
        for col in X.columns:
            le = LabelEncoder()
            le.fit(X[col])
            self.encoders[col] = le
        return self

    def transform(self, X):
        X_transformed = X.copy()
        for col in X.columns:
            X_transformed[col] = self.encoders[col].transform(X[col])
        return X_transformed

    def inverse_transform(self, X):
        X_inverse = X.copy()
        for col in X.columns:
            X_inverse[col] = self.encoders[col].inverse_transform(X[col])
        return X_inverse

app = Flask(__name__)

# Load dataset
df = pd.read_excel(r"D:\Project\Build-a-website-to-predict-laptop-prices\Laptops_dataset.xlsx")

# Extract unique values
unique_values = {
    'Ram': sorted(df['Ram'].unique().tolist()),
    'Memory': sorted(df['Memory'].unique().tolist()),
    'Size': sorted(df['Size'].unique().tolist()),
    'GPU_type': sorted(df['GPU_type'].unique().tolist()),
    'CPU_type': sorted(df['CPU_type'].unique().tolist())
}

# Load model and preprocessor
model = joblib.load(r"D:\Project\Build-a-website-to-predict-laptop-prices\best_model.pkl")
preprocessor = joblib.load(r"D:\Project\Build-a-website-to-predict-laptop-prices\preprocessor.pkl")

@app.route('/')
def index():
    return render_template('new_web.html')

@app.route('/form')
def form():
    # Extract unique values from dataset
    ram_values = sorted(df['Ram'].unique().tolist())
    memory_values = sorted(df['Memory'].unique().tolist())
    size_values = sorted(df['Size'].unique().tolist())
    gpu_values = sorted(df['GPU_type'].unique().tolist())
    cpu_values = sorted(df['CPU_type'].unique().tolist())
    
    return render_template('form_web.html',
                         ram_values=ram_values,
                         memory_values=memory_values,
                         size_values=size_values,
                         gpu_values=gpu_values,
                         cpu_values=cpu_values)

@app.route("/submit_form", methods=['POST'])
def prediction():
    if request.method == "POST":
        try:
            # Get form data
            input_data = {
                'Ram': int(request.form['ram']),
                'Memory': int(request.form['memory']),
                'Size': float(request.form['size']),
                'GPU_type': request.form['gpu_type'],
                'CPU_type': request.form['cpu_type']
            }
            
            # Create DataFrame and process
            input_df = pd.DataFrame([input_data])
            processed_features = preprocessor.transform(input_df)
            prediction = model.predict(processed_features)[0]
            
            # Find similar laptops in price range
            price_range = (prediction - 2000000, prediction + 2000000)
            similar_laptops = df[
                (df['Price'] >= price_range[0]) & 
                (df['Price'] <= price_range[1])
            ][['Name', 'Link', 'Price']].to_dict('records')
            
            return render_template('output_web.html',
                                ram=input_data['Ram'],
                                memory=input_data['Memory'],
                                size=input_data['Size'],
                                gpu_type=input_data['GPU_type'],
                                cpu_type=input_data['CPU_type'],
                                output=f"{prediction:,.0f}",
                                similar_laptops=similar_laptops)
        
        except Exception as e:
            return render_template('error.html', error=str(e))

@app.route("/api/predict", methods=["POST"])
def api_predict():
    try:
        data = request.get_json()
        input_data = {
            'Ram': int(data['ram']),
            'Memory': int(data['memory']),
            'Size': float(data['size']),
            'GPU_type': data['gpu_type'],
            'CPU_type': data['cpu_type']
        }
        
        # Create DataFrame with proper column order
        columns = ['Ram', 'Memory', 'Size', 'GPU_type', 'CPU_type']
        input_df = pd.DataFrame([input_data])[columns]
        
        processed_features = preprocessor.transform(input_df)
        prediction = model.predict(processed_features)[0]
        
        return jsonify({"price": float(prediction)})
    
    except Exception as e:
        return jsonify({"error": str(e)}), 400

if __name__ == "__main__":
    app.run(debug=True)

