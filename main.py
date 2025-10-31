from flask import Flask, request, jsonify, render_template
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, FunctionTransformer
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import joblib

# Initialize the Flask app
app = Flask(__name__)

# Define the preprocessor and model pipeline
skewed_features = ['Area', 'Production', 'Annual_Rainfall', 'Fertilizer', 'Pesticide']

def log_transform(x):
    return np.log1p(x)

preprocessor = ColumnTransformer(
    transformers=[
        ('num', Pipeline(steps=[
            ('log', FunctionTransformer(log_transform, validate=True)),
            ('scaler', StandardScaler())
        ]), skewed_features),
        ('cat', OneHotEncoder(handle_unknown='ignore'), ['Crop', 'Season', 'State'])
    ])

pipeline_rf = Pipeline(steps=[
    ('preprocessor', preprocessor),
    ('regressor', RandomForestRegressor(random_state=42))
])

# Load dataset and train the model
df = pd.read_csv('crop_yield.csv')
X = df.drop('Yield', axis=1)
y = df['Yield']

# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)


# Hyperparameter tuning
param_grid_rf = {
    'regressor__n_estimators': [100, 200],
    'regressor__max_depth': [10, 20, None]
}

grid_search_rf = GridSearchCV(pipeline_rf, param_grid_rf, cv=3, scoring='r2', n_jobs=-1)
grid_search_rf.fit(X_train, y_train)

# Save the trained model
model = grid_search_rf.best_estimator_
joblib.dump(model, 'crop_yield_model.pkl')

# Load the model
model = joblib.load('crop_yield_model.pkl')

# Define the route for the home page
@app.route('/')
def home():
    return render_template('index.html')


# Define the route to handle prediction requests
@app.route('/predict', methods=['POST'])
def predict():
    data = request.get_json()

    # Convert data into DataFrame
    input_data = pd.DataFrame(data, index=[0])
  

    # Make prediction
    
    prediction = model.predict(input_data)

    return jsonify({'predicted_yield': prediction[0]})

if __name__ == "__main__":
    app.run(debug=True)


