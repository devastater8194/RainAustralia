import pandas as pd
import numpy as np
from flask import Flask, request, jsonify
from flask_cors import CORS # To handle Cross-Origin Resource Sharing

from sklearn.impute import SimpleImputer
from sklearn.preprocessing import MinMaxScaler, OneHotEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split # Import for data splitting
from sklearn.metrics import accuracy_score # Import for accuracy calculation

# --- Flask App Setup ---
app = Flask(__name__)
CORS(app) # Enable CORS for all routes, allowing frontend to access

# --- Global variables for model and preprocessors ---
imputer = None
scaler = None
encoder = None
model = None
numeric_cols = []
categorical_cols = []
encoded_cols = []
MODEL_FEATURES = []
test_inputs_processed = None # To store preprocessed test inputs
test_targets_numeric = None  # To store numeric test targets
model_initialized = False # Flag to track if the model was successfully initialized

# --- Data Loading, Preprocessing, and Model Training (runs once on app startup) ---
def train_and_load_model():
    global imputer, scaler, encoder, model, numeric_cols, categorical_cols, encoded_cols, MODEL_FEATURES, test_inputs_processed, test_targets_numeric, model_initialized

    try:
        raw_df = pd.read_csv('weatherAUS.csv')
        print("weatherAUS.csv loaded successfully.")
    except FileNotFoundError:
        print("Error: 'weatherAUS.csv' not found. Model will not be initialized.")
        print("Please ensure 'weatherAUS.csv' is in the same directory as app.py for full functionality.")
        model_initialized = False
        return # Exit the function if data is not found

    raw_df.dropna(subset=['RainToday', 'RainTomorrow'], inplace=True)

    # Define input and target columns
    input_cols = list(raw_df.columns)[1:-1] # Exclude 'Date' and 'RainTomorrow'
    target_col = 'RainTomorrow'

    # Split data into training and testing sets first
    # Using a fixed random_state for reproducibility
    # test_size=0.2 means 20% of data for testing
    train_df, test_df = train_test_split(raw_df, test_size=0.2, random_state=42, stratify=raw_df[target_col])

    print(f"Training data shape: {train_df.shape}")
    print(f"Testing data shape: {test_df.shape}")

    # Identify numeric and categorical columns based on the full raw_df for consistency
    numeric_cols.extend(raw_df[input_cols].select_dtypes(include=np.number).columns.tolist())
    categorical_cols.extend(raw_df[input_cols].select_dtypes('object').columns.tolist())

    print(f"Numeric Columns: {numeric_cols}")
    print(f"Categorical Columns: {categorical_cols}")

    # --- Pre-Imputation Handling for Entirely NaN Numeric Columns (on train_df) ---
    for col in numeric_cols:
        if train_df[col].isnull().all():
            print(f"Warning: Column '{col}' is entirely NaN in training data. Filling with 0 for imputation.")
            train_df[col].fillna(0, inplace=True)
        # Also apply to test_df to ensure consistency before imputation
        if test_df[col].isnull().all():
            test_df[col].fillna(0, inplace=True)


    # --- Imputation (fit on train_df, transform both) ---
    imputer = SimpleImputer(strategy='mean')
    imputer.fit(train_df[numeric_cols]) # Fit ONLY on training data

    train_df[numeric_cols] = imputer.transform(train_df[numeric_cols])
    test_df[numeric_cols] = imputer.transform(test_df[numeric_cols]) # Transform test data


    # --- Scaling (fit on train_df, transform both) ---
    scaler = MinMaxScaler()
    scaler.fit(train_df[numeric_cols]) # Fit ONLY on training data

    train_df[numeric_cols] = scaler.transform(train_df[numeric_cols])
    test_df[numeric_cols] = scaler.transform(test_df[numeric_cols]) # Transform test data


    # --- One-Hot Encoding (fit on train_df, transform both) ---
    encoder = OneHotEncoder(sparse_output=False, handle_unknown='ignore')
    encoder.fit(train_df[categorical_cols]) # Fit ONLY on training data

    encoded_cols.extend(list(encoder.get_feature_names_out(categorical_cols)))
    print(f"Expected encoded columns: {len(encoded_cols)}")

    # Transform categorical columns for both train and test
    transformed_train_categorical = encoder.transform(train_df[categorical_cols])
    transformed_test_categorical = encoder.transform(test_df[categorical_cols])
    
    # Create DataFrames from transformed arrays
    encoded_train_df = pd.DataFrame(transformed_train_categorical, columns=encoded_cols, index=train_df.index)
    encoded_test_df = pd.DataFrame(transformed_test_categorical, columns=encoded_cols, index=test_df.index)

    # Drop original categorical columns and concatenate encoded ones
    train_df = train_df.drop(columns=categorical_cols)
    train_df = pd.concat([train_df, encoded_train_df], axis=1)

    test_df = test_df.drop(columns=categorical_cols)
    test_df = pd.concat([test_df, encoded_test_df], axis=1)

    # --- Prepare data for model training and testing ---
    X_train = train_df[numeric_cols + encoded_cols]
    y_train = train_df[target_col]
    X_test = test_df[numeric_cols + encoded_cols]
    y_test = test_df[target_col]

    # Convert targets to numerical (Yes/No to 1/0)
    y_train_numeric = y_train.map({'Yes': 1, 'No': 0})
    y_test_numeric = y_test.map({'Yes': 1, 'No': 0})

    # --- Model Training ---
    model = LogisticRegression(solver='liblinear', random_state=42)
    model.fit(X_train, y_train_numeric)

    # Store preprocessed test data and targets globally for accuracy endpoint
    test_inputs_processed = X_test
    test_targets_numeric = y_test_numeric

    # Define the final list of features the model expects
    MODEL_FEATURES.extend(numeric_cols + encoded_cols)

    print("Model trained and loaded successfully into memory!")
    model_initialized = True


# Call the training function when the app starts
with app.app_context():
    train_and_load_model()


@app.route('/predict', methods=['POST'])
def predict():
    """
    API endpoint to receive weather data and return rain prediction.
    """
    if not model_initialized:
        return jsonify({"error": "Model not initialized. 'weatherAUS.csv' might be missing or an error occurred during training."}), 500

    if not request.is_json:
        return jsonify({"error": "Request must be JSON"}), 400

    data = request.get_json()
    
    # Convert input dictionary to a DataFrame for preprocessing
    input_df = pd.DataFrame([data])
    
    # Ensure all columns expected by the model are present in the input_df
    # and fill missing ones with NaN. This is crucial for consistent preprocessing.
    for col in numeric_cols + categorical_cols:
        if col not in input_df.columns:
            input_df[col] = np.nan
    
    # Pre-imputation handling for entirely NaN numeric columns in input_df
    for col in numeric_cols:
        if input_df[col].isnull().all():
            input_df[col].fillna(0, inplace=True) # Fill with 0 or another sensible default

    try:
        # Apply imputation (only on numeric columns)
        input_df[numeric_cols] = imputer.transform(input_df[numeric_cols])

        # Apply scaling (only on numeric columns)
        input_df[numeric_cols] = scaler.transform(input_df[numeric_cols])

        # Apply one-hot encoding (only on categorical columns)
        encoded_data = encoder.transform(input_df[categorical_cols])
        encoded_df = pd.DataFrame(encoded_data, columns=encoded_cols, index=input_df.index)

        # Combine numeric and encoded features
        X_processed = pd.concat([input_df[numeric_cols], encoded_df], axis=1)

        # Ensure the order of columns in X_processed matches MODEL_FEATURES
        X_processed = X_processed[MODEL_FEATURES]

        # Make prediction
        prediction_numeric = model.predict(X_processed)[0]
        prediction_label = 'Yes' if prediction_numeric == 1 else 'No'

        # Get probabilities
        probabilities = model.predict_proba(X_processed)[0]
        prob_no_rain = probabilities[0] # Probability of 'No'
        prob_yes_rain = probabilities[1] # Probability of 'Yes'

        return jsonify({
            'prediction': prediction_label,
            'probability_no_rain': round(prob_no_rain * 100, 2),
            'probability_yes_rain': round(prob_yes_rain * 100, 2)
        })

    except Exception as e:
        print(f"Prediction error: {e}")
        return jsonify({"error": str(e)}), 500

@app.route('/accuracy', methods=['GET'])
def get_accuracy():
    """
    API endpoint to return the model's accuracy on the test set.
    """
    global test_inputs_processed, test_targets_numeric, model_initialized

    if not model_initialized or test_inputs_processed is None or test_targets_numeric is None:
        return jsonify({"error": "Model not initialized or test data not prepared."}), 500

    try:
        test_predictions = model.predict(test_inputs_processed)
        accuracy = accuracy_score(test_targets_numeric, test_predictions)
        return jsonify({"accuracy": round(accuracy * 100, 2)})
    except Exception as e:
        print(f"Accuracy calculation error: {e}")
        return jsonify({"error": f"Failed to calculate accuracy: {e}"}), 500


if __name__ == '__main__':
    # Run the Flask app
    app.run(host='0.0.0.0', port=5000) # Listen on all interfaces
