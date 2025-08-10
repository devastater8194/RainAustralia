# RainAustralia
a machine learning model that works on historical data provided by metereological  department of australia for rain from 2007 to 2017 ..and this model is trained on this data and it predicts that whether it will rain tomorrow or not and this model is 85 to 88 percent accurate 

🌧 Rain Prediction in Australia
This project uses historical weather data from the Meteorological Department of Australia (2007–2017) to predict whether it will rain tomorrow.

The dataset includes daily weather observations such as temperature, humidity, wind speed, and rainfall amounts. Using this data, we trained a Machine Learning classification model that achieves 85–88% accuracy.

📂 Project Overview
Goal: Predict "Rain Tomorrow" (Yes/No) based on historical weather patterns.

Data Source: Bureau of Meteorology, Australia (2007–2017).

Approach:

Perform Exploratory Data Analysis (EDA) to identify trends and patterns.

Preprocess the dataset (handling missing values, encoding categorical variables, scaling).

Train a classification model and evaluate performance.

📊 Technologies & Libraries Used
Python – Core programming language

NumPy – Numerical operations

Pandas – Data manipulation

Matplotlib – Data visualization

Seaborn – Statistical visualization

Plotly – Interactive charts

Scikit-learn – Machine learning algorithms and evaluation metrics

🔍 Exploratory Data Analysis (EDA)
Checked for missing values and performed imputation where necessary.

Studied correlations between weather features (e.g., humidity, temperature, pressure) and rainfall.

Created heatmaps, histograms, and scatter plots to visualize relationships.

🤖 Machine Learning Model
Type: Classification (Binary) – Predicts "Yes" or "No".

Features Used:

Max/Min Temperature

Rainfall

Humidity (morning/evening)

Wind speed/direction

Pressure

Cloud cover, sunshine hours, etc.

Accuracy: ~85–88%

Evaluation Metrics: Accuracy, Precision, Recall, F1-score
