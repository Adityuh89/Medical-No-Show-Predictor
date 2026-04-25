# Medical-No-Show-Predictor
🏥 Medical No-Show Prediction System
📌 Overview

This project is an end-to-end Machine Learning system designed to predict whether a patient will miss a scheduled medical appointment (“no-show”). It is built on a real-world dataset containing 49,593 records (2016–2022) from a rehabilitation center in Southern Brazil.

What makes this project unique is the integration of weather data (rain, temperature, storms) with healthcare records, allowing deeper behavioral insights.

🚧 Challenges Faced

This was not a straightforward ML problem.

Severe Class Imbalance (~1:9 ratio)
No-show cases were rare, making the model biased toward predicting attendance.
Feature Complexity
Combining healthcare + temporal + weather data required careful feature engineering.
Low Signal Problem
Many features had weak individual predictive power, making model improvement difficult.
Model Performance Trade-offs
Increasing recall reduced precision and vice versa — requiring careful threshold tuning.

This project involved extensive experimentation, debugging, and multiple iterations to reach meaningful results.

⚙️ ML Approach
🔹 Data Processing
Handled missing values using SimpleImputer
Feature engineering:
waiting_days
appointment_hour
Weather indicators (rain, heat, storms)
🔹 Models Used
Random Forest
XGBoost (final best model)
🔹 Optimization
GridSearchCV with StratifiedKFold
Imbalance handling:
class_weight
scale_pos_weight
🔹 Threshold Tuning

Adjusted classification threshold to improve recall vs precision trade-off, making predictions more practical.

📊 Results (XGBoost)
Accuracy: ~82%
F1 Score: ~32%
Precision: ~24%
Recall: ~50%
🌐 Deployment
Built an interactive web app using Streamlit
Takes patient + appointment + weather input
Outputs:
No-show probability
Risk category (Low / Medium / High)
💡 Key Insights
🌧️ Rain and storms increase no-show probability
🌡️ Extreme temperatures impact attendance
⏳ Waiting time is a strong behavioral factor
🛠️ Tech Stack
Python, Pandas, NumPy
Scikit-learn, XGBoost
Streamlit
Joblib
👨‍💻 Author
Aditya
B.Tech CSE (AI/ML)
