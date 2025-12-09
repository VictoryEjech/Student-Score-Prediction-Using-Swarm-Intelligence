import joblib
import numpy as np

# Load the model and scaler
model = joblib.load('student_score_predictor.pkl')
scaler = joblib.load('feature_scaler.pkl')

# Create a test student
test_student = np.array([[3.2, 0.80, 0.85, 15, 4, 0.75, 1, 21, 75, 72, 7.0, 6,
                          3.2*0.80, 15*0.85, 7.0*15/10]])

# Scale features
test_scaled = scaler.transform(test_student)

# Make prediction
prediction = model.predict(test_scaled)[0]
probability = model.predict_proba(test_scaled)[0]

print(f"Prediction: {'HIGH SCORE' if prediction == 1 else 'AT RISK'}")
print(f"Probability: {probability[1]*100:.1f}%")