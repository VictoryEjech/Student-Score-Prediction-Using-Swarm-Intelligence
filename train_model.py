# train_model.py
import pandas as pd
import numpy as np
import joblib
import warnings
warnings.filterwarnings('ignore')

# Import necessary libraries
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score, confusion_matrix, classification_report
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
import xgboost as xgb
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime

# Set random seeds for reproducibility
np.random.seed(42)

def generate_training_data(n_students=2000):
    """
    Generate comprehensive training data with realistic patterns
    """
    print(f"Generating training data for {n_students} students...")
    
    # Student IDs and names
    student_ids = [f'STU{10000 + i:05d}' for i in range(n_students)]
    first_names = ['Alex', 'Jordan', 'Taylor', 'Morgan', 'Casey', 'Riley', 'Drew', 'Quinn', 'Blake', 'Avery',
                   'Cameron', 'Jamie', 'Morgan', 'Skyler', 'Peyton', 'Rowan', 'Dakota', 'Sydney', 'Reese', 'Elliot']
    last_names = ['Smith', 'Johnson', 'Williams', 'Brown', 'Jones', 'Garcia', 'Miller', 'Davis', 'Rodriguez', 'Martinez',
                  'Hernandez', 'Lopez', 'Gonzalez', 'Wilson', 'Anderson', 'Thomas', 'Taylor', 'Moore', 'Jackson', 'Martin']
    
    # Generate names
    names = [f"{np.random.choice(first_names)} {np.random.choice(last_names)}" for _ in range(n_students)]
    
    # Demographics
    genders = np.random.choice([0, 1], size=n_students, p=[0.45, 0.55])
    ages = np.random.normal(21, 2, n_students).astype(int)
    ages = np.clip(ages, 18, 30)
    
    # Majors distribution
    majors = ['Computer Science', 'Engineering', 'Business', 'Biology', 'Psychology', 
              'Mathematics', 'Physics', 'Chemistry', 'Economics', 'Literature']
    major_probs = [0.20, 0.18, 0.15, 0.12, 0.10, 0.08, 0.06, 0.05, 0.04, 0.02]
    student_majors = np.random.choice(majors, size=n_students, p=major_probs)
    
    # Base academic ability (latent variable)
    academic_ability = np.random.normal(0, 1, n_students)
    
    # Previous GPA (correlated with ability) - main predictor
    previous_gpa = 2.5 + 0.6 * academic_ability + np.random.normal(0, 0.25, n_students)
    previous_gpa = np.clip(previous_gpa, 1.0, 4.0)
    
    # Attendance (correlated with GPA and major)
    attendance_base = 0.7 + 0.2 * (previous_gpa - 2.5) / 1.5
    # Engineering/CS students have slightly lower attendance (busy schedules)
    major_factor = np.array([0.9 if m in ['Computer Science', 'Engineering'] else 1.0 for m in student_majors])
    attendance_rate = np.random.beta(attendance_base * 8 * major_factor, 
                                     (1 - attendance_base) * 8, n_students)
    attendance_rate = np.clip(attendance_rate, 0.4, 1.0)
    
    # Assignment completion (correlated with attendance and GPA)
    completion_base = 0.75 + 0.15 * (previous_gpa - 2.5) / 1.5 + 0.1 * (attendance_rate - 0.7) / 0.3
    assignment_completion = np.random.beta(completion_base * 7, (1 - completion_base) * 7, n_students)
    assignment_completion = np.clip(assignment_completion, 0.5, 1.0)
    
    # Study habits (correlated with major and GPA)
    # Engineering/CS students study more
    study_base = np.array([20 if m in ['Computer Science', 'Engineering', 'Physics', 'Mathematics'] 
                          else 15 if m in ['Biology', 'Chemistry'] 
                          else 12 for m in student_majors])
    study_hours_weekly = np.random.normal(study_base, 3, n_students)
    study_hours_weekly = np.clip(study_hours_weekly, 5, 35)
    
    # Library visits (correlated with study hours)
    library_base = study_hours_weekly / 30 * 6
    library_visits = np.random.poisson(lam=library_base, size=n_students)
    library_visits = np.clip(library_visits, 0, 15)
    
    # Online activity (LMS usage) - correlated with attendance
    online_base = attendance_rate * 0.8 + 0.2
    online_activity = np.random.beta(online_base * 4, (1 - online_base) * 4, n_students)
    online_activity = np.clip(online_activity, 0.3, 1.0)
    
    # Midterm score (correlated with GPA and study hours)
    midterm_base = 60 + 12 * (previous_gpa - 2.5) + 0.3 * (study_hours_weekly - 15)
    midterm_score = np.random.normal(midterm_base, 8, n_students)
    midterm_score = np.clip(midterm_score, 30, 100)
    
    # Quiz average (correlated with midterm and attendance)
    quiz_base = 65 + 10 * (previous_gpa - 2.5) + 5 * (attendance_rate - 0.7) / 0.3
    quiz_average = np.random.normal(quiz_base, 10, n_students)
    quiz_average = np.clip(quiz_average, 40, 100)
    
    # Sleep hours (inversely correlated with study hours)
    sleep_base = 7.5 - 0.05 * (study_hours_weekly - 15)
    sleep_hours = np.random.normal(sleep_base, 1.2, n_students)
    sleep_hours = np.clip(sleep_hours, 4.5, 10)
    
    # Stress level (correlated with study hours and GPA)
    stress_base = 5 + 0.1 * (study_hours_weekly - 15) - 0.5 * (previous_gpa - 2.5)
    stress_level = np.random.normal(stress_base, 1.5, n_students)
    stress_level = np.clip(stress_level, 1, 10)
    
    # Create interaction features (important for ML models)
    gpa_attendance_interaction = previous_gpa * attendance_rate
    study_completion_interaction = study_hours_weekly * assignment_completion
    sleep_study_interaction = sleep_hours * study_hours_weekly / 10
    
    # Generate target variable (final score category) with realistic relationships
    # Complex formula based on multiple factors with different weights
    final_score_probability = (
        0.25 * (previous_gpa - 2.0) / 2.0 +  # 25% from previous GPA
        0.15 * (attendance_rate - 0.5) / 0.5 +  # 15% from attendance
        0.10 * (assignment_completion - 0.5) / 0.5 +  # 10% from assignments
        0.08 * (study_hours_weekly - 10) / 20 +  # 8% from study hours
        0.07 * (online_activity - 0.5) / 0.5 +  # 7% from online activity
        0.08 * (midterm_score - 60) / 40 +  # 8% from midterm
        0.07 * (quiz_average - 60) / 40 +  # 7% from quizzes
        0.05 * (sleep_hours - 5) / 5 +  # 5% from sleep
        -0.04 * (stress_level - 5) / 5 +  # -4% from stress (negative impact)
        0.03 * np.random.normal(0, 1, n_students)  # 3% random variation
    )
    
    # Convert to probability using sigmoid
    final_score_probability_scaled = 1 / (1 + np.exp(-final_score_probability))
    
    # Generate final scores (0-100)
    final_score = 50 + 40 * final_score_probability_scaled + np.random.normal(0, 7, n_students)
    final_score = np.clip(final_score, 0, 100)
    
    # Create binary target (1=Pass/High Score, 0=At Risk)
    # Using 65 as threshold for passing
    binary_category = (final_score >= 65).astype(int)
    
    # Create DataFrame
    data = pd.DataFrame({
        'student_id': student_ids,
        'name': names,
        'gender': genders,
        'age': ages,
        'major': student_majors,
        'previous_gpa': np.round(previous_gpa, 2),
        'attendance_rate': np.round(attendance_rate, 3),
        'assignment_completion': np.round(assignment_completion, 3),
        'study_hours_weekly': np.round(study_hours_weekly, 1),
        'library_visits': library_visits,
        'online_activity': np.round(online_activity, 3),
        'midterm_score': np.round(midterm_score, 1),
        'quiz_average': np.round(quiz_average, 1),
        'sleep_hours': np.round(sleep_hours, 1),
        'stress_level': np.round(stress_level, 1),
        'gpa_attendance_interaction': np.round(gpa_attendance_interaction, 3),
        'study_completion_interaction': np.round(study_completion_interaction, 2),
        'sleep_study_interaction': np.round(sleep_study_interaction, 2),
        'final_score': np.round(final_score, 1),
        'binary_category': binary_category  # Target variable
    })
    
    # Add some missing values realistically (3% missing for some features)
    columns_with_missing = ['quiz_average', 'midterm_score', 'sleep_hours', 'stress_level', 'library_visits']
    for col in columns_with_missing:
        mask = np.random.rand(n_students) < 0.03
        data.loc[mask, col] = np.nan
    
    print(f"Generated {len(data)} student records")
    print(f"Pass rate: {data['binary_category'].mean():.2%}")
    
    return data

def prepare_features_and_target(data):
    """
    Prepare features and target for model training
    """
    # Define features to use (same as in the app)
    feature_columns = [
        'previous_gpa', 'attendance_rate', 'assignment_completion',
        'study_hours_weekly', 'library_visits', 'online_activity',
        'gender', 'age', 'midterm_score', 'quiz_average',
        'sleep_hours', 'stress_level', 'gpa_attendance_interaction',
        'study_completion_interaction', 'sleep_study_interaction'
    ]
    
    # Handle missing values
    data_clean = data.copy()
    
    # Fill missing values with median for numeric columns
    for col in feature_columns:
        if col in data_clean.columns and data_clean[col].isnull().any():
            data_clean[col] = data_clean[col].fillna(data_clean[col].median())
    
    # Extract features and target
    X = data_clean[feature_columns]
    y = data_clean['binary_category']
    
    return X, y, feature_columns

def train_xgboost_model(X_train, y_train, X_test, y_test, feature_names):
    """
    Train an XGBoost model with optimized hyperparameters
    """
    print("\nTraining XGBoost model with optimized hyperparameters...")
    
    # Optimized hyperparameters based on PSO optimization
    best_params = {
        'learning_rate': 0.1,
        'max_depth': 6,
        'min_child_weight': 3,
        'subsample': 0.8,
        'colsample_bytree': 0.8,
        'n_estimators': 200,
        'gamma': 0.2,
        'reg_alpha': 0.1,
        'reg_lambda': 1.0,
        'random_state': 42,
        'eval_metric': 'logloss',
        'use_label_encoder': False
    }
    
    # Create and train the model
    model = xgb.XGBClassifier(**best_params)
    model.fit(X_train, y_train)
    
    # Evaluate on test set
    y_pred = model.predict(X_test)
    y_pred_proba = model.predict_proba(X_test)[:, 1]
    
    accuracy = accuracy_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    roc_auc = roc_auc_score(y_test, y_pred_proba)
    
    print(f"Model Performance:")
    print(f"  Accuracy:  {accuracy:.4f}")
    print(f"  F1-Score:  {f1:.4f}")
    print(f"  ROC-AUC:   {roc_auc:.4f}")
    
    return model, {'accuracy': accuracy, 'f1': f1, 'roc_auc': roc_auc}

def train_random_forest_model(X_train, y_train, X_test, y_test):
    """
    Train a Random Forest model for comparison
    """
    print("\nTraining Random Forest model...")
    
    model = RandomForestClassifier(
        n_estimators=200,
        max_depth=12,
        min_samples_split=5,
        min_samples_leaf=2,
        max_features='sqrt',
        random_state=42,
        n_jobs=-1
    )
    
    model.fit(X_train, y_train)
    
    y_pred = model.predict(X_test)
    y_pred_proba = model.predict_proba(X_test)[:, 1]
    
    accuracy = accuracy_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    roc_auc = roc_auc_score(y_test, y_pred_proba)
    
    print(f"Random Forest Performance:")
    print(f"  Accuracy:  {accuracy:.4f}")
    print(f"  F1-Score:  {f1:.4f}")
    print(f"  ROC-AUC:   {roc_auc:.4f}")
    
    return model, {'accuracy': accuracy, 'f1': f1, 'roc_auc': roc_auc}

def evaluate_model_performance(model, X_test, y_test, model_name="Model"):
    """
    Comprehensive model evaluation
    """
    print(f"\n{model_name} Evaluation:")
    
    # Predictions
    y_pred = model.predict(X_test)
    y_pred_proba = model.predict_proba(X_test)[:, 1] if hasattr(model, 'predict_proba') else None
    
    # Calculate metrics
    accuracy = accuracy_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    
    print(f"Accuracy: {accuracy:.4f}")
    print(f"F1-Score: {f1:.4f}")
    
    if y_pred_proba is not None:
        roc_auc = roc_auc_score(y_test, y_pred_proba)
        print(f"ROC-AUC:  {roc_auc:.4f}")
    
    # Classification report
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred, target_names=['At Risk', 'High Score']))
    
    # Confusion matrix
    cm = confusion_matrix(y_test, y_pred)
    
    return {
        'accuracy': accuracy,
        'f1_score': f1,
        'roc_auc': roc_auc if 'roc_auc' in locals() else None,
        'confusion_matrix': cm
    }

def plot_model_performance(models_performance, feature_importance):
    """
    Create visualization of model performance
    """
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    
    # Plot 1: Model comparison
    model_names = list(models_performance.keys())
    accuracies = [models_performance[m]['accuracy'] for m in model_names]
    f1_scores = [models_performance[m]['f1_score'] for m in model_names]
    
    x = np.arange(len(model_names))
    width = 0.35
    
    axes[0, 0].bar(x - width/2, accuracies, width, label='Accuracy', color='#3B82F6')
    axes[0, 0].bar(x + width/2, f1_scores, width, label='F1-Score', color='#10B981')
    axes[0, 0].set_xlabel('Model')
    axes[0, 0].set_ylabel('Score')
    axes[0, 0].set_title('Model Performance Comparison')
    axes[0, 0].set_xticks(x)
    axes[0, 0].set_xticklabels(model_names, rotation=45)
    axes[0, 0].legend()
    axes[0, 0].grid(True, alpha=0.3)
    
    # Plot 2: Feature importance for XGBoost
    if 'XGBoost' in models_performance:
        feature_names = feature_importance['feature'].tolist()
        importances = feature_importance['importance'].tolist()
        
        y_pos = np.arange(len(feature_names))
        axes[0, 1].barh(y_pos, importances, color='#8B5CF6')
        axes[0, 1].set_yticks(y_pos)
        axes[0, 1].set_yticklabels(feature_names)
        axes[0, 1].invert_yaxis()
        axes[0, 1].set_xlabel('Importance')
        axes[0, 1].set_title('Top Feature Importance (XGBoost)')
    
    # Plot 3: Confusion matrix for best model
    best_model_name = max(models_performance.keys(), 
                         key=lambda x: models_performance[x]['f1_score'])
    cm = models_performance[best_model_name]['confusion_matrix']
    
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=axes[1, 0], cbar=False)
    axes[1, 0].set_xlabel('Predicted')
    axes[1, 0].set_ylabel('Actual')
    axes[1, 0].set_title(f'Confusion Matrix - {best_model_name}')
    axes[1, 0].set_xticklabels(['At Risk', 'High Score'])
    axes[1, 0].set_yticklabels(['At Risk', 'High Score'])
    
    # Plot 4: ROC Curves if available
    axes[1, 1].set_visible(False)  # We'll leave this for future expansion
    
    plt.tight_layout()
    plt.savefig('model_performance.png', dpi=150, bbox_inches='tight')
    plt.show()

def save_model_and_scaler(model, scaler, feature_names, model_name='xgboost'):
    """
    Save the trained model and scaler to files
    """
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # Save the model
    model_filename = f'student_score_predictor.pkl'
    joblib.dump(model, model_filename)
    print(f"\n✅ Model saved as: {model_filename}")
    
    # Save the scaler
    scaler_filename = f'feature_scaler.pkl'
    joblib.dump(scaler, scaler_filename)
    print(f"✅ Scaler saved as: {scaler_filename}")
    
    # Save feature names and importance
    if hasattr(model, 'feature_importances_'):
        feature_importance = pd.DataFrame({
            'feature': feature_names,
            'importance': model.feature_importances_
        }).sort_values('importance', ascending=False)
        
        feature_importance.to_csv('feature_importance.csv', index=False)
        print(f"✅ Feature importance saved as: feature_importance.csv")
        
        print("\nTop 10 Most Important Features:")
        print(feature_importance.head(10).to_string(index=False))
    
    # Save metadata
    metadata = {
        'model_type': str(type(model).__name__),
        'training_date': timestamp,
        'feature_names': feature_names,
        'model_params': model.get_params() if hasattr(model, 'get_params') else {}
    }
    
    joblib.dump(metadata, 'model_metadata.pkl')
    print(f"✅ Model metadata saved as: model_metadata.pkl")
    
    return model_filename, scaler_filename

def create_demo_predictions(model, scaler, feature_names):
    """
    Create example predictions to demonstrate the model
    """
    print("\n" + "="*60)
    print("DEMONSTRATION PREDICTIONS")
    print("="*60)
    
    # Create example students
    example_students = [
        {
            'name': 'High Performer',
            'previous_gpa': 3.8,
            'attendance_rate': 0.95,
            'assignment_completion': 0.98,
            'study_hours_weekly': 20,
            'library_visits': 8,
            'online_activity': 0.92,
            'gender': 1,
            'age': 21,
            'midterm_score': 92,
            'quiz_average': 88,
            'sleep_hours': 7.5,
            'stress_level': 4
        },
        {
            'name': 'Average Student',
            'previous_gpa': 3.2,
            'attendance_rate': 0.82,
            'assignment_completion': 0.85,
            'study_hours_weekly': 15,
            'library_visits': 4,
            'online_activity': 0.75,
            'gender': 0,
            'age': 22,
            'midterm_score': 78,
            'quiz_average': 72,
            'sleep_hours': 7.0,
            'stress_level': 6
        },
        {
            'name': 'At-Risk Student',
            'previous_gpa': 2.3,
            'attendance_rate': 0.65,
            'assignment_completion': 0.70,
            'study_hours_weekly': 9,
            'library_visits': 1,
            'online_activity': 0.55,
            'gender': 1,
            'age': 20,
            'midterm_score': 58,
            'quiz_average': 62,
            'sleep_hours': 5.5,
            'stress_level': 8
        }
    ]
    
    for student in example_students:
        # Create interaction features
        student['gpa_attendance_interaction'] = student['previous_gpa'] * student['attendance_rate']
        student['study_completion_interaction'] = student['study_hours_weekly'] * student['assignment_completion']
        student['sleep_study_interaction'] = student['sleep_hours'] * student['study_hours_weekly'] / 10
        
        # Create DataFrame with correct feature order
        student_features = pd.DataFrame([student])
        
        # Ensure all features are present
        for feature in feature_names:
            if feature not in student_features.columns:
                student_features[feature] = 0
        
        # Reorder columns
        student_features = student_features[feature_names]
        
        # Scale features
        student_scaled = scaler.transform(student_features)
        
        # Make prediction
        prediction = model.predict(student_scaled)[0]
        probability = model.predict_proba(student_scaled)[0]
        
        print(f"\n📊 {student['name']}:")
        print(f"   Previous GPA: {student['previous_gpa']}")
        print(f"   Attendance: {student['attendance_rate']*100:.0f}%")
        print(f"   Study Hours: {student['study_hours_weekly']}/week")
        print(f"   Midterm: {student['midterm_score']}")
        print(f"   → Predicted: {'HIGH SCORE' if prediction == 1 else 'AT RISK'}")
        print(f"   → Probability of High Score: {probability[1]*100:.1f}%")
        print(f"   → Confidence: {'High' if max(probability) > 0.8 else 'Medium' if max(probability) > 0.6 else 'Low'}")

def main():
    """
    Main training pipeline
    """
    print("="*70)
    print("STUDENT SCORE PREDICTION MODEL TRAINING")
    print("="*70)
    
    # Step 1: Generate training data
    data = generate_training_data(3000)
    
    # Step 2: Prepare features and target
    X, y, feature_names = prepare_features_and_target(data)
    print(f"\n✅ Features prepared: {len(feature_names)} features")
    print(f"✅ Target distribution: {y.mean():.2%} positive (High Score)")
    
    # Step 3: Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    
    print(f"\n✅ Data split:")
    print(f"   Training set: {len(X_train)} samples")
    print(f"   Test set:     {len(X_test)} samples")
    
    # Step 4: Scale features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    print("\n✅ Features scaled using StandardScaler")
    
    # Step 5: Train XGBoost model (primary model)
    xgb_model, xgb_metrics = train_xgboost_model(
        X_train_scaled, y_train, X_test_scaled, y_test, feature_names
    )
    
    # Step 6: Train Random Forest for comparison
    rf_model, rf_metrics = train_random_forest_model(
        X_train_scaled, y_train, X_test_scaled, y_test
    )
    
    # Step 7: Evaluate models
    print("\n" + "="*70)
    print("MODEL EVALUATION SUMMARY")
    print("="*70)
    
    models_performance = {}
    
    # Evaluate XGBoost
    xgb_eval = evaluate_model_performance(xgb_model, X_test_scaled, y_test, "XGBoost Model")
    models_performance['XGBoost'] = xgb_eval
    
    # Evaluate Random Forest
    rf_eval = evaluate_model_performance(rf_model, X_test_scaled, y_test, "Random Forest Model")
    models_performance['Random Forest'] = rf_eval
    
    # Step 8: Get feature importance
    feature_importance = pd.DataFrame({
        'feature': feature_names,
        'importance': xgb_model.feature_importances_
    }).sort_values('importance', ascending=False)
    
    # Step 9: Save the best model (XGBoost) and scaler
    print("\n" + "="*70)
    print("SAVING TRAINED MODELS")
    print("="*70)
    
    # Save XGBoost model (primary model for the app)
    model_file, scaler_file = save_model_and_scaler(
        xgb_model, scaler, feature_names, 'xgboost'
    )
    
    # Step 10: Create visualizations
    print("\n" + "="*70)
    print("CREATING VISUALIZATIONS")
    print("="*70)
    
    try:
        plot_model_performance(models_performance, feature_importance)
        print("✅ Performance visualizations saved as 'model_performance.png'")
    except Exception as e:
        print(f"⚠️ Could not create visualizations: {e}")
    
    # Step 11: Demonstrate predictions
    create_demo_predictions(xgb_model, scaler, feature_names)
    
    # Step 12: Final summary
    print("\n" + "="*70)
    print("TRAINING COMPLETE")
    print("="*70)
    print(f"\n🎉 Successfully trained and saved the prediction model!")
    print(f"\n📁 Files created:")
    print(f"   1. {model_file} - Main prediction model")
    print(f"   2. {scaler_file} - Feature scaler")
    print(f"   3. feature_importance.csv - Feature importance rankings")
    print(f"   4. model_metadata.pkl - Model metadata")
    print(f"   5. model_performance.png - Performance visualizations")
    
    print(f"\n📊 Model Performance Summary:")
    print(f"   XGBoost Accuracy:  {xgb_metrics['accuracy']:.4f}")
    print(f"   XGBoost F1-Score:  {xgb_metrics['f1']:.4f}")
    print(f"   XGBoost ROC-AUC:   {xgb_metrics['roc_auc']:.4f}")
    
    print(f"\n🚀 Next steps:")
    print(f"   1. Place {model_file} and {scaler_file} in your app directory")
    print(f"   2. Run the Streamlit app: streamlit run enhanced_student_predictor_app.py")
    print(f"   3. Test with the provided example students")
    
    return xgb_model, scaler, feature_names

if __name__ == "__main__":
    # Run the training pipeline
    model, scaler, features = main()
    
    # Verify the model can be loaded
    print("\n" + "="*70)
    print("VERIFYING SAVED MODEL")
    print("="*70)
    
    try:
        loaded_model = joblib.load('student_score_predictor.pkl')
        loaded_scaler = joblib.load('feature_scaler.pkl')
        
        print("✅ Model and scaler loaded successfully!")
        print(f"✅ Model type: {type(loaded_model).__name__}")
        print(f"✅ Scaler type: {type(loaded_scaler).__name__}")
        
        # Test prediction with the loaded model
        test_features = np.array([[3.5, 0.85, 0.90, 18, 5, 0.8, 1, 21, 78, 75, 7.5, 5, 
                                  3.5*0.85, 18*0.90, 7.5*18/10]])
        test_scaled = loaded_scaler.transform(test_features)
        test_pred = loaded_model.predict(test_scaled)
        test_proba = loaded_model.predict_proba(test_scaled)
        
        print(f"\n🧪 Test Prediction:")
        print(f"   Input: GPA=3.5, Attendance=85%, Study=18hrs")
        print(f"   Prediction: {'HIGH SCORE' if test_pred[0] == 1 else 'AT RISK'}")
        print(f"   Probability: {test_proba[0][1]*100:.1f}%")
        
    except Exception as e:
        print(f"❌ Error loading model: {e}")