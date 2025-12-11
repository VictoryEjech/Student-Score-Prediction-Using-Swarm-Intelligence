import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import random

# Set random seeds for reproducibility
np.random.seed(42)
random.seed(42)

def generate_realistic_student_data(num_students=1000):
    """Generate realistic student data with academic, demographic, and behavioral features"""
    
    # Student IDs
    student_ids = [f'STU{10000 + i:05d}' for i in range(num_students)]
    
    # Demographics
    genders = np.random.choice(['Male', 'Female', 'Other'], size=num_students, p=[0.48, 0.50, 0.02])
    ages = np.random.randint(18, 25, size=num_students)
    
    # Majors distribution
    majors = ['Computer Science', 'Engineering', 'Business', 'Biology', 'Psychology', 
              'Mathematics', 'Physics', 'Chemistry', 'Economics', 'Literature']
    major_probs = [0.25, 0.20, 0.15, 0.10, 0.08, 0.07, 0.05, 0.04, 0.03, 0.03]
    student_majors = np.random.choice(majors, size=num_students, p=major_probs)
    
    # Academic performance features (with correlations)
    
    # Base academic ability (latent variable)
    academic_ability = np.random.normal(0, 1, num_students)
    
    # Previous GPA (correlated with ability)
    previous_gpa = 2.5 + 0.5 * academic_ability + np.random.normal(0, 0.3, num_students)
    previous_gpa = np.clip(previous_gpa, 1.0, 4.0)  # GPA between 1.0 and 4.0
    
    # Attendance (correlated with GPA)
    attendance_base = 0.7 + 0.2 * (previous_gpa - 2.5)/1.5
    attendance_rate = np.random.beta(attendance_base * 10, (1 - attendance_base) * 10, num_students)
    attendance_rate = np.clip(attendance_rate, 0.4, 1.0)
    
    # Assignment completion (correlated with attendance and GPA)
    completion_base = 0.75 + 0.15 * (previous_gpa - 2.5)/1.5 + 0.1 * (attendance_rate - 0.7)/0.3
    assignment_completion = np.random.beta(completion_base * 8, (1 - completion_base) * 8, num_students)
    assignment_completion = np.clip(assignment_completion, 0.5, 1.0)
    
    # Study habits
    study_hours_weekly = np.random.gamma(shape=2.5, scale=2.0, size=num_students)
    # Cap at 30 hours
    study_hours_weekly = np.where(study_hours_weekly > 30, 30, study_hours_weekly)
    
    # Library visits (correlated with study hours)
    library_base = study_hours_weekly / 30 * 5
    library_visits = np.random.poisson(lam=library_base, size=num_students)
    
    # Online activity (LMS usage)
    online_activity = np.random.beta(3, 2, num_students) * attendance_rate * 0.8 + 0.2
    
    # Extra features for more realism
    num_courses = np.random.choice([3, 4, 5, 6], size=num_students, p=[0.1, 0.4, 0.4, 0.1])
    
    # Part-time job (affects study time)
    has_job = np.random.choice([0, 1], size=num_students, p=[0.6, 0.4])
    job_hours = has_job * np.random.randint(5, 25, size=num_students)
    
    # Sleep hours (affects performance)
    sleep_hours = np.random.normal(7.0, 1.5, num_students)
    sleep_hours = np.clip(sleep_hours, 4.0, 10.0)
    
    # Stress level (1-10 scale)
    stress_level = np.random.normal(5, 2, num_students)
    stress_level = np.clip(stress_level, 1, 10)
    
    # Previous exam scores (midterm)
    midterm_score = 60 + 10 * (previous_gpa - 2.5) + np.random.normal(0, 8, num_students)
    midterm_score = np.clip(midterm_score, 30, 100)
    
    # Quiz average
    quiz_average = 70 + 8 * (previous_gpa - 2.5) + np.random.normal(0, 10, num_students)
    quiz_average = np.clip(quiz_average, 40, 100)
    
    # Create interaction features
    gpa_attendance_interaction = previous_gpa * attendance_rate
    study_completion_interaction = study_hours_weekly * assignment_completion
    sleep_study_interaction = sleep_hours * study_hours_weekly / 10
    
    # Generate target variable (final score category) with realistic relationships
    # Complex formula based on multiple factors
    final_score_probability = (
        0.25 * (previous_gpa - 2.0) / 2.0 +  # 25% from previous GPA
        0.15 * (attendance_rate - 0.5) / 0.5 +  # 15% from attendance
        0.10 * (assignment_completion - 0.5) / 0.5 +  # 10% from assignments
        0.10 * (study_hours_weekly - 10) / 20 +  # 10% from study hours
        0.08 * (online_activity - 0.5) / 0.5 +  # 8% from online activity
        0.07 * (midterm_score - 60) / 40 +  # 7% from midterm
        0.05 * (quiz_average - 60) / 40 +  # 5% from quizzes
        0.05 * (sleep_hours - 5) / 5 +  # 5% from sleep
        -0.05 * (stress_level - 5) / 5 +  # -5% from stress
        -0.03 * (job_hours / 20) +  # -3% from job hours
        0.07 * np.random.normal(0, 1, num_students)  # 7% random variation
    )
    
    # Convert to pass/fail with some noise
    final_score_probability_scaled = 1 / (1 + np.exp(-final_score_probability))
    
    # Generate final scores (0-100)
    final_score = 50 + 40 * final_score_probability_scaled + np.random.normal(0, 8, num_students)
    final_score = np.clip(final_score, 0, 100)
    
    # Create score categories
    # 0: Fail (<60), 1: Pass (60-79), 2: Good (80-100)
    score_category = np.zeros(num_students, dtype=int)
    score_category[final_score >= 60] = 1
    score_category[final_score >= 80] = 2
    
    # Also create binary classification (pass/fail)
    binary_category = (final_score >= 60).astype(int)
    
    # Generate dates for semester timeline
    start_date = datetime(2023, 9, 1)
    enrollment_dates = [start_date + timedelta(days=random.randint(0, 14)) 
                        for _ in range(num_students)]
    
    # Create DataFrame
    data = pd.DataFrame({
        'student_id': student_ids,
        'gender': genders,
        'age': ages,
        'major': student_majors,
        'previous_gpa': np.round(previous_gpa, 2),
        'attendance_rate': np.round(attendance_rate, 2),
        'assignment_completion': np.round(assignment_completion, 2),
        'study_hours_weekly': np.round(study_hours_weekly, 1),
        'library_visits': library_visits,
        'online_activity': np.round(online_activity, 2),
        'num_courses': num_courses,
        'has_job': has_job,
        'job_hours': job_hours,
        'sleep_hours': np.round(sleep_hours, 1),
        'stress_level': np.round(stress_level, 1),
        'midterm_score': np.round(midterm_score, 1),
        'quiz_average': np.round(quiz_average, 1),
        'gpa_attendance_interaction': np.round(gpa_attendance_interaction, 3),
        'study_completion_interaction': np.round(study_completion_interaction, 2),
        'sleep_study_interaction': np.round(sleep_study_interaction, 2),
        'final_score': np.round(final_score, 1),
        'score_category': score_category,  # 0=Fail, 1=Pass, 2=Good
        'binary_category': binary_category,  # 0=Fail, 1=Pass
        'enrollment_date': enrollment_dates
    })
    
    # Add some missing values realistically (5% missing for some features)
    columns_with_missing = ['quiz_average', 'midterm_score', 'sleep_hours', 'stress_level']
    for col in columns_with_missing:
        mask = np.random.rand(num_students) < 0.05
        data.loc[mask, col] = np.nan
    
    return data

# Generate the data
print("Generating realistic student dataset...")
student_data = generate_realistic_student_data(1500)  # 1500 students

# Save to CSV
csv_filename = "your_student_data.csv"
student_data.to_csv(csv_filename, index=False)

# Also save a smaller version for testing
student_data.head(300).to_csv("your_student_data_sample.csv", index=False)

print(f"Dataset saved as '{csv_filename}'")
print(f"Sample dataset saved as 'your_student_data_sample.csv'")
print("\nDataset Information:")
print(f"Number of students: {len(student_data)}")
print(f"Number of features: {len(student_data.columns)}")
print("\nFirst few rows:")
print(student_data.head())

# Create a summary statistics file
summary_stats = pd.DataFrame({
    'Feature': student_data.select_dtypes(include=[np.number]).columns,
    'Mean': student_data.select_dtypes(include=[np.number]).mean(),
    'Std': student_data.select_dtypes(include=[np.number]).std(),
    'Min': student_data.select_dtypes(include=[np.number]).min(),
    'Max': student_data.select_dtypes(include=[np.number]).max(),
    'Missing_%': student_data.isnull().mean() * 100
})

print("\nSummary Statistics:")
print(summary_stats.to_string())

print("\nTarget Variable Distribution:")
print("Score Category (0=Fail, 1=Pass, 2=Good):")
print(student_data['score_category'].value_counts().sort_index())
print(f"\nBinary Category (0=Fail, 1=Pass):")
print(student_data['binary_category'].value_counts().sort_index())
print(f"\nPass Rate: {student_data['binary_category'].mean():.2%}")

# Create feature description file
feature_descriptions = pd.DataFrame({
    'Feature': [
        'student_id', 'gender', 'age', 'major', 'previous_gpa', 'attendance_rate',
        'assignment_completion', 'study_hours_weekly', 'library_visits', 'online_activity',
        'num_courses', 'has_job', 'job_hours', 'sleep_hours', 'stress_level',
        'midterm_score', 'quiz_average', 'gpa_attendance_interaction',
        'study_completion_interaction', 'sleep_study_interaction', 'final_score',
        'score_category', 'binary_category', 'enrollment_date'
    ],
    'Description': [
        'Unique student identifier',
        'Gender of the student',
        'Age of the student',
        'Major/Field of study',
        'Previous semester GPA (1.0-4.0 scale)',
        'Percentage of classes attended (0.0-1.0)',
        'Percentage of assignments completed (0.0-1.0)',
        'Average study hours per week',
        'Number of library visits per month',
        'Level of online course engagement (0.0-1.0)',
        'Number of courses enrolled in current semester',
        'Whether student has a part-time job (0=No, 1=Yes)',
        'Hours worked per week (if has_job=1)',
        'Average hours of sleep per night',
        'Self-reported stress level (1-10 scale)',
        'Midterm exam score (0-100)',
        'Average quiz score (0-100)',
        'Interaction term: GPA × Attendance',
        'Interaction term: Study Hours × Assignment Completion',
        'Interaction term: Sleep Hours × Study Hours',
        'Final course score (0-100)',
        'Final score category (0=Fail, 1=Pass, 2=Good)',
        'Binary classification target (0=Fail, 1=Pass)',
        'Date of enrollment in the course'
    ],
    'Type': [
        'String', 'Categorical', 'Integer', 'Categorical', 'Float', 'Float',
        'Float', 'Float', 'Integer', 'Float', 'Integer', 'Binary',
        'Integer', 'Float', 'Float', 'Float', 'Float', 'Float',
        'Float', 'Float', 'Float', 'Integer', 'Binary', 'Date'
    ]
})

feature_descriptions.to_csv('feature_descriptions.csv', index=False)
print(f"\nFeature descriptions saved to 'feature_descriptions.csv'")

# Generate correlation matrix
correlation_matrix = student_data.select_dtypes(include=[np.number]).corr()
print("\nTop correlations with final_score:")
correlations_with_target = correlation_matrix['final_score'].abs().sort_values(ascending=False)
print(correlations_with_target.head(10))