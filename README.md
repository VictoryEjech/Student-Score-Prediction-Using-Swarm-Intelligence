# Student Score Prediction Using Swarm Intelligence
NAME: EJECHI CHUKWUBUIKEM VICTORY  MATNO: ENG2308246  DEPT:  CIVIL ENGINEERING COURSE: SCHOOL: UNIVERSITY OF DELTA, AGBOR, DELTA STATE, NIGERIA  GEE307	Introduction to Artificial Intelligence, Machine Learning and Convergent Technologies

# About the system
This system assists teachers in pinpointing areas for development, customizing educational opportunities, and offering focused assistance to pupils who are having difficulty.

# Run the Program via batch
How to Run the Complete System
Step 1: Install All Required Packages

pip install streamlit pandas numpy scikit-learn xgboost joblib matplotlib seaborn plotly openpyxl

Step 2: Ensure Model Files Exist

Place these files in your working directory
(or run train_model.py to generate them):

student_score_predictor.pkl ,
feature_scaler.pkl

Step 3: Run the Application

streamlit run advanced_student_analytics_system.py
================================================

you can now view the Advanced Student Analytics Streamlit app of the student score prediction in your browser.

  Network URL: http://10.0.10.170:8501
  External URL: http://172.210.53.196:8501

or click on the link below:
https://super-enigma-jjp9jrgpw5xgcpqgv-8501.app.github.dev/

==========Complete Feature Set======================================

1. 🎯 Single Student Prediction Interface
✅ Fully Functional Features:

Comprehensive Input Form with 4 tabs:

📊 Academic Info: GPA, attendance, assignments

⏰ Study Habits: Study hours, library visits, online activity

😴 Lifestyle: Sleep, stress, employment

📈 Current Performance: Midterm, quizzes, projects

Real-time Predictions with confidence scores

Personalized Recommendations based on prediction

Detailed Analytics with performance metrics

Export Options: Download reports, add to batch

2. 📊 Batch Student Prediction Interface
✅ Fully Functional Features:

Multiple Input Methods:

📁 CSV/Excel Upload: Direct file upload with validation

📝 Manual Entry: Interactive editable tables

🎯 Sample Data: Generate realistic test data

Batch Processing: Analyze 100+ students simultaneously

Comprehensive Results:

Summary statistics and metrics

Interactive data tables with filtering

Visual distribution charts

Export Options:

Download as CSV or Excel

Generate summary reports

Save for comparative analysis

3. 📈 Comparative Analysis Interface
✅ Fully Functional Features:

Automatic Enablement: Requires batch prediction data

Performance Distribution:

Pie charts showing high-performer vs at-risk ratios

Risk score distribution histograms

GPA vs Risk scatter plots with trend lines

Feature Comparison:

Side-by-side comparison of key metrics

Statistical significance testing (p-values)

Performance gap analysis

Trend Analysis:

GPA vs Midterm performance trends

Study hours vs Quiz scores

Risk progression patterns

Student Clustering:

Automated K-means clustering

2D/3D cluster visualization

Cluster characteristics analysis

Advanced Tools:

Specific student comparison with radar charts

Feature correlation matrices

Predictive insights and recommendations

4. 🔗 Feature Links & Quick Access
✅ Fully Functional Features:

System Information Links:

PSO-Optimized XGBoost model details

15+ Predictive Features documentation

Real-time Analytics dashboard

Batch Capabilities Links:

CSV/Excel Upload interface

Manual Data Entry forms

Sample Data Generation tools

Analysis Features Links:

Comparative Analysis tools

Trend Identification dashboards

Cluster Detection algorithms

🎮 User Workflow Examples
Example 1: Individual Student Assessment
Go to 🎯 Single Student Prediction tab

Fill in student details across all 4 input tabs

Click 🚀 Generate Prediction & Recommendations

View detailed results with personalized recommendations

Download report or add to batch for comparison

Example 2: Class-wide Analysis
Go to 📊 Batch Student Prediction tab

Choose 📁 Upload File and upload your student data CSV

Click 🚀 Run Batch Prediction

View summary statistics and individual predictions

Go to 📈 Comparative Analysis tab

Explore performance distributions, trends, and clusters

Generate comprehensive group report

Example 3: Department-level Planning
Use 🎯 Sample Data Generation to create test dataset

Run 📊 Batch Prediction on sample data

Use 📈 Comparative Analysis to identify patterns

Generate Group Recommendations for interventions

Export all findings for department meetings


