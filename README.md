# Math Score Prediction

## Project Overview

This project aims to predict math scores of students based on various independent features such as gender, ethnicity, parental level of education, lunch type, test preparation course, reading score, and writing score. The objective is to create a model that can accurately predict math scores, providing insights into factors that influence student performance.

## Dataset

The dataset contains the following independent features:

- **Gender**: Gender of the student
- **Race/Ethnicity**: Ethnic background of the student
- **Parental Level of Education**: Education level of the student's parents
- **Lunch**: Type of lunch the student receives (e.g., standard or free/reduced)
- **Test Preparation Course**: Whether the student completed a test preparation course
- **Reading Score**: Student's score in reading
- **Writing Score**: Student's score in writing

The target variable is:

- **Math Score**: The score the student achieved in mathematics

## Model Training

1. **Data Preprocessing**:
   - Handling missing values
   - Encoding categorical variables
   - Scaling numerical features

2. **Model Selection**:
   - Tried various regression models such as Linear Regression, Decision Trees, and Random Forest.
   - Evaluated models using metrics like Mean Absolute Error (MAE) and R-squared.

3. **Model Evaluation**:
   - Chosen the best model based on performance metrics.
   - Tested the model on a separate test dataset to validate performance.

## Flask API

A Flask application was developed to provide an interface for users to input features and obtain predicted math scores. The input features are:

- Gender
- Race/Ethnicity
- Parental Level of Education
- Lunch
- Test Preparation Course
- Reading Score
- Writing Score

The predicted output is the math score.

## How to Run the Project

1. **Clone the repository**:
   ```bash
   git clone https://github.com/hetbhagatji09/MlProject
   

## Navigate to the project directory
  ```bash
  cd MlProject
