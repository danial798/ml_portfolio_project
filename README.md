# Chronic Kidney Disease (CKD) Prediction

This project aims to predict the risk of Chronic Kidney Disease (CKD) using machine learning techniques. It includes a complete pipeline from data acquisition and preprocessing to model training and a user-friendly Streamlit web application.

## Project Structure

- `kidney_disease.csv`: The dataset used for training and testing.
- `train_model.py`: Script to preprocess data, train models, and save the best one.
- `app.py`: Streamlit application for real-time predictions.
- `eda.py`: Script for Exploratory Data Analysis.
- `requirements.txt`: List of dependencies.
- `best_model.pkl`: Trained Random Forest model.
- `scaler.pkl`: StandardScaler object for feature scaling.
- `label_encoders.pkl`: LabelEncoder objects for categorical variables.

## Setup & Usage

1. **Install Dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

2. **Train Model (Optional):**
   The model is already trained and saved. To retrain:
   ```bash
   python train_model.py
   ```

3. **Run App:**
   ```bash
   streamlit run app.py
   ```

## Model Performance

The Random Forest Classifier achieved an accuracy of approximately **97.5%** on the test set.

## Features

## Deployment

To deploy this app to Streamlit Community Cloud:

1.  **Push to GitHub:**
    -   Initialize a git repository (already done).
    -   Commit your changes:
        ```bash
        git add .
        git commit -m "Initial commit"
        ```
    -   Create a new repository on GitHub.
    -   Push your code:
        ```bash
        git remote add origin <your-repo-url>
        git push -u origin master
        ```

2.  **Deploy on Streamlit Cloud:**
    -   Go to [share.streamlit.io](https://share.streamlit.io/).
    -   Connect your GitHub account.
    -   Select "New app".
    -   Choose your repository and the main file (`app.py`).
    -   Click "Deploy".
