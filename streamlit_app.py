# =================================================================
# FILE: streamlit_app.py
# GOAL: Create interactive frontend with model selection and dynamic prediction.
# =================================================================

import streamlit as st
import pandas as pd
import numpy as np
import joblib
import os
from datetime import date, timedelta
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split

# --- 1. File Paths and Asset Loading ---
BASE_PATH = os.path.dirname(os.path.abspath(__file__))
ASSETS_PATH = os.path.join(BASE_PATH, 'assets')
RESULTS_PATH = os.path.join(BASE_PATH, 'results')
DATA_INPUT_PATH = os.path.join(BASE_PATH, 'data', 'processed', 'features_df.pkl')

@st.cache_resource
def load_and_train_models():
    """Loads assets and trains LR and RF models for comparison, only run once."""
    try:
        # Load necessary assets
        xgb_model = joblib.load(os.path.join(RESULTS_PATH, 'final_xgb_model.pkl'))
        preprocessor = joblib.load(os.path.join(ASSETS_PATH, 'preprocessor.pkl'))
        selected_features = np.load(os.path.join(ASSETS_PATH, 'selected_features.npy'), allow_pickle=True).tolist()
        df = pd.read_pickle(DATA_INPUT_PATH)
        
        # Re-create training set for comparison models
        y = df['y']
        X = df.drop(columns=['y'])
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=False)
        
        # Process and select features for training
        X_train_processed = preprocessor.transform(X_train)
        feature_names_processed = preprocessor.get_feature_names_out()
        X_train_processed_df = pd.DataFrame(X_train_processed, columns=feature_names_processed)
        X_train_final = X_train_processed_df[selected_features]
        
        # Train comparison models
        lr_model = LinearRegression()
        lr_model.fit(X_train_final, y_train)
        
        rf_model = RandomForestRegressor(n_estimators=300, max_depth=15, random_state=42, n_jobs=-1)
        rf_model.fit(X_train_final, y_train)

        return {
            'XGBoost (Boosting)': xgb_model,
            'Random Forest (Bagging)': rf_model,
            'Linear Regression (Baseline)': lr_model
        }, preprocessor, selected_features

    except Exception as e:
        st.error(f"Error initializing models or loading assets. Please ensure all previous notebooks were run correctly: {e}")
        st.stop()

MODELS, PREPROCESSOR, SELECTED_FEATURES = load_and_train_models()


# --- 2. Feature Engineering Function (FIXED) ---
# This function replicates the logic from 02_feature_engineering.ipynb

def create_forecast_features(input_date, ad_spend, promo_flag, holiday_flag):
    """Creates all required features for a single date prediction."""
    
    # Placeholder for lag values (In production, this data would be fetched from a database)
    PLACEHOLDER_LAG = 3500 

    # 1. Base DataFrame: MUST INCLUDE ALL COLUMNS THE PREPROCESSOR WAS FITTED ON
    data = {
        'ad_spend_usd': [ad_spend],
        'promo_flag': [promo_flag],
        'holiday_flag': [holiday_flag],
        
        # === FIX: ADDED MISSING COLUMNS REQUIRED BY ColumnTransformer ===
        'clicks': [0],
        'impressions': [0],
        'ctr': [0],
        'bounce_rate': [0],
        # ===============================================================

        # Include other static features that were in the training data but are not user inputs
        'avg_session_duration_sec': [0], 'organic_search': [0], 'referral_traffic': [0],
    }
    
    features_df = pd.DataFrame(data, index=[pd.to_datetime(input_date)])

    # 2. Time-Based Features (Must match 02_feature_engineering.ipynb)
    features_df['year'] = features_df.index.year
    features_df['month'] = features_df.index.month
    features_df['day'] = features_df.index.day
    features_df['dow'] = features_df.index.dayofweek
    features_df['week_of_year'] = features_df.index.isocalendar().week.astype(int)
    features_df['is_weekend'] = features_df['dow'].apply(lambda x: 1 if x >= 5 else 0)

    # 3. Lagged and Rolling Features
    LAG_PERIODS = [1, 2, 7, 30] 
    for lag in LAG_PERIODS:
        features_df[f'y_lag_{lag}'] = PLACEHOLDER_LAG 
    features_df['y_rolling_mean_7'] = PLACEHOLDER_LAG
    
    return features_df


# --- 3. Streamlit UI Design ---

st.title("📈 Multi-Model Web Traffic Forecast")
st.markdown("Select a model and input parameters to generate a prediction.")

# Model Selector (New Input)
selected_model_name = st.selectbox(
    "1. Select Model for Prediction:", 
    options=list(MODELS.keys())
)
selected_model = MODELS[selected_model_name]

# --- Input Parameters ---
st.header("2. Input Prediction Parameters")

# Define the minimum prediction date (tomorrow)
min_date = date.today() + timedelta(days=1)

col_date, col_ad_spend = st.columns(2)
with col_date:
    target_date = st.date_input(
        "Forecast Date:", 
        min_value=min_date,
        value=min_date,
        help="The model will extract time features (month, day of week) from this date."
    )

with col_ad_spend:
    ad_spend = st.number_input(
        "Planned Ad Spend (USD):", 
        min_value=0.0, 
        value=20.0, 
        step=0.1,
        help="This is a key external feature for prediction."
    )

col_promo, col_holiday = st.columns(2)
with col_promo:
    promo_flag = st.selectbox(
        "Is there a Promotion (`promo_flag`)?", 
        options=[0, 1],
        format_func=lambda x: "Yes (1)" if x == 1 else "No (0)"
    )

with col_holiday:
    holiday_flag = st.selectbox(
        "Is the day a Holiday (`holiday_flag`)?", 
        options=[0, 1],
        format_func=lambda x: "Yes (1)" if x == 1 else "No (0)"
    )

# --- Dynamic Prediction Button ---
if st.button(f"🔮 Get Forecast using {selected_model_name}", type="primary"):
    
    # 1. Create features from input
    X_predict = create_forecast_features(
        target_date, ad_spend, promo_flag, holiday_flag
    )
    
    # 2. Preprocess (Scale and Encode)
    # This line now works because X_predict contains all the necessary columns
    X_predict_processed = PREPROCESSOR.transform(X_predict)
    
    # 3. Filter using only the selected features
    X_predict_df = pd.DataFrame(
        X_predict_processed, 
        columns=PREPROCESSOR.get_feature_names_out()
    )
    X_predict_final = X_predict_df[SELECTED_FEATURES]
    
    # 4. Predict using the selected model
    prediction = selected_model.predict(X_predict_final)[0]
    
    # --- 5. Output Result ---
    
    st.success(f"### 🎉 {selected_model_name} Forecast for {target_date.strftime('%Y-%m-%d')}:")
    st.balloons()
    
    st.metric(
        label="Predicted Web Traffic (Customers)", 
        value=f"{int(prediction):,}"
    )

    st.info(f"""
        **Model Used:** {selected_model_name}
        **Key Inputs:** Ad Spend: ${ad_spend}, Promo: {promo_flag}, Holiday: {holiday_flag}
    """)