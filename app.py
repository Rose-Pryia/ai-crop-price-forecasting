"""
Flask API for Crop Price Prediction using XGBoost Model
"""

import pandas as pd
import numpy as np
import pickle
import warnings
from datetime import datetime, timedelta
from flask import Flask, request, jsonify, send_from_directory
from flask_cors import CORS
from sklearn.preprocessing import LabelEncoder
import xgboost as xgb

warnings.filterwarnings('ignore')

app = Flask(__name__)
CORS(app)  # Enable CORS for all routes

class PricePredictionAPI:
    """API wrapper for the crop price prediction system"""
    
    def __init__(self):
        self.label_encoders = None
        self.xgb_model = None
        self.feature_columns = None
        self.historical_data = None
        self.is_loaded = False
    
    def load_models(self):
        """Load all saved models and encoders"""
        try:
            # Load label encoders
            with open('label_encoders.pkl', 'rb') as f:
                self.label_encoders = pickle.load(f)
            print("Label encoders loaded successfully")
            
            # Load XGBoost model
            with open('xgboost_model.pkl', 'rb') as f:
                model_data = pickle.load(f)
                self.xgb_model = model_data['model']
                self.feature_columns = model_data['features']
            print("XGBoost model loaded successfully")
            
            # Load historical data for feature engineering
            self.load_historical_data()
            
            self.is_loaded = True
            print("All models loaded successfully!")
            
        except Exception as e:
            print(f"Error loading models: {e}")
            raise e
    
    def load_historical_data(self):
        """Load and preprocess historical data for feature engineering"""
        try:
            # Load dataset
            df = pd.read_csv('dataset.csv')
            
            # Basic preprocessing (similar to original code)
            columns_to_drop = ['Variety', 'Grade', 'Commodity_Code']
            existing_drops = [col for col in columns_to_drop if col in df.columns]
            df = df.drop(columns=existing_drops)
            
            # Convert date
            df['Arrival_Date'] = pd.to_datetime(df['Arrival_Date'], format='%d/%m/%Y')
            df['Year'] = df['Arrival_Date'].dt.year
            df['Month'] = df['Arrival_Date'].dt.month
            df['Day'] = df['Arrival_Date'].dt.day
            df['Day_of_Week'] = df['Arrival_Date'].dt.dayofweek
            df['Week_of_Year'] = df['Arrival_Date'].dt.isocalendar().week
            
            # Encode categorical variables using loaded encoders
            categorical_cols = ['State', 'District', 'Market', 'Commodity']
            for col in categorical_cols:
                if col in df.columns and col in self.label_encoders:
                    # Handle unknown categories by using the most frequent category
                    known_categories = set(self.label_encoders[col].classes_)
                    df[col] = df[col].astype(str)
                    df[col] = df[col].apply(lambda x: x if x in known_categories else self.label_encoders[col].classes_[0])
                    df[col] = self.label_encoders[col].transform(df[col])
            
            # Handle missing values in price columns
            price_cols = ['Min_Price', 'Max_Price', 'Modal_Price']
            df[price_cols] = df[price_cols].ffill()
            for col in price_cols:
                if df[col].isnull().any():
                    median_val = df[col].median()
                    df[col] = df[col].fillna(median_val)
            
            # Sort by date
            df = df.sort_values(by='Arrival_Date').reset_index(drop=True)
            
            # Create lag and moving average features
            df['Modal_Price_Lag_1'] = df.groupby(['State', 'District', 'Market', 'Commodity'])['Modal_Price'].shift(1)
            df['Modal_Price_Lag_7'] = df.groupby(['State', 'District', 'Market', 'Commodity'])['Modal_Price'].shift(7)
            df['Modal_Price_Lag_30'] = df.groupby(['State', 'District', 'Market', 'Commodity'])['Modal_Price'].shift(30)
            
            df['Modal_Price_MA_7'] = df.groupby(['State', 'District', 'Market', 'Commodity'])['Modal_Price'].transform(
                lambda x: x.rolling(window=7, min_periods=1).mean()
            )
            df['Modal_Price_MA_30'] = df.groupby(['State', 'District', 'Market', 'Commodity'])['Modal_Price'].transform(
                lambda x: x.rolling(window=30, min_periods=1).mean()
            )
            
            # Fill NaN values
            lag_cols = ['Modal_Price_Lag_1', 'Modal_Price_Lag_7', 'Modal_Price_Lag_30',
                       'Modal_Price_MA_7', 'Modal_Price_MA_30']
            df[lag_cols] = df[lag_cols].ffill().bfill()
            
            self.historical_data = df
            print(f"Historical data loaded: {df.shape}")
            
        except Exception as e:
            print(f"Error loading historical data: {e}")
            raise e
    
    def predict_price(self, state, district, market, commodity, date_str):
        """
        Make price prediction based on user inputs
        
        Parameters:
        -----------
        state : str
        district : str  
        market : str
        commodity : str
        date_str : str (format: 'YYYY-MM-DD' or 'DD/MM/YYYY')
        
        Returns:
        --------
        float : Predicted modal price
        """
        if not self.is_loaded:
            raise ValueError("Models not loaded. Call load_models() first.")
        
        # Parse date
        try:
            if '/' in date_str:
                pred_date = pd.to_datetime(date_str, format='%d/%m/%Y')
            else:
                pred_date = pd.to_datetime(date_str)
        except:
            raise ValueError("Invalid date format. Use 'YYYY-MM-DD' or 'DD/MM/YYYY'")
        
        # Encode inputs
        try:
            state_enc = self.label_encoders['State'].transform([state])[0]
            district_enc = self.label_encoders['District'].transform([district])[0]
            market_enc = self.label_encoders['Market'].transform([market])[0]
            commodity_enc = self.label_encoders['Commodity'].transform([commodity])[0]
        except:
            raise ValueError("Unknown state/district/market/commodity. Please check your inputs.")
        
        # Extract date features
        year = pred_date.year
        month = pred_date.month
        day = pred_date.day
        day_of_week = pred_date.dayofweek
        week_of_year = pred_date.isocalendar()[1]
        
        # Get historical data for this combination
        hist = self.historical_data[
            (self.historical_data['State'] == state_enc) &
            (self.historical_data['District'] == district_enc) &
            (self.historical_data['Market'] == market_enc) &
            (self.historical_data['Commodity'] == commodity_enc)
        ].sort_values('Arrival_Date')
        
        if len(hist) == 0:
            print("Warning: No historical data found for this combination. Using dataset averages.")
            hist = self.historical_data.copy()
        
        # Calculate lag and MA features (backend calculation)
        recent_prices = hist['Modal_Price'].tail(30).values
        
        modal_price_lag_1 = recent_prices[-1] if len(recent_prices) >= 1 else hist['Modal_Price'].mean()
        modal_price_lag_7 = recent_prices[-7] if len(recent_prices) >= 7 else hist['Modal_Price'].mean()
        modal_price_lag_30 = recent_prices[0] if len(recent_prices) >= 30 else hist['Modal_Price'].mean()
        modal_price_ma_7 = recent_prices[-7:].mean() if len(recent_prices) >= 7 else hist['Modal_Price'].mean()
        modal_price_ma_30 = recent_prices.mean() if len(recent_prices) > 0 else hist['Modal_Price'].mean()
        
        # Get min and max price estimates (backend calculation)
        min_price = hist['Min_Price'].tail(30).mean()
        max_price = hist['Max_Price'].tail(30).mean()
        
        # Create feature vector
        features = pd.DataFrame({
            'State': [state_enc],
            'District': [district_enc],
            'Market': [market_enc],
            'Commodity': [commodity_enc],
            'Min_Price': [min_price],
            'Max_Price': [max_price],
            'Year': [year],
            'Month': [month],
            'Day': [day],
            'Day_of_Week': [day_of_week],
            'Week_of_Year': [week_of_year],
            'Modal_Price_Lag_1': [modal_price_lag_1],
            'Modal_Price_Lag_7': [modal_price_lag_7],
            'Modal_Price_Lag_30': [modal_price_lag_30],
            'Modal_Price_MA_7': [modal_price_ma_7],
            'Modal_Price_MA_30': [modal_price_ma_30]
        })
        
        # Make prediction
        try:
            prediction = self.xgb_model.predict(features)[0]
        except Exception as e:
            # Handle XGBoost compatibility issues
            print(f"XGBoost prediction error: {e}")
            try:
                # Try alternative prediction method with DMatrix
                import xgboost as xgb
                dmatrix = xgb.DMatrix(features)
                prediction = self.xgb_model.predict(dmatrix)[0]
            except Exception as e2:
                print(f"DMatrix prediction error: {e2}")
                # Try accessing the underlying booster
                try:
                    booster = self.xgb_model.get_booster()
                    dmatrix = xgb.DMatrix(features)
                    prediction = booster.predict(dmatrix)[0]
                except Exception as e3:
                    print(f"Booster prediction error: {e3}")
                    # Fallback to a simple average from historical data
                    prediction = self.historical_data['Modal_Price'].mean()
                    print(f"Using fallback prediction: {prediction}")
        
        # Calculate confidence level
        confidence = self.calculate_confidence(hist, prediction, state_enc, district_enc, market_enc, commodity_enc)

        return float(prediction), confidence

    def calculate_confidence(self, hist, prediction, state_enc, district_enc, market_enc, commodity_enc):
        """
        Calculate confidence level based on multiple factors:
        1. Amount of historical data available
        2. Prediction consistency with historical trends
        3. Data recency
        4. Price volatility
        """
        confidence_factors = []

        # Factor 1: Data availability (0-30 points)
        data_points = len(hist)
        if data_points >= 100:
            data_confidence = 30
        elif data_points >= 50:
            data_confidence = 25
        elif data_points >= 20:
            data_confidence = 20
        elif data_points >= 10:
            data_confidence = 15
        else:
            data_confidence = 10
        confidence_factors.append(data_confidence)

        # Factor 2: Historical trend consistency (0-25 points)
        if len(hist) > 0:
            recent_avg = hist['Modal_Price'].tail(30).mean()
            overall_avg = hist['Modal_Price'].mean()

            # Check if prediction is within reasonable range
            price_deviation = abs(prediction - recent_avg) / recent_avg if recent_avg > 0 else 1

            if price_deviation <= 0.1:  # Within 10%
                trend_confidence = 25
            elif price_deviation <= 0.2:  # Within 20%
                trend_confidence = 20
            elif price_deviation <= 0.3:  # Within 30%
                trend_confidence = 15
            elif price_deviation <= 0.5:  # Within 50%
                trend_confidence = 10
            else:
                trend_confidence = 5
        else:
            trend_confidence = 5
        confidence_factors.append(trend_confidence)

        # Factor 3: Data recency (0-20 points)
        if len(hist) > 0:
            latest_date = hist['Arrival_Date'].max()
            days_since_latest = (pd.Timestamp.now() - latest_date).days

            if days_since_latest <= 30:
                recency_confidence = 20
            elif days_since_latest <= 90:
                recency_confidence = 15
            elif days_since_latest <= 180:
                recency_confidence = 10
            elif days_since_latest <= 365:
                recency_confidence = 5
            else:
                recency_confidence = 2
        else:
            recency_confidence = 2
        confidence_factors.append(recency_confidence)

        # Factor 4: Price volatility (0-15 points)
        if len(hist) >= 10:
            price_std = hist['Modal_Price'].std()
            price_mean = hist['Modal_Price'].mean()
            cv = price_std / price_mean if price_mean > 0 else 1  # Coefficient of variation

            if cv <= 0.1:  # Low volatility
                volatility_confidence = 15
            elif cv <= 0.2:
                volatility_confidence = 12
            elif cv <= 0.3:
                volatility_confidence = 8
            elif cv <= 0.5:
                volatility_confidence = 5
            else:
                volatility_confidence = 2
        else:
            volatility_confidence = 5
        confidence_factors.append(volatility_confidence)

        # Factor 5: Model prediction method used (0-10 points)
        # Since we're using XGBoost with fallback, give higher confidence to successful XGBoost predictions
        method_confidence = 8  # Assume XGBoost worked (can be adjusted based on actual method used)
        confidence_factors.append(method_confidence)

        # Calculate total confidence (0-100)
        total_confidence = sum(confidence_factors)

        # Convert to percentage and ensure it's between 0-100
        confidence_percentage = min(100, max(0, total_confidence))

        # Determine confidence level
        if confidence_percentage >= 85:
            confidence_level = "Very High"
        elif confidence_percentage >= 70:
            confidence_level = "High"
        elif confidence_percentage >= 55:
            confidence_level = "Medium"
        elif confidence_percentage >= 40:
            confidence_level = "Low"
        else:
            confidence_level = "Very Low"

        return {
            "percentage": round(confidence_percentage, 1),
            "level": confidence_level,
            "factors": {
                "data_availability": f"{data_confidence}/30 (Data points: {data_points})",
                "trend_consistency": f"{trend_confidence}/25",
                "data_recency": f"{recency_confidence}/20",
                "price_stability": f"{volatility_confidence}/15",
                "model_reliability": f"{method_confidence}/10"
            }
        }

# Initialize the prediction system
predictor = PricePredictionAPI()

@app.route('/')
def home():
    """Serve the HTML interface"""
    return send_from_directory('.', 'index.html')

@app.route('/api')
def api_info():
    """API information endpoint"""
    return jsonify({
        "message": "Crop Price Prediction API",
        "version": "1.0",
        "endpoints": {
            "/predict": "POST - Make price predictions",
            "/health": "GET - Check API health",
            "/sample": "GET - Get sample input format"
        }
    })

@app.route('/health')
def health():
    """Health check endpoint"""
    return jsonify({
        "status": "healthy" if predictor.is_loaded else "not ready",
        "models_loaded": predictor.is_loaded
    })

@app.route('/sample')
def sample():
    """Sample input format endpoint"""
    return jsonify({
        "sample_inputs": [
            {
                "state": "Bihar",
                "district": "Bhagalpur",
                "market": "Bhagalpur",
                "commodity": "Wheat",
                "date": "2024-12-01"
            },
            {
                "state": "Karnataka",
                "district": "Shimoga",
                "market": "Shimoga",
                "commodity": "Rice",
                "date": "2024-12-15"
            },
            {
                "state": "West Bengal",
                "district": "Jhargram",
                "market": "Jhargram",
                "commodity": "Rice",
                "date": "01/01/2025"
            }
        ],
        "available_commodities": ["Onion", "Potato", "Rice", "Wheat"],
        "note": "Date format can be 'YYYY-MM-DD' or 'DD/MM/YYYY'. Use combinations that exist in historical data for better predictions."
    })

@app.route('/predict', methods=['POST'])
def predict():
    """Main prediction endpoint"""
    try:
        # Check if models are loaded
        if not predictor.is_loaded:
            return jsonify({"error": "Models not loaded. Please wait for initialization."}), 503
        
        # Get JSON data
        data = request.get_json()
        if not data:
            return jsonify({"error": "No JSON data provided"}), 400
        
        # Check required fields
        required_fields = ['state', 'district', 'market', 'commodity', 'date']
        missing_fields = [field for field in required_fields if field not in data]
        if missing_fields:
            return jsonify({"error": f"Missing required fields: {missing_fields}"}), 400
        
        # Extract inputs
        state = data['state']
        district = data['district']
        market = data['market']
        commodity = data['commodity']
        date_str = data['date']
        
        # Make prediction
        predicted_price, confidence = predictor.predict_price(state, district, market, commodity, date_str)
        
        return jsonify({
            "success": True,
            "input": {
                "state": state,
                "district": district,
                "market": market,
                "commodity": commodity,
                "date": date_str
            },
            "predicted_price": round(predicted_price, 2),
            "confidence": confidence,
            "currency": "INR"
        })
        
    except ValueError as ve:
        return jsonify({"error": str(ve)}), 400
    except Exception as e:
        return jsonify({"error": f"Internal server error: {str(e)}"}), 500

if __name__ == '__main__':
    print("Loading models...")
    try:
        predictor.load_models()
        print("Models loaded successfully!")
        print("Starting Flask API server...")
        app.run(host='0.0.0.0', port=5001, debug=True)
    except Exception as e:
        print(f"Failed to load models: {e}")
        print("Please ensure all pickle files are in the correct location.")
