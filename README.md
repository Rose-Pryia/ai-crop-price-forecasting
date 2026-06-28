# Crop Price Prediction API

A Flask-based REST API for predicting crop prices using XGBoost machine learning model. The API allows users to input basic information (state, district, market, commodity, and date) while the backend automatically calculates complex features like lag prices, moving averages, and historical trends.

## Features

- **Simple Input**: Users only need to provide 5 basic inputs
- **Backend Feature Engineering**: Automatically calculates lag prices, moving averages, min/max prices
- **XGBoost Model**: Uses trained XGBoost model for accurate predictions
- **RESTful API**: Clean JSON-based API endpoints
- **Error Handling**: Comprehensive error handling and validation

## Installation

1. **Install Dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

2. **Ensure Model Files are Present**:
   Make sure these files are in the same directory as `app.py`:
   - `xgboost_model.pkl` - Trained XGBoost model
   - `label_encoders.pkl` - Label encoders for categorical variables
   - `dataset (1).csv` - Historical data for feature engineering

## Usage

### Starting the API Server

```bash
python app.py
```

The server will start on `http://localhost:5001`

### API Endpoints

#### 1. Health Check
```
GET /health
```
Returns the health status of the API and whether models are loaded.

#### 2. Sample Input Format
```
GET /sample
```
Returns a sample input format for the prediction endpoint.

#### 3. Price Prediction
```
POST /predict
```

**Request Body** (JSON):
```json
{
    "state": "Karnataka",
    "district": "Kalburgi",
    "market": "Kalburgi",
    "commodity": "Wheat",
    "date": "2024-12-01"
}
```

**Response** (JSON):
```json
{
    "success": true,
    "input": {
        "state": "Karnataka",
        "district": "Kalburgi",
        "market": "Kalburgi",
        "commodity": "Wheat",
        "date": "2024-12-01"
    },
    "predicted_price": 2150.75,
    "currency": "INR"
}
```

### Input Requirements

- **state**: State name (string)
- **district**: District name (string)
- **market**: Market name (string)
- **commodity**: Commodity name (string)
- **date**: Date in format 'YYYY-MM-DD' or 'DD/MM/YYYY' (string)

### Backend Calculations

The API automatically calculates these features in the backend:
- **Lag Prices**: 1-day, 7-day, and 30-day lag prices
- **Moving Averages**: 7-day and 30-day moving averages
- **Min/Max Prices**: Historical minimum and maximum prices
- **Date Features**: Year, month, day, day of week, week of year
- **Encoded Categories**: Label-encoded categorical variables

## Testing

Run the test script to verify the API is working:

```bash
python test_api.py
```

This will test all endpoints with sample data.

### Manual Testing with curl

```bash
# Health check
curl http://localhost:5001/health

# Sample format
curl http://localhost:5001/sample

# Make prediction
curl -X POST http://localhost:5001/predict \
  -H "Content-Type: application/json" \
  -d '{
    "state": "Karnataka",
    "district": "Kalburgi", 
    "market": "Kalburgi",
    "commodity": "Wheat",
    "date": "2024-12-01"
  }'
```

## Sample Inputs

Here are some sample inputs you can try:

### Example 1: Wheat in Karnataka
```json
{
    "state": "Karnataka",
    "district": "Kalburgi",
    "market": "Kalburgi",
    "commodity": "Wheat",
    "date": "2024-12-01"
}
```

### Example 2: Tomato in Maharashtra
```json
{
    "state": "Maharashtra", 
    "district": "Pune",
    "market": "Pune",
    "commodity": "Tomato",
    "date": "2024-12-15"
}
```

### Example 3: Rice in Punjab
```json
{
    "state": "Punjab",
    "district": "Ludhiana",
    "market": "Ludhiana", 
    "commodity": "Rice",
    "date": "01/01/2025"
}
```

## Error Handling

The API provides detailed error messages for:
- Missing required fields
- Invalid date formats
- Unknown state/district/market/commodity combinations
- Model loading issues
- Internal server errors

## Architecture

The API follows this flow:
1. **Input Validation**: Validates required fields and formats
2. **Data Encoding**: Encodes categorical variables using saved label encoders
3. **Feature Engineering**: Calculates lag prices, moving averages, and date features
4. **Prediction**: Uses XGBoost model to predict the modal price
5. **Response**: Returns formatted JSON response with prediction

## Notes

- The API uses historical data to calculate features, so predictions are based on past trends
- If no historical data is found for a specific combination, it falls back to dataset averages
- All prices are in Indian Rupees (INR)
- The model was trained on agricultural market data and works best with known state/district/market/commodity combinations
