"""
Federated Learning Client 1
This client connects to the server and trains on client1-data.csv
"""
import flwr as fl
import numpy as np
import pandas as pd
import pickle
import xgboost as xgb
from typing import Dict, List, Tuple, Optional
from model_utils import load_label_encoders, prepare_features_for_training, get_model_params, create_model_from_params
import warnings
warnings.filterwarnings('ignore')

class FlowerClient(fl.client.NumPyClient):
    """Flower client for federated learning"""
    
    def __init__(self, client_id: int, data_file: str):
        self.client_id = client_id
        self.data_file = data_file
        self.model = None
        self.X_train = None
        self.y_train = None
        self.feature_columns = None
        self.label_encoders = None
        
        # Load data
        self.load_data()
        
        # Initialize model
        self.initialize_model()
    
    def load_data(self):
        """Load and prepare training data"""
        print(f"\n[Client {self.client_id}] Loading data from {self.data_file}...")
        try:
            df = pd.read_csv(self.data_file)
            print(f"[Client {self.client_id}] Loaded {len(df)} rows")
            
            # Load label encoders
            try:
                self.label_encoders = load_label_encoders()
            except FileNotFoundError:
                print(f"[Client {self.client_id}] Creating new label encoders...")
                self.create_label_encoders(df)
            
            # Prepare features
            X, y, feature_cols = prepare_features_for_training(df, self.label_encoders)
            self.X_train = X.values
            self.y_train = y.values
            self.feature_columns = feature_cols
            
            print(f"[Client {self.client_id}] Prepared {len(self.X_train)} samples for training")
            
        except Exception as e:
            print(f"[Client {self.client_id}] Error loading data: {e}")
            raise
    
    def create_label_encoders(self, df: pd.DataFrame):
        """Create label encoders if they don't exist"""
        from sklearn.preprocessing import LabelEncoder
        
        categorical_cols = ['State', 'District', 'Market', 'Commodity']
        self.label_encoders = {}
        
        for col in categorical_cols:
            if col in df.columns:
                le = LabelEncoder()
                df[col] = df[col].astype(str)
                le.fit(df[col].unique())
                self.label_encoders[col] = le
        
        # Save encoders
        with open('label_encoders.pkl', 'wb') as f:
            pickle.dump(self.label_encoders, f)
        print(f"[Client {self.client_id}] Created and saved label encoders")
    
    def initialize_model(self):
        """Initialize the XGBoost model"""
        self.model = xgb.XGBRegressor(
            n_estimators=100,
            max_depth=6,
            learning_rate=0.1,
            random_state=42,
            n_jobs=-1
        )
    
    def get_parameters(self, config: Dict) -> List[np.ndarray]:
        """Return model parameters as a list of NumPy arrays"""
        # For XGBoost, we need to extract parameters differently
        # Convert model to a format that can be serialized
        model_params = get_model_params(self.model)
        
        # Convert to NumPy arrays for Flower
        # We'll use feature importance and model parameters
        params = []
        params.append(np.array(model_params['feature_importance'], dtype=np.float32))
        params.append(np.array([model_params['n_estimators']], dtype=np.float32))
        params.append(np.array([model_params['max_depth']], dtype=np.float32))
        params.append(np.array([model_params['learning_rate']], dtype=np.float32))
        
        return params
    
    def set_parameters(self, parameters: List[np.ndarray]):
        """Update model parameters from server"""
        # Extract parameters
        if len(parameters) >= 4:
            feature_importance = parameters[0]
            n_estimators = int(parameters[1][0])
            max_depth = int(parameters[2][0])
            learning_rate = float(parameters[3][0])
            
            # Update model parameters
            self.model = xgb.XGBRegressor(
                n_estimators=n_estimators,
                max_depth=max_depth,
                learning_rate=learning_rate,
                random_state=42,
                n_jobs=-1
            )
    
    def fit(self, parameters: List[np.ndarray], config: Dict) -> Tuple[List[np.ndarray], int, Dict]:
        """Train the model on local data"""
        round_num = config.get("round", 0)
        print(f"\n[Client {self.client_id}] Round {round_num}: Starting local training...")
        
        # Set parameters from server
        self.set_parameters(parameters)
        
        # Train the model
        try:
            self.model.fit(
                self.X_train,
                self.y_train,
                eval_set=[(self.X_train, self.y_train)],
                verbose=False
            )
            
            # Calculate training metrics
            train_pred = self.model.predict(self.X_train)
            train_mae = np.mean(np.abs(train_pred - self.y_train))
            train_rmse = np.sqrt(np.mean((train_pred - self.y_train) ** 2))
            
            print(f"[Client {self.client_id}] Round {round_num}: Training complete")
            print(f"[Client {self.client_id}] MAE: {train_mae:.2f}, RMSE: {train_rmse:.2f}")
            
            # Return updated parameters
            updated_params = self.get_parameters(config)
            
            return updated_params, len(self.X_train), {
                "mae": float(train_mae),
                "rmse": float(train_rmse),
                "client_id": self.client_id
            }
        except Exception as e:
            print(f"[Client {self.client_id}] Error during training: {e}")
            # Return current parameters even if training fails
            return self.get_parameters(config), len(self.X_train), {"error": str(e)}
    
    def evaluate(self, parameters: List[np.ndarray], config: Dict) -> Tuple[float, int, Dict]:
        """Evaluate the model on local data"""
        round_num = config.get("round", 0)
        print(f"[Client {self.client_id}] Round {round_num}: Evaluating...")
        
        # Set parameters
        self.set_parameters(parameters)
        
        # Evaluate
        try:
            # Use a subset for evaluation if dataset is large
            eval_size = min(1000, len(self.X_train))
            indices = np.random.choice(len(self.X_train), eval_size, replace=False)
            X_eval = self.X_train[indices]
            y_eval = self.y_train[indices]
            
            predictions = self.model.predict(X_eval)
            mae = np.mean(np.abs(predictions - y_eval))
            rmse = np.sqrt(np.mean((predictions - y_eval) ** 2))
            
            print(f"[Client {self.client_id}] Evaluation - MAE: {mae:.2f}, RMSE: {rmse:.2f}")
            
            return float(rmse), len(X_eval), {
                "mae": float(mae),
                "rmse": float(rmse),
                "client_id": self.client_id
            }
        except Exception as e:
            print(f"[Client {self.client_id}] Error during evaluation: {e}")
            return float('inf'), len(self.X_train), {"error": str(e)}

def main():
    """Start the client"""
    import sys
    
    # Get server address from command line or use default
    if len(sys.argv) > 1:
        server_address = sys.argv[1]
    else:
        server_address = "localhost:8080"
        print("Warning: No server address provided. Using default: localhost:8080")
        print("Usage: python client1.py SERVER_ADDRESS")
        print("  Example (local IP): python client1.py 192.168.1.100:8080")
        print("  Example (ngrok): python client1.py 0.tcp.ngrok.io:12345")
    
    # Parse server address
    if ':' not in server_address:
        server_address = f"{server_address}:8080"
    
    print("=" * 60)
    print("Federated Learning Client 1 - Crop Price Prediction")
    print("=" * 60)
    
    # Detect connection type
    if 'ngrok' in server_address.lower() or 'tcp.ngrok' in server_address.lower():
        print(f"🌐 Connecting via NGROK: {server_address}")
    else:
        print(f"🔗 Connecting via LOCAL NETWORK: {server_address}")
    
    print(f"Training on: client1-data.csv")
    print()
    
    # Create client
    client = FlowerClient(client_id=1, data_file="client1-data.csv")
    
    # Start Flower client
    try:
        fl.client.start_numpy_client(
            server_address=server_address,
            client=client
        )
    except Exception as e:
        print(f"\n❌ Connection failed: {e}")
        print("\nTroubleshooting:")
        print("  1. Check if server is running")
        print("  2. Verify server address is correct")
        print("  3. Ensure network connectivity")
        if 'ngrok' not in server_address.lower():
            print("  4. Try using ngrok if on different networks")
        sys.exit(1)

if __name__ == "__main__":
    main()

