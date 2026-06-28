"""
Federated Learning Server using Flower (flwr)
Central server that aggregates model updates from clients
Supports both local IP and ngrok tunneling
"""
import flwr as fl
import pickle
import numpy as np
from typing import List, Tuple, Dict, Optional
import xgboost as xgb
from model_utils import load_label_encoders, get_model_params, create_model_from_params
import warnings
import sys
import argparse
import os
warnings.filterwarnings('ignore')

# Try to import ngrok
try:
    from pyngrok import ngrok
    NGROK_AVAILABLE = True
except ImportError:
    NGROK_AVAILABLE = False
    print("Warning: pyngrok not installed. Install with: pip install pyngrok")

class FedAvgStrategy(fl.server.strategy.FedAvg):
    """Custom strategy for federated averaging with XGBoost"""
    
    def __init__(self, initial_model, label_encoders, feature_columns, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.initial_model = initial_model
        self.label_encoders = label_encoders
        self.feature_columns = feature_columns
        self.round_num = 0
        # Pre-compute initial parameters to avoid requesting from clients
        self._initial_parameters = self._get_initial_parameters()
    
    def _get_initial_parameters(self) -> fl.common.Parameters:
        """Get initial parameters from the global model"""
        # Default parameters
        default_n_estimators = 100
        default_max_depth = 6
        default_learning_rate = 0.1
        
        # Get model parameters, handling both trained and untrained models
        try:
            model_params = get_model_params(self.initial_model)
            feature_importance = model_params.get('feature_importance', [])
            n_estimators = model_params.get('n_estimators', default_n_estimators)
            max_depth = model_params.get('max_depth', default_max_depth)
            learning_rate = model_params.get('learning_rate', default_learning_rate)
        except Exception as e:
            # Model not trained yet or has compatibility issues, use default parameters
            print(f"Using default parameters (model may not be trained or has compatibility issues): {e}")
            num_features = len(self.feature_columns) if self.feature_columns else 15
            feature_importance = [0.0] * num_features
            
            # Try to get parameters from model attributes directly, fallback to defaults
            try:
                # Try accessing attributes directly without get_params()
                n_estimators = getattr(self.initial_model, 'n_estimators', default_n_estimators)
                max_depth = getattr(self.initial_model, 'max_depth', default_max_depth)
                learning_rate = getattr(self.initial_model, 'learning_rate', default_learning_rate)
            except Exception:
                # If that also fails, use defaults
                n_estimators = default_n_estimators
                max_depth = default_max_depth
                learning_rate = default_learning_rate
        
        # Convert to the same format as client.get_parameters()
        # This matches what clients return: [feature_importance, n_estimators, max_depth, learning_rate]
        params_list = []
        params_list.append(np.array(feature_importance, dtype=np.float32))
        params_list.append(np.array([n_estimators], dtype=np.float32))
        params_list.append(np.array([max_depth], dtype=np.float32))
        params_list.append(np.array([learning_rate], dtype=np.float32))
        
        # Convert to Flower Parameters object
        return fl.common.ndarrays_to_parameters(params_list)
    
    def initialize_parameters(self, client_manager: fl.server.ClientManager) -> Optional[fl.common.Parameters]:
        """Provide initial parameters instead of requesting from clients"""
        print("Providing initial parameters from global model")
        return self._initial_parameters
    
    def aggregate_fit(self, rnd: int, results: List[Tuple[fl.server.client_proxy.ClientProxy, fl.common.FitRes]], 
                     failures: List[BaseException]) -> Optional[fl.common.Parameters]:
        """Aggregate model updates from clients"""
        if not results:
            return None
        
        # Aggregate weights from clients
        aggregated_params = super().aggregate_fit(rnd, results, failures)
        
        if aggregated_params is None:
            return None
        
        # Convert aggregated parameters back to model
        # Since XGBoost doesn't support direct weight averaging like neural networks,
        # we'll use parameter averaging and create a new model
        self.round_num = rnd
        print(f"\n=== Round {rnd} Aggregation Complete ===")
        print(f"Received updates from {len(results)} clients")
        
        return aggregated_params

def initialize_global_model():
    """Initialize the global model from saved model"""
    try:
        # Load existing model
        with open('xgboost_model.pkl', 'rb') as f:
            model_data = pickle.load(f)
            model = model_data['model']
            feature_columns = model_data['features']
        
        print("Loaded existing global model")
        return model, feature_columns
    except FileNotFoundError:
        # Create a new model with default parameters
        print("Creating new global model")
        model = xgb.XGBRegressor(
            n_estimators=100,
            max_depth=6,
            learning_rate=0.1,
            random_state=42
        )
        feature_columns = ['State', 'District', 'Market', 'Commodity', 'Min_Price', 'Max_Price',
                          'Year', 'Month', 'Day', 'Day_of_Week', 'Week_of_Year',
                          'Modal_Price_Lag_1', 'Modal_Price_Lag_7', 'Modal_Price_Lag_30',
                          'Modal_Price_MA_7', 'Modal_Price_MA_30']
        return model, feature_columns

def get_initial_model_params():
    """Get initial model parameters to send to clients"""
    model, feature_columns = initialize_global_model()
    model_params = get_model_params(model)
    return model_params, feature_columns

def save_global_model(model, feature_columns):
    """Save the global model"""
    model_data = {
        'model': model,
        'features': feature_columns
    }
    with open('xgboost_model.pkl', 'wb') as f:
        pickle.dump(model_data, f)
    print("Global model saved")

def start_ngrok_tunnel(port: int = 8080):
    """Start ngrok tunnel for the server"""
    if not NGROK_AVAILABLE:
        print("Error: ngrok not available. Install with: pip install pyngrok")
        return None
    
    try:
        # Start ngrok tunnel for TCP (Flower uses gRPC which needs TCP)
        # Note: ngrok.connect returns a NgrokTunnel object
        tunnel = ngrok.connect(port, "tcp")
        public_url = tunnel.public_url
        
        # Extract host and port from ngrok URL
        # Format: tcp://0.tcp.ngrok.io:12345
        if public_url.startswith("tcp://"):
            tcp_address = public_url.replace("tcp://", "")
        else:
            tcp_address = public_url
        
        print(f"\n{'='*60}")
        print("NGROK TUNNEL ACTIVE")
        print(f"{'='*60}")
        print(f"Public URL: {public_url}")
        print(f"TCP Address: {tcp_address}")
        print(f"\nClients can connect using:")
        print(f"  python client1.py {tcp_address}")
        print(f"  python client2.py {tcp_address}")
        print(f"\nOr copy the TCP address above and use it with clients")
        print(f"{'='*60}\n")
        return tunnel
    except Exception as e:
        print(f"Error starting ngrok tunnel: {e}")
        print("Continuing without ngrok...")
        return None

def main():
    """Start the federated learning server"""
    parser = argparse.ArgumentParser(description='Federated Learning Server')
    parser.add_argument('--use-ngrok', action='store_true', 
                       help='Use ngrok tunnel for public access')
    parser.add_argument('--port', type=int, default=8080,
                       help='Server port (default: 8080)')
    parser.add_argument('--ngrok-token', type=str, default=None,
                       help='Ngrok authtoken (optional, can be set via NGROK_AUTHTOKEN env var)')
    
    args = parser.parse_args()
    
    print("=" * 60)
    print("Federated Learning Server - Crop Price Prediction")
    print("=" * 60)
    
    # Setup ngrok if requested
    ngrok_tunnel = None
    if args.use_ngrok:
        if args.ngrok_token:
            try:
                ngrok.set_auth_token(args.ngrok_token)
            except Exception as e:
                print(f"Warning: Could not set ngrok token: {e}")
        elif 'NGROK_AUTHTOKEN' in os.environ:
            try:
                ngrok.set_auth_token(os.environ['NGROK_AUTHTOKEN'])
            except Exception as e:
                print(f"Warning: Could not set ngrok token from env: {e}")
        
        ngrok_tunnel = start_ngrok_tunnel(args.port)
    
    # Load label encoders
    try:
        label_encoders = load_label_encoders()
        print("Label encoders loaded")
    except FileNotFoundError:
        print("Warning: label_encoders.pkl not found. Clients will need to create it.")
        label_encoders = None
    
    # Initialize global model
    model, feature_columns = initialize_global_model()
    
    # Create federated averaging strategy
    strategy = FedAvgStrategy(
        initial_model=model,
        label_encoders=label_encoders,
        feature_columns=feature_columns,
        min_fit_clients=2,  # Wait for at least 2 clients
        min_available_clients=2,  # Need 2 clients available
        min_evaluate_clients=2,
        fraction_fit=1.0,  # Use all available clients
        fraction_evaluate=1.0,
        on_fit_config_fn=lambda rnd: {"round": rnd},
        on_evaluate_config_fn=lambda rnd: {"round": rnd},
    )
    
    # Display connection information
    print(f"\nStarting Flower server on 0.0.0.0:{args.port}")
    if not args.use_ngrok or ngrok_tunnel is None:
        print("\nLocal IP Connection:")
        print("  Find your IP: ifconfig | grep 'inet ' | grep -v 127.0.0.1")
        print(f"  Clients connect: python client1.py YOUR_IP:{args.port}")
    print("\nWaiting for clients to connect...")
    print("Server will aggregate updates from clients using Federated Averaging")
    print("\nPress Ctrl+C to stop the server\n")
    
    try:
        fl.server.start_server(
            server_address=f"0.0.0.0:{args.port}",
            config=fl.server.ServerConfig(num_rounds=1),
            strategy=strategy,
        )
    except KeyboardInterrupt:
        print("\n\nShutting down server...")
    finally:
        # Clean up ngrok tunnel
        if ngrok_tunnel:
            try:
                ngrok.disconnect(ngrok_tunnel.public_url)
                print("Ngrok tunnel closed")
            except Exception as e:
                print(f"Error closing ngrok tunnel: {e}")

if __name__ == "__main__":
    main()

