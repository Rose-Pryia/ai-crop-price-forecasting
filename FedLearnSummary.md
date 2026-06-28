# Federated Learning System - Setup Summary

## ✅ Completed Components

### 1. Dataset Preparation

- ✅ Split `dataset.csv` into `client1-data.csv` and `client2-data.csv` (50/50 split)
- ✅ Generated 50 sample entries for each client
- ✅ Each client now has ~100,050 rows of data

### 2. Server Components

- ✅ `server.py` - Federated learning server using Flower
  - Listens on `0.0.0.0:8080`
  - Implements Federated Averaging strategy
  - Aggregates model updates from clients
  - Supports 10 training rounds (configurable)

### 3. Client Components

- ✅ `client1.py` - Client 1 federated learning client
  - Trains on `client1-data.csv`
  - Connects to server via IP address
  - Sends model updates, receives global model
- ✅ `client2.py` - Client 2 federated learning client
  - Trains on `client2-data.csv`
  - Connects to server via IP address
  - Sends model updates, receives global model

### 4. Web Interfaces

- ✅ `client1_web.py` - Web interface for Client 1

  - Simple black and white HTML design
  - Manual entry form for crop data
  - Bulk CSV upload functionality
  - View recent entries
  - Runs on `http://localhost:5002`

- ✅ `client2_web.py` - Web interface for Client 2
  - Simple black and white HTML design
  - Manual entry form for crop data
  - Bulk CSV upload functionality
  - View recent entries
  - Runs on `http://localhost:5003`

### 5. Supporting Files

- ✅ `model_utils.py` - Model utilities for federated learning

  - Model serialization/deserialization
  - Feature preparation
  - Label encoder management

- ✅ `split_dataset.py` - Dataset splitting script
- ✅ `generate_sample_data.py` - Sample data generation script

### 6. Run Scripts

- ✅ `run_server.sh` - Server startup script
- ✅ `run_client1.sh` - Client 1 startup script
- ✅ `run_client2.sh` - Client 2 startup script
- ✅ `run_client1_web.sh` - Client 1 web interface script
- ✅ `run_client2_web.sh` - Client 2 web interface script

### 7. Documentation

- ✅ `FEDERATED_LEARNING_README.md` - Comprehensive documentation
- ✅ `QUICK_START.md` - Quick start guide
- ✅ `SETUP_SUMMARY.md` - This file

### 8. Dependencies

- ✅ Updated `requirements.txt` with:
  - `flwr>=1.8.0` - Federated learning framework
  - `fastapi>=0.104.0` - Web framework
  - `uvicorn>=0.24.0` - ASGI server
  - `python-multipart>=0.0.6` - File upload support
  - `joblib>=1.3.0` - Model serialization

## 📁 File Structure

```
crop-predict/
├── server.py                      # Federated learning server
├── client1.py                     # Client 1 federated learning
├── client2.py                     # Client 2 federated learning
├── client1_web.py                 # Client 1 web interface
├── client2_web.py                 # Client 2 web interface
├── model_utils.py                 # Model utilities
├── split_dataset.py               # Dataset splitter
├── generate_sample_data.py        # Sample data generator
├── dataset.csv                    # Original dataset (200k rows)
├── client1-data.csv              # Client 1 dataset (~100k rows)
├── client2-data.csv              # Client 2 dataset (~100k rows)
├── xgboost_model.pkl             # Global model (updated during FL)
├── label_encoders.pkl            # Label encoders
├── requirements.txt               # Dependencies
├── run_server.sh                  # Server startup script
├── run_client1.sh                 # Client 1 startup script
├── run_client2.sh                 # Client 2 startup script
├── run_client1_web.sh             # Client 1 web startup script
├── run_client2_web.sh             # Client 2 web startup script
├── FEDERATED_LEARNING_README.md   # Full documentation
├── QUICK_START.md                 # Quick start guide
└── SETUP_SUMMARY.md               # This file
```

## 🚀 How to Use

### Server (Your Laptop)

```bash
python server.py
```

### Client 1 (Laptop 1)

```bash
python client1.py SERVER_IP:8080
# Example: python client1.py 192.168.1.100:8080
```

### Client 2 (Laptop 2)

```bash
python client2.py SERVER_IP:8080
# Example: python client2.py 192.168.1.100:8080
```

### Web Interfaces

```bash
# Client 1 web interface
python client1_web.py
# Access at http://localhost:5002

# Client 2 web interface
python client2_web.py
# Access at http://localhost:5003
```

## 🔄 Federated Learning Flow

1. **Server Initialization**

   - Server starts and loads/initializes global model
   - Waits for clients to connect

2. **Client Connection**

   - Clients connect to server via IP address
   - Server sends initial model parameters

3. **Local Training**

   - Each client trains on their local dataset
   - Clients send model updates to server

4. **Aggregation**

   - Server aggregates updates using Federated Averaging
   - Creates new global model

5. **Distribution**

   - Server sends updated global model to clients

6. **Repeat**
   - Process repeats for multiple rounds (default: 10)

## 📊 Data Management

### Adding New Data

**Via Web Interface:**

- Open client web interface
- Fill in form with crop data
- Click "Add Entry"
- Data appended to client's CSV file

**Via Bulk Upload:**

- Prepare CSV file with required columns
- Use "Bulk Upload CSV" in web interface
- Upload file
- Entries appended to client's CSV file

### Data Format

Required columns:

- State, District, Market, Commodity
- Variety, Grade
- Arrival_Date (DD/MM/YYYY format)
- Min_Price, Max_Price, Modal_Price
- Commodity_Code

## 🔧 Configuration

### Number of Rounds

Edit `server.py`:

```python
fl.server.start_server(
    config=fl.server.ServerConfig(num_rounds=10),  # Change this
    ...
)
```

### Model Hyperparameters

Edit `client1.py` and `client2.py`:

```python
self.model = xgb.XGBRegressor(
    n_estimators=100,      # Number of trees
    max_depth=6,           # Tree depth
    learning_rate=0.1,     # Learning rate
    ...
)
```

### Port Numbers

- Server: `8080` (in `server.py`)
- Client 1 web: `5002` (in `client1_web.py`)
- Client 2 web: `5003` (in `client2_web.py`)

## 🌐 Network Setup

### Finding Server IP

**macOS/Linux:**

```bash
ifconfig | grep "inet " | grep -v 127.0.0.1
```

**Windows:**

```bash
ipconfig
```

### Requirements

- All machines on same WiFi network
- Port 8080 accessible (check firewall)
- Server IP address known to clients

## 📝 Notes

1. **XGBoost Federated Learning**: This implementation uses parameter averaging for XGBoost. The approach averages hyperparameters (n_estimators, max_depth, learning_rate) and feature importance, then retrains models on clients.

2. **Data Privacy**: Each client's data stays on their machine. Only model parameters are shared.

3. **Model Updates**: The global model (`xgboost_model.pkl`) is updated after each federated learning round.

4. **Label Encoders**: Created automatically if they don't exist. The first client to connect will create them.

5. **Web Interfaces**: Can run independently of federated learning. They just manage the CSV files.

## 🐛 Troubleshooting

### Connection Issues

- Verify server IP address
- Check firewall settings
- Ensure all machines on same network
- Test with: `ping SERVER_IP`

### Missing Files

- Run `split_dataset.py` to create client datasets
- Ensure `xgboost_model.pkl` exists (or will be created)

### Port Conflicts

- Change port numbers in respective files
- Update client connection strings

## ✨ Features

✅ Federated learning with Flower framework
✅ XGBoost model support
✅ Two-client architecture
✅ Web interfaces for data entry
✅ Bulk CSV upload
✅ Automatic dataset splitting
✅ Sample data generation
✅ Comprehensive documentation
✅ Easy-to-use run scripts

## 📚 Documentation Files

- `FEDERATED_LEARNING_README.md` - Full system documentation
- `QUICK_START.md` - Quick start guide
- `SETUP_SUMMARY.md` - This summary

## 🎯 Next Steps

1. Start the server on your laptop
2. Start clients on other laptops
3. Monitor training progress
4. Add new data via web interfaces
5. Experiment with hyperparameters
6. Add more clients if needed

---

**System Status**: ✅ Ready to Use

All components have been created and tested. The system is ready for federated learning deployment.
