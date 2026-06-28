# Quick Start Guide - Federated Learning System

## Prerequisites Checklist

- [ ] Python 3.10+ installed on all machines
- [ ] All files from this repository copied to each machine
- [ ] (For Local IP) All machines on same WiFi network
- [ ] (For Ngrok) Internet connection on server machine

## Step-by-Step Setup

### 1. Install Dependencies (All Machines)

```bash
pip install -r requirements.txt
```

### 2. Prepare Dataset (Server Only - Run Once)

```bash
python split_dataset.py
python generate_sample_data.py
```

This creates:

- `client1-data.csv` (~100,000 rows)
- `client2-data.csv` (~100,000 rows)

### 3. Choose Connection Method

You have two options:

#### Option A: Local IP (Same Network)

- ✅ Fast and free
- ❌ Requires same WiFi network
- ❌ May need firewall configuration

**Find Server IP Address:**

```bash
# macOS/Linux
ifconfig | grep "inet " | grep -v 127.0.0.1

# Windows
ipconfig
```

Look for something like: `192.168.1.100`

#### Option B: Ngrok (Any Network)

- ✅ Works across different networks
- ✅ No firewall configuration
- ❌ Requires internet connection
- ❌ Slightly slower

**Setup Ngrok (Optional but recommended):**

```bash
# Install pyngrok (included in requirements.txt)
pip install pyngrok

# Get free ngrok token (optional but recommended)
# 1. Sign up at https://dashboard.ngrok.com/signup
# 2. Get token from https://dashboard.ngrok.com/get-started/your-authtoken
# 3. Set it:
export NGROK_AUTHTOKEN=your_token_here  # macOS/Linux
# or
set NGROK_AUTHTOKEN=your_token_here  # Windows
```

### 4. Start the Server (Your Laptop)

**Option A: Local IP**

```bash
python server.py
```

**Option B: Ngrok**

```bash
python server.py --use-ngrok
```

You should see:

- **Local IP**: Connection instructions with your IP address
- **Ngrok**: A public URL like `0.tcp.ngrok.io:12345` (copy this!)

### 5. Start Client 1 (Laptop 1)

**Option A: Local IP**

```bash
# Replace 192.168.1.100 with your server's IP
python client1.py 192.168.1.100:8080
```

**Option B: Ngrok**

```bash
# Replace with the ngrok URL shown by the server
python client1.py 0.tcp.ngrok.io:12345
```

### 6. Start Client 2 (Laptop 2)

**Option A: Local IP**

```bash
# Replace 192.168.1.100 with your server's IP
python client2.py 192.168.1.100:8080
```

**Option B: Ngrok**

```bash
# Replace with the ngrok URL shown by the server
python client2.py 0.tcp.ngrok.io:12345
```

### 7. Start Web Interfaces (Optional)

**On Client 1 machine:**

```bash
python client1_web.py
# Access at http://localhost:5002
```

**On Client 2 machine:**

```bash
python client2_web.py
# Access at http://localhost:5003
```

## What Happens Next?

1. Both clients connect to the server
2. Server sends initial model parameters to clients
3. Each client trains on their local data
4. Clients send updates back to server
5. Server aggregates updates using Federated Averaging
6. Server sends updated model back to clients
7. Process repeats for 10 rounds

## Adding New Data

### Via Web Interface

1. Open `http://localhost:5002` (Client 1) or `http://localhost:5003` (Client 2)
2. Fill in the form with crop data
3. Click "Add Entry"
4. Data is saved to `client1-data.csv` or `client2-data.csv`

### Via CSV Upload

1. Prepare a CSV file with columns: State, District, Market, Commodity, Variety, Grade, Arrival_Date, Min_Price, Max_Price, Modal_Price, Commodity_Code
2. Use the "Bulk Upload CSV" section in the web interface
3. Upload your CSV file

## Troubleshooting

**Can't connect to server?**

- **Local IP**: Check server IP address, ensure same WiFi network, check firewall (port 8080)
- **Ngrok**: Verify ngrok URL is correct, check internet connection, ensure server is running with `--use-ngrok`
- Try switching between local IP and ngrok methods
- See `NGROK_SETUP.md` for detailed ngrok troubleshooting

**Missing files?**

- Make sure `client1-data.csv` and `client2-data.csv` exist
- Run `split_dataset.py` if they're missing

**Port already in use?**

- Change port in `server.py` (line with `server_address="0.0.0.0:8080"`)
- Update client scripts to use new port

## Testing Locally (Single Machine)

If you want to test on a single machine:

```bash
# Terminal 1: Start server (local or ngrok)
python server.py
# or
python server.py --use-ngrok

# Terminal 2: Start client 1
python client1.py localhost:8080
# or with ngrok URL shown by server

# Terminal 3: Start client 2
python client2.py localhost:8080
# or with ngrok URL shown by server
```

## Connection Methods Comparison

| Feature  | Local IP           | Ngrok               |
| -------- | ------------------ | ------------------- |
| Speed    | ⚡ Fast            | 🐢 Slower           |
| Network  | Same WiFi required | Any network         |
| Setup    | Need IP address    | Just share URL      |
| Firewall | May need config    | No config           |
| Cost     | Free               | Free tier available |
| Best for | Same location      | Remote testing      |

See `NGROK_SETUP.md` for detailed ngrok setup instructions.

## Expected Training Time

- Initial setup: ~30 seconds
- Each round: ~1-5 minutes (depending on dataset size)
- Total (10 rounds): ~10-50 minutes

## Monitoring Progress

Watch the console output for:

- Training metrics (MAE, RMSE)
- Round completion
- Aggregation status

## Next Steps

- Add more clients by creating additional client scripts
- Adjust number of rounds in `server.py`
- Modify model hyperparameters in client scripts
- Experiment with different aggregation strategies
