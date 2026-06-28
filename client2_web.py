"""
Web interface for Client 2 to add/upload crop data
Simple black and white HTML interface
"""
from fastapi import FastAPI, Request, Form, File, UploadFile
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
import pandas as pd
import os
from datetime import datetime
import uvicorn

app = FastAPI(title="Client 2 - Crop Data Entry")

# Data file for client 2
CLIENT_DATA_FILE = "client2-data.csv"

HTML_TEMPLATE = """
<!DOCTYPE html>
<html>
<head>
    <title>Client 2 - Crop Data Entry</title>
    <style>
        * {
            margin: 0;
            padding: 0;
            box-sizing: border-box;
        }
        body {
            font-family: Arial, sans-serif;
            background-color: white;
            color: black;
            padding: 20px;
            max-width: 1200px;
            margin: 0 auto;
        }
        h1 {
            border-bottom: 2px solid black;
            padding-bottom: 10px;
            margin-bottom: 30px;
        }
        .section {
            margin-bottom: 40px;
            border: 1px solid black;
            padding: 20px;
        }
        .section h2 {
            margin-bottom: 20px;
            border-bottom: 1px solid black;
            padding-bottom: 10px;
        }
        .form-group {
            margin-bottom: 15px;
        }
        label {
            display: block;
            margin-bottom: 5px;
            font-weight: bold;
        }
        input, select {
            width: 100%;
            padding: 8px;
            border: 1px solid black;
            font-size: 14px;
        }
        button {
            background-color: black;
            color: white;
            padding: 10px 20px;
            border: none;
            cursor: pointer;
            font-size: 16px;
            margin-top: 10px;
        }
        button:hover {
            background-color: #333;
        }
        .message {
            padding: 10px;
            margin: 10px 0;
            border: 1px solid black;
        }
        .success {
            background-color: #f0f0f0;
            color: black;
        }
        .error {
            background-color: #f0f0f0;
            color: black;
            border-color: black;
        }
        table {
            width: 100%;
            border-collapse: collapse;
            margin-top: 20px;
        }
        th, td {
            border: 1px solid black;
            padding: 8px;
            text-align: left;
        }
        th {
            background-color: black;
            color: white;
        }
    </style>
</head>
<body>
    <h1>Client 2 - Crop Price Data Entry</h1>
    
    <div class="section">
        <h2>Add New Crop Entry</h2>
            <form id="addForm" method="post" action="/add" onsubmit="submitForm(event, 'addForm')">
            <div class="form-group">
                <label>State:</label>
                <input type="text" name="state" required>
            </div>
            <div class="form-group">
                <label>District:</label>
                <input type="text" name="district" required>
            </div>
            <div class="form-group">
                <label>Market:</label>
                <input type="text" name="market" required>
            </div>
            <div class="form-group">
                <label>Commodity:</label>
                <input type="text" name="commodity" required>
            </div>
            <div class="form-group">
                <label>Variety:</label>
                <input type="text" name="variety" required>
            </div>
            <div class="form-group">
                <label>Grade:</label>
                <input type="text" name="grade" required>
            </div>
            <div class="form-group">
                <label>Arrival Date (DD/MM/YYYY):</label>
                <input type="text" name="arrival_date" placeholder="DD/MM/YYYY" required>
            </div>
            <div class="form-group">
                <label>Min Price:</label>
                <input type="number" name="min_price" step="0.01" required>
            </div>
            <div class="form-group">
                <label>Max Price:</label>
                <input type="number" name="max_price" step="0.01" required>
            </div>
            <div class="form-group">
                <label>Modal Price:</label>
                <input type="number" name="modal_price" step="0.01" required>
            </div>
            <div class="form-group">
                <label>Commodity Code:</label>
                <input type="number" name="commodity_code" required>
            </div>
            <button type="submit">Add Entry</button>
        </form>
    </div>
    
    <div class="section">
        <h2>Bulk Upload CSV</h2>
        <form id="uploadForm" method="post" action="/upload" enctype="multipart/form-data" onsubmit="submitForm(event, 'uploadForm')">
            <div class="form-group">
                <label>Select CSV File:</label>
                <input type="file" name="file" accept=".csv" required>
            </div>
            <button type="submit">Upload CSV</button>
        </form>
    </div>
    
    <div class="section">
        <h2>Recent Entries (Last 10)</h2>
        <div id="recentEntries">
            Loading...
        </div>
    </div>
    
    <script>
        function submitForm(event, formId) {
            event.preventDefault();
            const form = document.getElementById(formId);
            const formData = new FormData(form);
            
            if (formId === 'addForm') {
                fetch('/add', {
                    method: 'POST',
                    body: formData
                })
                .then(response => response.json())
                .then(data => {
                    alert(data.message);
                    if (data.success) {
                        form.reset();
                        loadRecentEntries();
                    }
                })
                .catch(error => {
                    alert('Error: ' + error);
                });
            } else if (formId === 'uploadForm') {
                fetch('/upload', {
                    method: 'POST',
                    body: formData
                })
                .then(response => response.json())
                .then(data => {
                    alert(data.message);
                    if (data.success) {
                        form.reset();
                        loadRecentEntries();
                    }
                })
                .catch(error => {
                    alert('Error: ' + error);
                });
            }
        }
        
        function loadRecentEntries() {
            fetch('/recent')
                .then(response => response.json())
                .then(data => {
                    if (data.entries && data.entries.length > 0) {
                        let html = '<table><thead><tr>';
                        const headers = Object.keys(data.entries[0]);
                        headers.forEach(h => html += '<th>' + h + '</th>');
                        html += '</tr></thead><tbody>';
                        data.entries.forEach(entry => {
                            html += '<tr>';
                            headers.forEach(h => html += '<td>' + entry[h] + '</td>');
                            html += '</tr>';
                        });
                        html += '</tbody></table>';
                        html += '<p>Total entries: ' + data.total + '</p>';
                        document.getElementById('recentEntries').innerHTML = html;
                    } else {
                        document.getElementById('recentEntries').innerHTML = '<p>No entries found.</p>';
                    }
                })
                .catch(error => {
                    document.getElementById('recentEntries').innerHTML = '<p>Error loading entries.</p>';
                });
        }
        
        // Load recent entries on page load
        loadRecentEntries();
    </script>
</body>
</html>
"""

def ensure_data_file():
    """Ensure the client data file exists with headers"""
    if not os.path.exists(CLIENT_DATA_FILE):
        df = pd.DataFrame(columns=[
            'State', 'District', 'Market', 'Commodity', 'Variety', 'Grade',
            'Arrival_Date', 'Min_Price', 'Max_Price', 'Modal_Price', 'Commodity_Code'
        ])
        df.to_csv(CLIENT_DATA_FILE, index=False)

@app.get("/", response_class=HTMLResponse)
async def home():
    """Serve the main page"""
    ensure_data_file()
    return HTML_TEMPLATE

@app.post("/add")
async def add_entry(
    state: str = Form(...),
    district: str = Form(...),
    market: str = Form(...),
    commodity: str = Form(...),
    variety: str = Form(...),
    grade: str = Form(...),
    arrival_date: str = Form(...),
    min_price: float = Form(...),
    max_price: float = Form(...),
    modal_price: float = Form(...),
    commodity_code: int = Form(...)
):
    """Add a new entry to the CSV"""
    try:
        ensure_data_file()
        
        # Read existing data
        df = pd.read_csv(CLIENT_DATA_FILE)
        
        # Add new entry
        new_entry = {
            'State': state,
            'District': district,
            'Market': market,
            'Commodity': commodity,
            'Variety': variety,
            'Grade': grade,
            'Arrival_Date': arrival_date,
            'Min_Price': min_price,
            'Max_Price': max_price,
            'Modal_Price': modal_price,
            'Commodity_Code': commodity_code
        }
        
        df = pd.concat([df, pd.DataFrame([new_entry])], ignore_index=True)
        df.to_csv(CLIENT_DATA_FILE, index=False)
        
        return JSONResponse({
            "success": True,
            "message": f"Entry added successfully. Total entries: {len(df)}"
        })
    except Exception as e:
        return JSONResponse({
            "success": False,
            "message": f"Error adding entry: {str(e)}"
        }, status_code=500)

@app.post("/upload")
async def upload_csv(file: UploadFile = File(...)):
    """Upload and append CSV file"""
    try:
        ensure_data_file()
        
        # Read uploaded file
        contents = await file.read()
        new_df = pd.read_csv(pd.io.common.BytesIO(contents))
        
        # Validate columns
        required_columns = [
            'State', 'District', 'Market', 'Commodity', 'Variety', 'Grade',
            'Arrival_Date', 'Min_Price', 'Max_Price', 'Modal_Price', 'Commodity_Code'
        ]
        
        if not all(col in new_df.columns for col in required_columns):
            return JSONResponse({
                "success": False,
                "message": f"CSV must contain columns: {', '.join(required_columns)}"
            }, status_code=400)
        
        # Read existing data
        existing_df = pd.read_csv(CLIENT_DATA_FILE)
        
        # Append new data
        combined_df = pd.concat([existing_df, new_df], ignore_index=True)
        combined_df.to_csv(CLIENT_DATA_FILE, index=False)
        
        return JSONResponse({
            "success": True,
            "message": f"Uploaded {len(new_df)} entries. Total entries: {len(combined_df)}"
        })
    except Exception as e:
        return JSONResponse({
            "success": False,
            "message": f"Error uploading file: {str(e)}"
        }, status_code=500)

@app.get("/recent")
async def get_recent():
    """Get recent entries"""
    try:
        ensure_data_file()
        df = pd.read_csv(CLIENT_DATA_FILE)
        
        # Get last 10 entries
        recent = df.tail(10).to_dict('records')
        
        return JSONResponse({
            "success": True,
            "entries": recent,
            "total": len(df)
        })
    except Exception as e:
        return JSONResponse({
            "success": False,
            "message": str(e)
        }, status_code=500)

if __name__ == "__main__":
    print("Starting Client 2 Web Interface on http://localhost:5003")
    uvicorn.run(app, host="0.0.0.0", port=5003)

