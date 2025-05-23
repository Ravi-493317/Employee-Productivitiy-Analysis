import matplotlib
matplotlib.use('Agg')  # Use non-GUI backend to prevent threading issues

from flask import Flask, render_template, request, redirect, session, jsonify
import mysql.connector
import pickle
import numpy as np
import matplotlib.pyplot as plt
import io
import base64
import datetime

app = Flask(__name__)
app.secret_key = 'your_secret_key'

# Load trained model and scaler
with open("models/employee_performance_model.pkl", 'rb') as f:
    model = pickle.load(f)
with open("models/scaler.pkl", 'rb') as f:
    scaler = pickle.load(f)

# MySQL Database Configuration
db = mysql.connector.connect(
    host='localhost',
    user='root',
    password='root',
    database='employee_db'
)
cursor = db.cursor()

# Ensure required tables exist
cursor.execute("""
CREATE TABLE IF NOT EXISTS employees (
    id INT AUTO_INCREMENT PRIMARY KEY,
    name VARCHAR(255) UNIQUE,
    password VARCHAR(255)
)
""")

cursor.execute("""
CREATE TABLE IF NOT EXISTS attendance (
    id INT AUTO_INCREMENT PRIMARY KEY,
    employee_name VARCHAR(255),
    punch_in DATETIME,
    punch_out DATETIME,
    total_hours FLOAT DEFAULT 0,
    task_completion FLOAT DEFAULT 0,
    FOREIGN KEY (employee_name) REFERENCES employees(name)
)
""")
db.commit()

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/register', methods=['GET', 'POST'])
def register():
    if request.method == 'POST':
        name = request.form['name']
        password = request.form['password']
        cursor.execute("INSERT INTO employees (name, password) VALUES (%s, %s)", (name, password))
        db.commit()
        return redirect('/login')
    return render_template('register.html')

@app.route('/login', methods=['GET', 'POST'])
def login():
    if request.method == 'POST':
        name = request.form['name']
        password = request.form['password']
        cursor.execute("SELECT * FROM employees WHERE name=%s AND password=%s", (name, password))
        user = cursor.fetchone()
        if user:
            session['employee'] = name
            if name.lower() == "hr":
                return render_template('predict.html')  # Directly render the page
            return redirect('/employee_dashboard')
        else:
            return "Invalid credentials!"
    return render_template('login.html')

@app.route('/logout')
def logout():
    session.pop('employee', None)
    return redirect('/login')

@app.route('/employee_dashboard', methods=['GET', 'POST'])
def employee_dashboard():
    if 'employee' not in session:
        return redirect('/login')
    
    name = session['employee']
    cursor.execute("SELECT punch_in, punch_out, total_hours, task_completion FROM attendance WHERE employee_name=%s ORDER BY id DESC LIMIT 1", (name,))
    record = cursor.fetchone()
    
    punch_status = "Punch In" if not record or record[1] else "Punch Out"
    
    if request.method == 'POST':
        if punch_status == "Punch In":
            cursor.execute("INSERT INTO attendance (employee_name, punch_in) VALUES (%s, NOW())", (name,))
        else:
            cursor.execute("SELECT punch_in FROM attendance WHERE employee_name=%s AND punch_out IS NULL ORDER BY id DESC LIMIT 1", (name,))
            punch_in_time = cursor.fetchone()
            
            if punch_in_time:
                punch_in = punch_in_time[0]
                punch_out = datetime.datetime.now()
                total_seconds = (punch_out - punch_in).total_seconds()
                total_hours = total_seconds / 3600  # Convert seconds to hours

                cursor.execute("""
                    UPDATE attendance 
                    SET punch_out=%s, total_hours=%s 
                    WHERE employee_name=%s AND punch_out IS NULL 
                    ORDER BY id DESC LIMIT 1
                """, (punch_out, round(total_hours, 2), name))
        
        db.commit()
        return redirect('/employee_dashboard')
    
    return render_template('employee_dashboard.html', name=name, punch_status=punch_status, record=record)

@app.route('/predict', methods=['POST'])
def predict():
    if 'employee' not in session:
        return redirect('/login')

    name = request.form.get('name')
    selected_date = request.form.get('date')

    if not name or not selected_date:
        return "Error: Missing employee name or date!", 400

    try:
        # Fetch total working hours and task completion sum for the selected date
        cursor.execute("""
            SELECT SUM(total_hours), SUM(task_completion) 
            FROM attendance 
            WHERE employee_name = %s AND DATE(punch_in) = %s
        """, (name, selected_date))
        
        result = cursor.fetchone()

        if result and result[0] is not None:
            total_hours, task_completion = result

            # Ensure data is properly formatted for model prediction
            input_data = np.array([[total_hours, task_completion]])
            scaled_data = scaler.transform(input_data)  
            prediction = model.predict(scaled_data)[0]

            productivity_status = "Productive" if prediction == 1 else "Not Productive"

            # Generate Graph
            img = io.BytesIO()
            plt.figure(figsize=(6, 4))
            plt.bar(['Total Hours', 'Task Completion'], [total_hours, task_completion], color=['blue', 'green'])
            plt.ylim(0, 100)
            plt.title(f'Performance Analysis for {name} on {selected_date}')
            plt.xlabel("Metrics")
            plt.ylabel("Value")
            plt.savefig(img, format='png', bbox_inches='tight')  
            plt.close()
            img.seek(0)

            graph_url = base64.b64encode(img.getvalue()).decode()

            return render_template('result.html', name=name, status=productivity_status, graph_url=graph_url)
        else:
            return f"No records found for {name} on {selected_date}!", 404

    except Exception as e:
        return f"An error occurred: {str(e)}", 500

@app.route('/store_task_completion', methods=['POST'])
def store_task_completion():
    if 'employee' not in session:
        return redirect('/login')
    
    name = session['employee']
    task_completion = float(request.form['task_completion'])

    cursor.execute("SELECT total_hours FROM attendance WHERE employee_name=%s ORDER BY id DESC LIMIT 1", (name,))
    result = cursor.fetchone()

    if result:
        cursor.execute("UPDATE attendance SET task_completion=%s WHERE employee_name=%s AND punch_out IS NOT NULL ORDER BY id DESC LIMIT 1", (task_completion, name))
        db.commit()
    
    return redirect('/employee_dashboard')  # Stay on the same page


@app.route('/get_employees_by_date')
def get_employees_by_date():
    selected_date = request.args.get('date')
    if not selected_date:
        return jsonify([])

    cursor.execute("""
        SELECT DISTINCT employee_name FROM attendance
        WHERE DATE(punch_in) = %s
    """, (selected_date,))
    employees = [row[0] for row in cursor.fetchall()]
    
    return jsonify(employees)

if __name__ == '__main__':
    app.run(debug=True)
