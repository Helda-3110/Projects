from flask import Flask, render_template, request, redirect, session
from sqlalchemy import create_engine, Column, String
from sqlalchemy.orm import sessionmaker
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.exc import IntegrityError
import bcrypt
from flask_mysqldb import MySQL
import numpy as np
import matplotlib.pyplot as plt
import itertools

# Suppress warnings
import warnings
warnings.filterwarnings("ignore")

app.secret_key = 'your_secret_key'

# MySQL configurations
app.config['MYSQL_HOST'] = 'localhost'
app.config['MYSQL_USER'] = 'root'
app.config['MYSQL_PASSWORD'] = '996567'
app.config['MYSQL_DB'] = 'lung_cancer_prediction_system'

mysql = MySQL(app)
def get_db():
    return mysql.connection

# Create a Flask app


# Create a database engine
engine = create_engine(db_connection_str)

# Create a session
Session = sessionmaker(bind=engine)
db_session = Session()

# Create a base class for declarative class definitions
Base = declarative_base()

# Define a User class to represent the users table
class User(Base):
    __tablename__ = 'users'
    username = Column(String(100), primary_key=True)
    password = Column(String(100))

# Function to authenticate user
def authenticate(username, password):
    user = db_session.query(User).filter_by(username=username).first()
    if user and bcrypt.checkpw(password.encode('utf-8'), user.password.encode('utf-8')):
        return True
    else:
        return False

# Function to add a new user to the database
def add_user(username, password):
    hashed_password = bcrypt.hashpw(password.encode('utf-8'), bcrypt.gensalt())
    new_user = User(username=username, password=hashed_password.decode('utf-8'))
    try:
        db_session.add(new_user)
        db_session.commit()
        return True
    except IntegrityError:
        db_session.rollback()
        return False

@app.route('/login', methods=['GET', 'POST'])
def login():
    if request.method == 'POST':
        username = request.form['username']
        password = request.form['password']
        try:
            cur = mysql.connection.cursor()
            cur.execute("SELECT * FROM login WHERE username = %s", (username,))
            user = cur.fetchone()

            if user and bcrypt.checkpw(password.encode('utf-8'), user[2].encode('utf-8')):
                session['username'] = username
                return redirect('/')
            else:
                error_message = 'Invalid username or password'
                return render_template('login.html', message=error_message)
        except Exception as e:
            print("Error during login:", e)  # Print detailed error message
            error_message = 'An error occurred while logging in'
            return render_template('login.html', message=error_message)
        finally:
            cur.close()
    else:
        if 'username' in session:
            return redirect('/')
        else:
            return render_template('login.html')

@app.route('/signup', methods=['GET', 'POST'])
def signup():
    if request.method == 'POST':
        username = request.form['username']
        password = request.form['password']
        try:
            hashed_password = bcrypt.hashpw(password.encode('utf-8'), bcrypt.gensalt())  # Hash the password
            cur = mysql.connection.cursor()
            cur.execute("INSERT INTO login (username, password) VALUES (%s, %s)", (username, hashed_password.decode('utf-8')))
            mysql.connection.commit()
            return redirect('/login')
        except Exception as e:
            print("Error during signup:", e)
            error_message = 'An error occurred while signing up'
            return render_template('signup.html', message=error_message)
        finally:
            cur.close()

    else:
        if 'username' in session:
            return redirect('/')
        else:
            return render_template('signup.html')

@app.route('/logout')
def logout():
    session.pop('username', None)
    return redirect('/login')

@app.route('/delete_account', methods=['GET', 'POST'])
def delete_account():
    if 'username' in session:
        try:
            cur = mysql.connection.cursor()
            cur.execute("DELETE FROM login WHERE username = %s", (session['username'],))
            mysql.connection.commit()
            session.pop('username', None)
            return redirect('/login')
        except Exception as e:
            print("Error:", e)
            error_message = 'An error occurred while deleting the account'
            return render_template('error.html', message=error_message)  # Render an error template
        finally:
            cur.close()
    else:
        return redirect('/login')  # Redirect to login page if not logged in

@app.route('/forgot_password', methods=['GET', 'POST'])
def forgot_password():
    global cur
    if request.method == 'POST':
        username = request.form['username']
        new_password = request.form['new_password']

        try:
            cur = mysql.connection.cursor()
            cur.execute("SELECT * FROM login WHERE username = %s", (username,))
            user = cur.fetchone()

            if not user:  # If user does not exist
                error_message = 'Invalid username'
                return render_template('forgot.html', message=error_message)

            hashed_password = bcrypt.hashpw(new_password.encode('utf-8'), bcrypt.gensalt())  # Hash the new password
            cur.execute("UPDATE login SET password = %s WHERE username = %s",
                        (hashed_password.decode('utf-8'), username))
            mysql.connection.commit()
            print("Password updated successfully")  # Add debug message
            return redirect('/login')  # Redirect to login page after password update
        except Exception as e:
            print("Error during password update:", e)  # Print error message
            error_message = 'An error occurred while updating the password'
            return render_template('forgot.html', message=error_message)
        finally:
            cur.close()
    else:
        return render_template('forgot.html')