import streamlit as st
import mysql.connector

# Connect to MySQL database
db = mysql.connector.connect(
    host="localhost",
    user="root",
    password="Helda2004",
    database="car_accident"
)
cursor = db.cursor()

# Initialize session state
if 'page' not in st.session_state:
    st.session_state.page = "login"

# Login page
def login():
    st.title("Login")
    username = st.text_input("Username")
    password = st.text_input("Password", type="password")
    if st.button("Login"):
        cursor.execute("SELECT * FROM users WHERE username=%s AND password=%s", (username, password))
        user = cursor.fetchone()
        if user:
            st.session_state.logged_in = True
            st.success("Logged in successfully! Redirecting...")
            st.write('<meta http-equiv="refresh" content="0;URL=http://localhost:8501/"/>', unsafe_allow_html=True)
        else:
            st.error("Invalid username or password")
    if st.button("If you don't have an account, sign up"):
        st.session_state.page = "signup"

# Signup page
def signup():
    st.title("Signup")
    new_username = st.text_input("New Username")
    new_password = st.text_input("New Password", type="password")
    if st.button("Signup"):
        cursor.execute("INSERT INTO users (username, password) VALUES (%s, %s)", (new_username, new_password))
        db.commit()
        st.success("Signup successful! Please login.")
        st.session_state.page = "login"
    if st.button("Back to Login"):
        st.session_state.page = "login"

# Determine which page to show based on the value of st.session_state.page
if st.session_state.page == "signup":
    signup()
else:
    login()
