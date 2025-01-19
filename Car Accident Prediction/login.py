import tkinter as tk
from tkinter import messagebox
import mysql.connector

# Connect to MySQL database
db = mysql.connector.connect(
    host="localhost",
    user="root",
    password="Helda2004",
    database="car_accident"
)
cursor = db.cursor()

# Initialize login window
login_window = tk.Tk()
login_window.title("Login")


# Function to handle login
def login():
    username = username_entry.get()
    password = password_entry.get()
    role = role_var.get()  # Get the selected role

    cursor.execute("SELECT * FROM users WHERE username=%s AND password=%s", (username, password))
    user = cursor.fetchone()
    if user:
        # Insert username and role into login_user table
        cursor.execute("INSERT INTO login_user (username, role) VALUES (%s, %s)", (username, role))
        db.commit()

        messagebox.showinfo("Success", "Logged in successfully! Redirecting...")
        login_window.withdraw()
        # Redirect to the main page (tk.py)
        import main
        main.main_page()
    else:
        messagebox.showerror("Error", "Invalid username or password")


# Username, password, and role labels and entry fields
username_label = tk.Label(login_window, text="Username:")
username_label.grid(row=0, column=0, padx=10, pady=5, sticky="e")
username_entry = tk.Entry(login_window)
username_entry.grid(row=0, column=1, padx=10, pady=5)

password_label = tk.Label(login_window, text="Password:")
password_label.grid(row=1, column=0, padx=10, pady=5, sticky="e")
password_entry = tk.Entry(login_window, show="*")
password_entry.grid(row=1, column=1, padx=10, pady=5)

role_label = tk.Label(login_window, text="Role:")
role_label.grid(row=2, column=0, padx=10, pady=5, sticky="e")
role_var = tk.StringVar()
role_dropdown = tk.OptionMenu(login_window, role_var, "User", "Staff")
role_dropdown.grid(row=2, column=1, padx=10, pady=5)

# Login button
login_button = tk.Button(login_window, text="Login", command=login)
login_button.grid(row=3, column=0, columnspan=2, pady=10)


# Function to switch to signup page
def switch_to_signup():
    login_window.withdraw()  # Hide the login window
    signup_window = tk.Tk()
    signup_window.title("Signup")
    signup_window.geometry("300x150")

    # Signup fields
    new_username_label = tk.Label(signup_window, text="New Username:")
    new_username_label.grid(row=0, column=0)
    new_username_entry = tk.Entry(signup_window)
    new_username_entry.grid(row=0, column=1)

    new_password_label = tk.Label(signup_window, text="New Password:")
    new_password_label.grid(row=1, column=0)
    new_password_entry = tk.Entry(signup_window, show="*")
    new_password_entry.grid(row=1, column=1)

    def signup():
        new_username = new_username_entry.get()
        new_password = new_password_entry.get()
        cursor.execute("INSERT INTO users (username, password) VALUES (%s, %s)", (new_username, new_password))
        db.commit()
        messagebox.showinfo("Success", "Signup successful! Please login.")
        signup_window.destroy()  # Close the signup window
        login_window.deiconify()  # Bring back the login window

    signup_button = tk.Button(signup_window, text="Signup", command=signup)
    signup_button.grid(row=2, column=0, columnspan=2)

    signup_window.protocol("WM_DELETE_WINDOW", lambda: login_window.deiconify())  # Handle window close event


# Signup button
signup_button = tk.Button(login_window, text="If you don't have an account, sign up", command=switch_to_signup)
signup_button.grid(row=4, column=0, columnspan=2, pady=10)

login_window.mainloop()