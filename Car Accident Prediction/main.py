import tkinter as tk
from tkinter import ttk, messagebox, filedialog
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from sqlalchemy import create_engine
from sklearn.metrics import confusion_matrix
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
import itertools
from tqdm import tqdm
from tkhtmlview import HTMLLabel
from bs4 import BeautifulSoup

db_connection_str = 'mysql://root:Helda2004@localhost/car_accident'

def create_connection():
    """Creates a connection to the MySQL database."""
    return create_engine(db_connection_str)

def fetch_data():
    """Fetches data from the 'accidents' table in the database."""
    engine = create_connection()
    conn = engine.raw_connection()  # Obtain a raw DBAPI2 connection object from the Engine
    sql_query = """
        SELECT
            `Accident_Date`, `Day_of_Week`, `Junction_Control`, `Junction_Detail`,
            `Accident_Severity`, `Light_Conditions`, `Local_Authority_(District)`,
            `Carriageway_Hazards`, `Number_of_Vehicles`, `Police_Force`,
            `Road_Surface_Conditions`, `Road_Type`, `Speed_limit`, `Time`,
            `Urban_or_Rural_Area`, `Weather_Conditions`, `Vehicle_Type`
        FROM
            accidents
    """
    df = pd.read_sql(sql_query, conn)
    conn.close()  # Close the connection
    return df

def prepare_data(df):
    """Prepares the data for modeling and visualization."""
    # Remove duplicates (assuming duplicates are not desired)
    df = df.drop_duplicates()

    # Define target variable
    df['is_serious_fatal'] = (df['Accident_Severity'] != 'Slight').astype(int)

    # Extract Month and Year from Accident_Date without converting to datetime
    df['Month'] = df['Accident_Date'].str.split('/').str[0]  # Extract month
    df['Year'] = df['Accident_Date'].str.split('/').str[2]  # Extract year

    # Select columns of interest
    selected_cols = ['is_serious_fatal', 'Day_of_Week', 'Month', 'Year',
                     'Junction_Control', 'Junction_Detail', 'Light_Conditions',
                     'Local_Authority_(District)', 'Carriageway_Hazards',
                     'Number_of_Vehicles', 'Police_Force',
                     'Road_Surface_Conditions', 'Road_Type', 'Speed_limit', 'Time',
                     'Urban_or_Rural_Area', 'Weather_Conditions', 'Vehicle_Type']
    df = df[selected_cols]

    # Prepare categorical variables for modeling
    for col in tqdm(df.columns):
        if col != 'is_serious_fatal':
            df[col] = df[col].fillna('None').astype(str)
            frequency_map = df[col].value_counts(normalize=True)
            df[col + '_Frequency'] = df[col].map(frequency_map)

    # Perform data transformation (consider incorporating transformations based on domain knowledge)
    # ... (e.g., handle missing values, categorical encoding)

    return df

def perform_analysis(df):
    plt.close('all')  # Close all existing plots
    fig = plt.figure(figsize=(6, 4))
    model = RandomForestClassifier()  # Instantiate the model
    X = df.drop('is_serious_fatal', axis=1)
    y_true = df['is_serious_fatal']

    # One-hot encode categorical variables
    X_encoded = pd.get_dummies(X)

    # Split data into train and test sets
    X_train, X_test, y_train, y_test = train_test_split(X_encoded, y_true, test_size=0.5, random_state=42)

    # Train the model
    model.fit(X_train, y_train)

    # Make predictions
    y_pred = model.predict(X_test)

    # Calculate the confusion matrix
    cm = confusion_matrix(y_test, y_pred)

    # Plot the confusion matrix
    class_names = ['Slight', 'Severe/Fatal']  # Positive class should come last
    plot_confusion_matrix(cm, class_names, fig)

def plot_confusion_matrix(cm, classes, fig):
    """
    This function plots a confusion matrix.

    Parameters:
        cm (array-like): Confusion matrix as returned by sklearn.metrics.confusion_matrix.
        classes (list): List of class names, e.g., ['Class 0', 'Class 1'].
        fig (matplotlib figure): Matplotlib figure to plot on.
    """
    # Ensure cm is an ndarray
    cm = np.array(cm)

    # Display the confusion matrix
    plt.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
    plt.title('Confusion Matrix')
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    # Set x and y-axis labels
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)
    fmt = '.0f'
    thresh = cm.max() / 2.0
    # Add text annotations to the plot
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, format(cm[i, j], fmt), horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")

    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    plt.tight_layout()
    plt.grid()

def graph(df):
    plt.close('all')  # Close all existing plots
    fig, ax = plt.subplots(2, 1, figsize=(10, 12))

    sns.countplot(x='Day_of_Week', data=df, ax=ax[0])
    ax[0].set_title('Count of Accidents by Day of the Week')
    ax[0].set_xlabel('Day of the Week')
    ax[0].set_ylabel('Count')
    ax[0].tick_params(axis='x', rotation=45)

    sns.countplot(x='Junction_Control', data=df, ax=ax[1])
    ax[1].set_title('Count of Accidents by Junction Control')
    ax[1].set_xlabel('Junction Control')
    ax[1].set_ylabel('Count')
    ax[1].tick_params(axis='x', rotation=45)

    plt.tight_layout()
    return fig

def load_graph_page(graph_page):
    graph_frame = tk.Frame(graph_page)
    graph_frame.pack(fill='both', expand=True)

    df = fetch_data()
    df = prepare_data(df)
    graph_fig = graph(df)

    graph_canvas = tk.Canvas(graph_frame)
    graph_canvas.pack(side='left', fill='both', expand=True)

    scrollbar = ttk.Scrollbar(graph_frame, orient='vertical', command=graph_canvas.yview)
    scrollbar.pack(side='right', fill='y')
    graph_canvas.configure(yscrollcommand=scrollbar.set)

    graph_inner_frame = tk.Frame(graph_canvas)
    graph_canvas.create_window((0, 0), window=graph_inner_frame, anchor='nw')

    graph_fig_canvas = FigureCanvasTkAgg(graph_fig, master=graph_inner_frame)
    graph_fig_canvas.draw()
    graph_fig_canvas.get_tk_widget().pack(fill='both', expand=True)

    graph_inner_frame.bind("<Configure>", lambda event, canvas=graph_canvas: update_scroll_region(event, canvas))

def update_scroll_region(event, canvas):
    canvas.configure(scrollregion=canvas.bbox("all"))

def show_data(df):
    # Display a subset of the data with download functionality
    top_window = tk.Toplevel()
    top_window.title("Data Display")
    top_window.geometry("800x600")
    data_frame = tk.Frame(top_window)
    data_frame.pack(fill="both", expand=True)
    data_text = tk.Text(data_frame, wrap="none")
    data_text.pack(side="left", fill="both", expand=True)
    scroll_y = tk.Scrollbar(data_frame, orient="vertical", command=data_text.yview)
    scroll_y.pack(side="right", fill="y")
    scroll_x = tk.Scrollbar(data_frame, orient="horizontal", command=data_text.xview)
    scroll_x.pack(side="bottom", fill="x")
    data_text.configure(yscrollcommand=scroll_y.set, xscrollcommand=scroll_x.set)
    data_text.insert("1.0", df.head(200).to_string(index=False))
    save_button = tk.Button(top_window, text="Save Data", command=lambda: save_data(df))
    save_button.pack()

def save_data(df):
    file_path = filedialog.asksaveasfilename(defaultextension=".csv", filetypes=[("CSV files", "*.csv")])
    if file_path:
        df.head(200).to_csv(file_path, index=False)
        messagebox.showinfo("Save Data", "Data saved successfully!")

def modify_data():
    """Allows modification of the data."""
    modify_window = tk.Toplevel()
    modify_window.title("Modify Data")
    modify_window.geometry("800x600")

    # Connect to the database
    conn = create_connection()

    selected_action = tk.StringVar()
    selected_action.set("Add Record")  # Default value

    # Function to add a new record
    def add_record():
        """Allows adding a new record."""
        add_window = tk.Toplevel()
        add_window.title("Add Record")
        add_window.geometry("800x600")

        # Labels and input fields for adding a new record
        label_accident_date = tk.Label(add_window, text="Accident Date:")
        label_accident_date.grid(row=0, column=0)
        entry_accident_date = tk.Entry(add_window)
        entry_accident_date.grid(row=0, column=1)

        label_day_of_week = tk.Label(add_window, text="Day of Week:")
        label_day_of_week.grid(row=1, column=0)
        day_of_week_options = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
        day_of_week_var = tk.StringVar()
        day_of_week_dropdown = tk.OptionMenu(add_window, day_of_week_var, *day_of_week_options)
        day_of_week_dropdown.grid(row=1, column=1)

        label_junction_control = tk.Label(add_window, text="Junction Control:")
        label_junction_control.grid(row=2, column=0)
        junction_control_options = ['Data missing', 'Out of Range', 'Give way', 'Uncontrolled']
        junction_control_var = tk.StringVar()
        junction_control_dropdown = tk.OptionMenu(add_window, junction_control_var, *junction_control_options)
        junction_control_dropdown.grid(row=2, column=1)

        label_junction_detail = tk.Label(add_window, text="Junction Detail:")
        label_junction_detail.grid(row=3, column=0)
        junction_detail_options = ['T Junction', 'Not at Junction', 'Staggered Junction']
        junction_detail_var = tk.StringVar()
        junction_detail_dropdown = tk.OptionMenu(add_window, junction_detail_var, *junction_detail_options)
        junction_detail_dropdown.grid(row=3, column=1)

        label_accident_severity = tk.Label(add_window, text="Accident Severity:")
        label_accident_severity.grid(row=4, column=0)
        accident_severity_options = ['Serious', 'Slight']
        accident_severity_var = tk.StringVar()
        accident_severity_dropdown = tk.OptionMenu(add_window, accident_severity_var, *accident_severity_options)
        accident_severity_dropdown.grid(row=4, column=1)

        label_light_conditions = tk.Label(add_window, text="Light Conditions:")
        label_light_conditions.grid(row=5, column=0)
        light_conditions_options = ['Daylight', 'Darkness - lights lit', 'Darkness - no lighting',
                                    'Darkness - lighting unknown']
        light_conditions_var = tk.StringVar()
        light_conditions_dropdown = tk.OptionMenu(add_window, light_conditions_var, *light_conditions_options)
        light_conditions_dropdown.grid(row=5, column=1)

        label_local_authority_district = tk.Label(add_window, text="Local Authority (District):")
        label_local_authority_district.grid(row=6, column=0)
        local_authority_district_options = ['Eden', 'Peterborough', 'South Cambridgeshire', 'Copeland',
                                            'South Lakeland', 'Huntingdonshire', 'Barrow-in-Furness',
                                            'Staffordshire Moorlands', 'Cannock Chase', 'Allerdale']
        local_authority_district_var = tk.StringVar()
        local_authority_district_dropdown = tk.OptionMenu(add_window, local_authority_district_var,
                                                          *local_authority_district_options)
        local_authority_district_dropdown.grid(row=6, column=1)

        label_carriageway_hazards = tk.Label(add_window, text="Carriageway Hazards:")
        label_carriageway_hazards.grid(row=7, column=0)
        carriageway_hazards_options = ['None', 'Previous accident', 'Other object on road']
        carriageway_hazards_var = tk.StringVar()
        carriageway_hazards_dropdown = tk.OptionMenu(add_window, carriageway_hazards_var, *carriageway_hazards_options)
        carriageway_hazards_dropdown.grid(row=7, column=1)

        label_number_of_vehicles = tk.Label(add_window, text="Number of Vehicles:")
        label_number_of_vehicles.grid(row=8, column=0)
        entry_number_of_vehicles = tk.Entry(add_window)
        entry_number_of_vehicles.grid(row=8, column=1)

        label_police_force = tk.Label(add_window, text="Police Force:")
        label_police_force.grid(row=9, column=0)
        police_force_options = ['Cumbria', 'Staffordshire']
        police_force_var = tk.StringVar()
        police_force_dropdown = tk.OptionMenu(add_window, police_force_var, *police_force_options)
        police_force_dropdown.grid(row=9, column=1)

        label_road_surface_conditions = tk.Label(add_window, text="Road Surface Conditions:")
        label_road_surface_conditions.grid(row=10, column=0)
        road_surface_conditions_options = ['Frost or ice', 'Dry', 'Wet or damp']
        road_surface_conditions_var = tk.StringVar()
        road_surface_conditions_dropdown = tk.OptionMenu(add_window, road_surface_conditions_var,
                                                         *road_surface_conditions_options)
        road_surface_conditions_dropdown.grid(row=10, column=1)

        label_road_type = tk.Label(add_window, text="Road Type:")
        label_road_type.grid(row=11, column=0)
        road_type_options = ['Single carriageway', 'Roundabout', 'Dual carriageway']
        road_type_var = tk.StringVar()
        road_type_dropdown = tk.OptionMenu(add_window, road_type_var, *road_type_options)
        road_type_dropdown.grid(row=11, column=1)

        label_speed_limit = tk.Label(add_window, text="Speed Limit:")
        label_speed_limit.grid(row=12, column=0)
        entry_speed_limit = tk.Entry(add_window)
        entry_speed_limit.grid(row=12, column=1)

        label_time = tk.Label(add_window, text="Time:")
        label_time.grid(row=13, column=0)
        entry_time = tk.Entry(add_window)
        entry_time.grid(row=13, column=1)

        label_urban_or_rural_area = tk.Label(add_window, text="Urban or Rural Area:")
        label_urban_or_rural_area.grid(row=14, column=0)
        urban_or_rural_area_options = ['Urban', 'Rural Area']
        urban_or_rural_area_var = tk.StringVar()
        urban_or_rural_area_dropdown = tk.OptionMenu(add_window, urban_or_rural_area_var, *urban_or_rural_area_options)
        urban_or_rural_area_dropdown.grid(row=14, column=1)

        label_weather_conditions = tk.Label(add_window, text="Weather Conditions:")
        label_weather_conditions.grid(row=15, column=0)
        weather_conditions_options = ['Fine no high winds', 'Fine + high winds', 'Raining + high winds',
                                      'Raining no high winds', 'Fog or mist', 'Other']
        weather_conditions_var = tk.StringVar()
        weather_conditions_dropdown = tk.OptionMenu(add_window, weather_conditions_var, *weather_conditions_options)
        weather_conditions_dropdown.grid(row=15, column=1)

        # Frame for Vehicle Type selection
        label_vehicle_type = tk.Label(add_window, text="Vehicle Type:")
        label_vehicle_type.grid(row=16, column=0)
        vehicle_type_options = ['Car', 'Motorcycle', 'Goods Lorry', 'Bus', 'Van', 'other']
        vehicle_type_var = tk.StringVar()
        vehicle_type_multiselect = tk.Listbox(add_window, selectmode=tk.MULTIPLE, height=len(vehicle_type_options))
        for i, option in enumerate(vehicle_type_options):
            vehicle_type_multiselect.insert(i, option)
        vehicle_type_multiselect.grid(row=16, column=1)

        # Function to add the record
        def add():
            print("data is Added successfully")
            # Fetch values from input fields
            accident_date = entry_accident_date.get()
            day_of_week = day_of_week_var.get()
            junction_control = junction_control_var.get()
            junction_detail = junction_detail_var.get()
            accident_severity = accident_severity_var.get()
            light_conditions = light_conditions_var.get()
            local_authority_district = local_authority_district_var.get()
            carriageway_hazards = carriageway_hazards_var.get()
            number_of_vehicles = entry_number_of_vehicles.get()
            police_force = police_force_var.get()
            road_surface_conditions = road_surface_conditions_var.get()
            road_type = road_type_var.get()
            speed_limit = entry_speed_limit.get()
            time = entry_time.get()
            urban_or_rural_area = urban_or_rural_area_var.get()
            weather_conditions = weather_conditions_var.get()
            vehicle_type = [vehicle_type_multiselect.get(idx) for idx in vehicle_type_multiselect.curselection()]

            # Store the record in the MySQL database
            add_record_to_database(accident_date, day_of_week, junction_control, junction_detail, accident_severity,
                                   light_conditions, local_authority_district, carriageway_hazards, number_of_vehicles,
                                   police_force, road_surface_conditions, road_type, speed_limit, time,
                                   urban_or_rural_area, weather_conditions, vehicle_type)

            # Show success message
            messagebox.showinfo("Success", "The record is added successfully. Now redirecting to Modify Data page.")
            add_window.destroy()  # Close the add window

        # Button to add the record
        add_button = tk.Button(add_window, text="Add", command=add)
        add_button.grid(row=17, columnspan=2)

        add_window.mainloop()

    def add_record_to_database(conn, accident_date, day_of_week, junction_control, junction_detail, accident_severity,
                               light_conditions, local_authority_district, carriageway_hazards, number_of_vehicles,
                               police_force, road_surface_conditions, road_type, speed_limit, time,
                               urban_or_rural_area, weather_conditions, vehicle_type):
        # Connect to the database and execute the SQL query to add the record
        cursor = conn.cursor()

        # Example SQL query to insert data into the table
        insert_query = """
            INSERT INTO accidents (Accident_Date, Day_of_Week, Junction_Control, junction_detail, accident_severity,
                               light_conditions, local_authority_district, carriageway_hazards, number_of_vehicles,
                               police_force, road_surface_conditions, road_type, speed_limit, time,
                               urban_or_rural_area, weather_conditions, vehicle_type)
            VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
            """
        vehicle_types_str = ', '.join(vehicle_type)

        record_data = (accident_date, day_of_week, junction_control, junction_detail, accident_severity,
                       light_conditions, local_authority_district, carriageway_hazards, number_of_vehicles,
                       police_force, road_surface_conditions, road_type, speed_limit, time,
                       urban_or_rural_area, weather_conditions, vehicle_types_str)
        try:
            # Execute the SQL query
            cursor.execute(insert_query, record_data)
            # Commit the transaction
            conn.commit()
        except Exception as e:
            # Rollback the transaction in case of error
            conn.rollback()
            # Display error message
            messagebox.showerror("Error", f"An error occurred: {e}")
        finally:
            # Close the cursor and database connection
            cursor.close()
            conn.close()
        pass

    # Function to delete a record
    def delete_record():
        # Show field for entering record ID
        id_frame.grid(row=2, column=0, columnspan=2)
        add_frame.grid_forget()
        modify_frame.grid_forget()

    # Function to modify a record
    def modify_record():
        # Show fields for modifying a record
        modify_frame.grid(row=2, column=0, columnspan=2)
        add_frame.grid_forget()
        id_frame.grid_forget()

    # Dropdown for Select an Action
    label_action = tk.Label(modify_window, text="Select an Action:")
    label_action.grid(row=0, column=0)
    action_options = ["Add Record", "Delete Record", "Modify Record"]
    dropdown_action = tk.OptionMenu(modify_window, selected_action, *action_options)
    dropdown_action.grid(row=0, column=1)

    # Frame for Add Record fields
    add_frame = tk.Frame(modify_window)

    # Add Record Fields
    label_accident_date = tk.Label(add_frame, text="Accident Date:")
    label_accident_date.grid(row=0, column=0)
    entry_accident_date = tk.Entry(add_frame)
    entry_accident_date.grid(row=0, column=1)

    label_day_of_week = tk.Label(add_frame, text="Day of Week:")
    label_day_of_week.grid(row=1, column=0)
    day_of_week_options = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
    day_of_week_var = tk.StringVar()
    day_of_week_dropdown = tk.OptionMenu(add_frame, day_of_week_var, *day_of_week_options)
    day_of_week_dropdown.grid(row=1, column=1)

    label_junction_control = tk.Label(add_frame, text="Junction Control:")
    label_junction_control.grid(row=2, column=0)
    junction_control_var = tk.StringVar()
    junction_control_dropdown = tk.OptionMenu(add_frame, junction_control_var, 'Data missing', 'Out of Range',
                                              'Give way', 'Uncontrolled')
    junction_control_dropdown.grid(row=2, column=1)

    # Frame for Modify Record fields
    modify_frame = tk.Frame(modify_window)

    # Modify Record Fields
    label_modified_accident_date = tk.Label(modify_frame, text="Modified Accident Date:")
    label_modified_accident_date.grid(row=0, column=0)
    entry_modified_accident_date = tk.Entry(modify_frame)
    entry_modified_accident_date.grid(row=0, column=1)

    label_modified_day_of_week = tk.Label(modify_frame, text="Modified Day of Week:")
    label_modified_day_of_week.grid(row=1, column=0)
    modified_day_of_week_var = tk.StringVar()
    modified_day_of_week_dropdown = tk.OptionMenu(modify_frame, modified_day_of_week_var, *day_of_week_options)
    modified_day_of_week_dropdown.grid(row=1, column=1)

    label_modified_junction_control = tk.Label(modify_frame, text="Modified Junction Control:")
    label_modified_junction_control.grid(row=2, column=0)
    modified_junction_control_var = tk.StringVar()
    modified_junction_control_dropdown = tk.OptionMenu(modify_frame, modified_junction_control_var, 'Data missing',
                                                        'Out of Range', 'Give way', 'Uncontrolled')
    modified_junction_control_dropdown.grid(row=2, column=1)

    # Frame for Delete Record field
    id_frame = tk.Frame(modify_window)

    # Label and entry for Record ID (for deleting and modifying records)
    label_record_id = tk.Label(id_frame, text="Record ID:")
    label_record_id.grid(row=0, column=0)
    entry_record_id = tk.Entry(id_frame)
    entry_record_id.grid(row=0, column=1)

    # Button to execute selected action
    execute_button = tk.Button(modify_window, text="Execute Action", command=lambda: execute_action())
    execute_button.grid(row=1, column=0, columnspan=2)

    # Function to execute selected action
    def execute_action():
        action = selected_action.get()
        if action == "Add Record":
            add_record()
        elif action == "Delete Record":
            delete_record()
        elif action == "Modify Record":
            modify_record()

    modify_window.mainloop()


def display_html_content(page_frame, content):
    # Remove style tags from HTML content
    soup = BeautifulSoup(content, 'html.parser')
    [s.extract() for s in soup('style')]
    content_without_style = str(soup)

    # Display HTML content without style tags
    html_label = HTMLLabel(page_frame, html=content_without_style)
    html_label.pack(fill="both", expand=True)


def main_page():
    main_window = tk.Tk()
    main_window.title("Car Accident Analysis")
    main_window.geometry("1000x800")  # Adjusted geometry
    main_window.iconbitmap("icon.ico")

    # Fetch and prepare data
    df = fetch_data()
    df = prepare_data(df)

    # Create a notebook for navigation
    notebook = ttk.Notebook(main_window)
    notebook.pack(fill="both", expand=True)

    # Create frames for each page
    overview_frame = tk.Frame(notebook)
    analysis_frame = tk.Frame(notebook)
    graph_frame = tk.Frame(notebook)
    data_frame = tk.Frame(notebook)
    modify_frame = tk.Frame(notebook)

    # Add frames to the notebook
    notebook.add(overview_frame, text="Main Page")
    notebook.add(analysis_frame, text="Analysis")
    notebook.add(graph_frame, text="Graph")
    notebook.add(data_frame, text="Data")
    notebook.add(modify_frame, text="Modify Data")

    # Display project overview content in the Overview Page
    project_overview_content = """
    Car Accident Analysis Project Overview

    Welcome to the Car Accident Analysis project overview page. In this project, we aim to analyze car accident data to 
    gain insights into the factors influencing accident severity. By leveraging machine learning techniques and 
    data visualization, we can better understand the patterns and trends in accident data, leading to 
    improved road safety measures.

    We have collected a comprehensive dataset comprising various attributes such as accident dates, locations, 
    weather conditions, road type, and severity levels. This dataset serves as the foundation for our analysis and 
    model development.
    

    Project Objectives

    The primary objectives of this project are:

    - To analyze the factors contributing to car accidents
    - To develop predictive models for accident severity
    - To provide actionable insights for stakeholders and policymakers

    Methodology

    We adopt a multi-step approach to achieve our objectives:

    - Data Collection: Gathering relevant car accident data from reliable sources
    - Data Preprocessing: Cleaning, transforming, and preparing the data for analysis
    - Exploratory Data Analysis (EDA): Exploring the dataset to understand distributions, correlations, and patterns
    - Feature Engineering: Selecting and engineering relevant features for model development
    - Model Development: Building machine learning models to predict accident severity
    - Evaluation and Interpretation: Assessing model performance and interpreting results
    - Visualization: Creating visualizations to communicate findings effectively

    Project Deliverables

    Upon completion of the project, we aim to deliver the following:

    - A comprehensive analysis report detailing the insights gained from the data
    - Predictive models for accident severity classification
    - Data visualizations to aid in understanding and decision-making

    We hope to provide valuable insights that contribute to enhancing road safety and reducing the frequency and 
    severity of car accidents.
    """

    overview_text = tk.Text(overview_frame, wrap="word", height=30, width=80, font=("Arial", 14))
    overview_text.insert(tk.END, project_overview_content)
    overview_text.config(state=tk.DISABLED)  # Make text widget read-only
    overview_text.pack(fill="both", expand=True)

    # Analysis Page
    perform_analysis(df)  # Call analysis function before embedding in frame
    analysis_fig = plt.gcf()
    analysis_fig_canvas = FigureCanvasTkAgg(analysis_fig, master=analysis_frame)
    analysis_fig_canvas.draw()
    analysis_fig_canvas.get_tk_widget().pack(fill="both", expand=True)

    # Graph Page
    graph_page = tk.Frame(graph_frame)
    graph_page.pack(fill='both', expand=True)
    load_graph_page(graph_page)

    # Data Page
    data_content = """
    <!DOCTYPE html>
    <html>
    <head>
        <style>
            body {
                font-family: Arial, sans-serif;
                padding: 20px;
                line-height: 1.6;
            }
            h1 {
                color: #333;
                font-size: 24px;
            }
            p {
                color: #666;
                font-size: 16px;
            }
        </style>
    </head>
    <body>
        <h1>Data Page</h1>
        <p>The Data page serves as a repository of raw accident data, offering stakeholders access to valuable insights 
        for research and analysis. Displaying a subset of the dataset, 
        users can peruse accident records and gain an understanding of the available information. 
        Additionally, users have the convenience of downloading the dataset in CSV format, 
        facilitating further analysis or integration with other tools and platforms.</p>
    </body>
    </html>
    """

    # Display HTML content on the Data page
    display_html_content(data_frame, data_content)

    show_data_button = tk.Button(data_frame, text="Show Data", command=lambda: show_data(df))
    show_data_button.pack()

    # Modify Data Page
    modify_data_content = """
    <!DOCTYPE html>
    <html>
    <head>
        <style>
            body {
                font-family: Arial, sans-serif;
                padding: 20px;
                line-height: 1.6;
            }
            h1 {
                color: #333;
                font-size: 24px;
            }
            p {
                color: #666;
                font-size: 16px;
            }
        </style>
    </head>
    <body>
        <h1>Modify Data Page</h1>
        <p>The Modify Data page facilitates seamless management of accident records, providing functionalities to add, 
        delete, or modify entries as needed. Through an intuitive interface, users can input new accident records, 
        specifying details such as date, location, severity, and weather conditions. Furthermore, 
        users have the ability to delete or update existing records, ensuring data accuracy and relevance. 
        By offering these capabilities, the Modify Data page empowers stakeholders to maintain an up-to-date and 
        comprehensive database of accident information, facilitating effective analysis and decision-making.</p>
    </body>
    </html>
    """

    # Display HTML content on the Modify Data page
    display_html_content(modify_frame, modify_data_content)

    modify_button = tk.Button(modify_frame, text="Modify Data", command=modify_data)
    modify_button.pack()

    main_window.mainloop()

if __name__ == "__main__":
    main_page()