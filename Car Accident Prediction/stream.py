import hashlib

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib import cm
from sqlalchemy import create_engine
from tqdm import tqdm
import itertools
from sklearn.metrics import confusion_matrix
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split

st.set_option('deprecation.showPyplotGlobalUse', False)

# Database connection details (replace with your actual credentials)
db_connection_str = 'mysql://root:Helda2004@localhost/car_accident'
def create_connection():
    """Creates a connection to the MySQL database."""
    return create_engine(db_connection_str)

if st.query_params.get("login_redirect"):
    # Redirect to stream.py
    st.experimental_rerun()

if 'logged_in' in st.session_state and st.session_state.logged_in:
    st.title("Welcome to the Main Page")
    st.write("This is the main page of your application.")
def fetch_data():
    """Fetches data from the 'accidents' table in the database."""
    conn = create_connection()
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
    st.header("Analysis page")
    st.write("Performing analysis...")  # Debug statement

    # Assuming you have trained a RandomForestClassifier model
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
    plot_confusion_matrix(cm, class_names)

    st.write("Analysis completed.")  # Debug statement

def plot_confusion_matrix(cm, classes, title='Confusion Matrix', cmap=plt.cm.Blues):
    """
    This function plots a confusion matrix.

    Parameters:
        cm (array-like): Confusion matrix as returned by sklearn.metrics.confusion_matrix.
        classes (list): List of class names, e.g., ['Class 0', 'Class 1'].
        title (str): Title for the plot.
        cmap (matplotlib colormap): Colormap for the plot.
    """
    # Ensure cm is an ndarray
    cm = np.array(cm)

    # Display the confusion matrix
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
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
    st.pyplot()  # Display the plot using Streamlit's pyplot wrapper

def graph(df):
    """Analyzes the prepared data and generates insights."""
    fig, ax = plt.subplots(figsize=(10, 6))
    sns.countplot(x='Day_of_Week', data=df, ax=ax)
    plt.title('Count of Accidents by Day of the Week')
    plt.xlabel('Day of the Week')
    plt.ylabel('Count')
    plt.xticks(rotation=45)
    st.pyplot(fig)

    fig, ax = plt.subplots(figsize=(10, 6))
    sns.countplot(x='Junction_Control', data=df, ax=ax)
    plt.title('Count of Accidents by Junction Control')
    plt.xlabel('Junction Control')
    plt.ylabel('Count')
    plt.xticks(rotation=90)
    st.pyplot(fig)

    # Additional analysis based on your requirements (e.g., correlations, heatmaps)
    # ...

def show_data(df):
    """Displays a subset of the data with download functionality."""
    st.dataframe(df.head(500))
    csv = df.to_csv(index=False)
    st.download_button(label="Download Data (200 rows)", data=csv, file_name="accidents.csv")
def add_record(conn, values):
    """Add a new record to the accidents table."""
    query = """
        INSERT INTO accidents (Accident_Date, Day_of_Week, Junction_Control, Junction_Detail,
            Accident_Severity, Light_Conditions, Local_Authority_(District),
            Carriageway_Hazards, Number_of_Vehicles, Police_Force,
            Road_Surface_Conditions, Road_Type, Speed_limit, Time,
            Urban_or_Rural_Area, Weather_Conditions, Vehicle_Type)
        VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
    """
    conn.execute(query, values)
def add_record(conn, values):
    """Add a new record to the accidents table."""
    query = """
        INSERT INTO accidents (Accident_Date, Day_of_Week, Junction_Control, Junction_Detail,
            Accident_Severity, Light_Conditions, Local_Authority_(District),
            Carriageway_Hazards, Number_of_Vehicles, Police_Force,
            Road_Surface_Conditions, Road_Type, Speed_limit, Time,
            Urban_or_Rural_Area, Weather_Conditions, Vehicle_Type)
        VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
    """

    # Check if all required fields are filled
    all_fields_filled = all(values)
    if not all_fields_filled:
        st.error("Please fill in all required fields.")
    else:
        # Add record to the database
        conn.execute(query, values)
        st.success("Record added successfully!")

def delete_record(conn, id):
    """Delete a record from the accidents table."""
    query = """
        DELETE FROM accidents WHERE id = %s
    """
    conn.execute(query, (id,))
def modify_record(conn, id, values):
    """Modify a record in the accidents table."""
    query = """
        UPDATE accidents SET Accident_Date = %s, Day_of_Week = %s, Junction_Control = %s,
            Junction_Detail = %s, Accident_Severity = %s, Light_Conditions = %s,
            Local_Authority_(District) = %s, Carriageway_Hazards = %s, Number_of_Vehicles = %s,
            Police_Force = %s, Road_Surface_Conditions = %s, Road_Type = %s,
            Speed_limit = %s, Time = %s, Urban_or_Rural_Area = %s, Weather_Conditions = %s,
            Vehicle_Type = %s
        WHERE id = %s
    """
    conn.execute(query, values + (id,))
def modify_data():
    """Allows modification of the data."""
    st.header("Modify Data")
    selected_action = st.radio("Select an Action", ["Add Record", "Delete Record", "Modify Record"])

    # Connect to the database
    conn = create_connection()

    if selected_action == "Add Record":
        st.subheader("Add Record")
        # Add forms or inputs for adding a new record
        accident_date = st.date_input("Accident Date")
        day_of_week = st.selectbox("Day of Week", ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday'])
        junction_control = st.multiselect("Junction Control", ['Data missing', 'Out of Range', 'Give way', 'Uncontrolled'] )
        junction_detail = st.multiselect("Junction Detail", ['T Junction', 'Not at Junction', 'Staggered Junction'])
        accident_severity = st.selectbox("Accident Severity", ['Serious', 'Slight'])
        light_conditions = st.selectbox("Light Conditions", ['Daylight', 'Darkness - lights lit', 'Darkness - no lighting', 'Darkness - lighting unknown'])
        local_authority_district = st.selectbox("Local Authority (District)", ['Eden', 'Peterborough', 'South Cambridgeshire', 'Copeland', 'South Lakeland', 'Huntingdonshire', 'Barrow-in-Furness', 'Staffordshire Moorlands', 'Cannock Chase', 'Allerdale'])
        carriageway_hazards = st.selectbox("Carriageway Hazards", ['None', 'Previous accident', 'Other object on road'])
        number_of_vehicles = st.number_input("Number of Vehicles", min_value=0, step=1)
        police_force = st.selectbox("Police Force", ['Cumbria', 'Staffordshire'])
        road_surface_conditions = st.selectbox("Road Surface Conditions", ['Frost or ice', 'Dry', 'Wet or damp'])
        road_type = st.selectbox("Road Type", ['Single carriageway', 'Roundabout', 'Dual carriageway'])
        speed_limit = st.number_input("Speed Limit", min_value=0, step=1)
        time = st.time_input("Time")
        urban_or_rural_area = st.selectbox("Urban or Rural Area", ['Urban', 'Rural Area'])
        weather_conditions = st.selectbox("Weather Conditions", ['Fine no high winds', 'Fine + high winds', 'Raining + high winds', 'Raining no high winds', 'Fog or mist', 'Other'])
        vehicle_type = st.multiselect("Vehicle Type", ['Car', 'Motorcycle', 'Goods Lorry', 'Bus', 'Van', 'other'])

        if st.button("Submit"):
            # Check if all required fields are filled
            if not (accident_date and day_of_week and junction_control and junction_detail and accident_severity and light_conditions and local_authority_district and carriageway_hazards and number_of_vehicles and police_force and road_surface_conditions and road_type and speed_limit and time and urban_or_rural_area and weather_conditions and vehicle_type):
                st.error("Please fill in all required fields.")
            else:
                # Add record to the database
                add_record(conn, (
                    accident_date, day_of_week, junction_control, junction_detail, accident_severity, light_conditions,
                    local_authority_district, carriageway_hazards, number_of_vehicles, police_force,
                    road_surface_conditions, road_type, speed_limit, time, urban_or_rural_area, weather_conditions,
                    vehicle_type))
                st.success("Record added successfully!")

    elif selected_action == "Delete Record":
        st.subheader("Delete Record")
        # Fetch records from the database to display for selection
        records = fetch_data()  # Assuming fetch_data fetches all records
        record_to_delete = st.selectbox("Select a record to delete", records)
        if st.button("Delete Record"):
            delete_record(conn, record_to_delete)  # Assuming record_to_delete contains the ID of the record
            st.success("Record deleted successfully!")

    elif selected_action == "Modify Record":
        st.subheader("Modify Record")
        # Fetch records from the database to display for selection
        records = fetch_data()  # Assuming fetch_data fetches all records
        record_to_modify = st.selectbox("Select a record to modify", records)

        # Add input fields to modify each column of the selected record
        modified_accident_date = st.date_input("Modified Accident Date", record_to_modify['Accident_Date'])
        modified_day_of_week = st.selectbox("Modified Day of Week", ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday'], index=record_to_modify['Day_of_Week'])
        # Add more input fields for other columns

        if st.button("Modify Record"):
            # Execute the modify_record function with the modified values
            modify_record(conn, record_to_modify.id, (modified_accident_date, modified_day_of_week))  # Pass values for other columns as well
            st.success("Record modified successfully!")


def main_page():
    st.set_page_config(page_title="Car Accident Analysis", page_icon="🚗")

    # Fetch and prepare data
    df = fetch_data()
    df = prepare_data(df)

    # Create a sidebar for navigation
    selected_page = st.sidebar.radio("Select a Page", ["Analysis", "Graph", "Data", "Modify Data"])

    if selected_page == "Analysis":
        perform_analysis(df)
    elif selected_page == "Graph":
        graph(df)
    elif selected_page == "Data":
        show_data(df)
    elif selected_page == "Modify Data":
        modify_data()

def main():
    main_page()



if __name__ == "__main__":
    main()