import gc
import warnings

import pandas as pd

warnings.filterwarnings("ignore")
from sqlalchemy import create_engine

# Set Pandas options to display a maximum of 1000 rows
pd.set_option('display.max_rows', 1000)

# MySQL database connection string
db_connection_str = 'mysql://root:Helda2004@localhost/car_accident'

# Create a database connection
conn = create_engine(db_connection_str)

# SQL query to fetch data from the database
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

# Read data from the MySQL database into a pandas DataFrame
df = pd.read_sql(sql_query, conn)

# Drop duplicates
item0 = df.shape[0]
df = df.drop_duplicates()
item1 = df.shape[0]
print(f"Number of duplicates: {item0 - item1}")

# Create target variable
df['is_serious_fatal'] = (df['Accident_Severity'] != 'Slight').astype(int)

# Convert 'Accident Date' to datetime
df['Accident_Date'] = pd.to_datetime(df['Accident_Date'])
df['Month'] = df['Accident_Date'].dt.month_name()
df['Year'] = df['Accident_Date'].dt.year

# Select relevant columns
selected_cols = ['is_serious_fatal', 'Day_of_Week', 'Month', 'Year',
                 'Junction_Control', 'Junction_Detail', 'Light_Conditions',
                 'Local_Authority_(District)', 'Carriageway_Hazards',
                 'Number_of_Vehicles', 'Police_Force',
                 'Road_Surface_Conditions', 'Road_Type', 'Speed_limit', 'Time',
                 'Urban_or_Rural_Area', 'Weather_Conditions', 'Vehicle_Type']
df = df[selected_cols]

# Perform garbage collection
gc.collect()

print(df.shape)
var = df.sample(5).T