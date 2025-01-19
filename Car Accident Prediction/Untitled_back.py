import gc
import warnings
import numpy as np
import pandas as pd
from sqlalchemy import create_engine
from sklearn.model_selection import train_test_split
from sklearn.utils.class_weight import compute_class_weight
from sklearn.metrics import roc_auc_score, confusion_matrix
import matplotlib.pyplot as plt
import itertools
from multiprocessing import Pool
from catboost import CatBoostClassifier, Pool
from tqdm import tqdm
import seaborn as sns

# Suppress warnings
warnings.filterwarnings("ignore")

# Set maximum display rows for pandas
pd.set_option('display.max_rows', 1000)

# Database connection string
db_connection_str = 'mysql://root:Helda2004@localhost/car_accident'

# Create database engine
conn = create_engine(db_connection_str)

# SQL query
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

# Read data from SQL
result_proxy = conn.execute(sql_query)
df = pd.DataFrame(result_proxy.fetchall(), columns=result_proxy.keys())
# Remove duplicates
item0 = df.shape[0]
df = df.drop_duplicates()
item1 = df.shape[0]
print(f"Number of duplicates: {item0 - item1}")

# Define target variable
df['is_serious_fatal'] = (df['Accident_Severity'] != 'Slight').astype(int)

# Extract Month and Year from Accident_Date without converting to datetime
df['Month'] = df['Accident_Date'].str.split('/').str[0]  # Extract month
df['Year'] = df['Accident_Date'].str.split('/').str[2]   # Extract year


# Select columns of interest
selected_cols = ['is_serious_fatal', 'Day_of_Week', 'Month', 'Year',
                 'Junction_Control', 'Junction_Detail', 'Light_Conditions',
                 'Local_Authority_(District)', 'Carriageway_Hazards',
                 'Number_of_Vehicles', 'Police_Force',
                 'Road_Surface_Conditions', 'Road_Type', 'Speed_limit', 'Time',
                 'Urban_or_Rural_Area', 'Weather_Conditions', 'Vehicle_Type']
df = df[selected_cols]

# Perform data transformation
df.info()

# Prepare categorical variables for modeling
for col in tqdm(df.columns):
    if col != 'is_serious_fatal':
        df[col] = df[col].fillna('None').astype(str)
        frequency_map = df[col].value_counts(normalize=True)
        df[col + '_Frequency'] = df[col].map(frequency_map)

# Machine learning
# Initialize data
y = df['is_serious_fatal'].values.reshape(-1,)
X = df.drop(['is_serious_fatal'], axis=1)

# Split data into train and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.5, random_state=0, stratify=y)

# Compute class weights
weights = compute_class_weight(class_weight='balanced', classes=np.unique(y_train), y=y_train)
class_weights = dict(zip(np.unique(y_train), weights))

# Identify categorical columns
cat_cols = X.select_dtypes(include=['object']).columns
cat_cols_idx = [X.columns.get_loc(col) for col in cat_cols]

# Train CatBoost model
model = CatBoostClassifier(cat_features=cat_cols_idx)
model.fit(X_train, y_train)

# Evaluate the model
train_score = model.score(X_train, y_train)
test_score = model.score(X_test, y_test)
print("Train Score:", train_score)
print("Test Score:", test_score)

# Specify the training parameters
train_pool = Pool(X_train, y_train, cat_features=cat_cols_idx)
test_pool = Pool(X_test, y_test, cat_features=cat_cols_idx)
model = CatBoostClassifier(iterations=100,
                           depth=5,
                           eval_metric='AUC',
                           border_count=22,
                           l2_leaf_reg=0.3,
                           learning_rate=1e-2,
                           class_weights=class_weights,
                           early_stopping_rounds=20,
                           verbose=0)

# Train the model
model.fit(train_pool, eval_set=test_pool)

# Make predictions using the resulting model
y_train_pred = model.predict_proba(X_train)[:,1]
y_test_pred = model.predict_proba(X_test)[:,1]

# Get the confusion matrix
cm = confusion_matrix(y_test, (y_test_pred > 0.5))
roc_auc_train = roc_auc_score(y_train, y_train_pred, sample_weight=[class_weights[label] for label in y_train])
roc_auc_test = roc_auc_score(y_test, y_test_pred, sample_weight=[class_weights[label] for label in y_test])
print(f"ROC AUC score for train: {round(roc_auc_train, 4)}, and for test: {round(roc_auc_test, 4)}")

# Calculate baseline ROC AUC
roc_auc_baseline = roc_auc_score(y_test, [np.mean(y_train)]*len(y_test), sample_weight=[class_weights[label] for label in y_test])
print("Baseline ROC AUC:", roc_auc_baseline)

# Plot the confusion matrix
def plot_confusion_matrix(cm, classes, title='Confusion Matrix', cmap=plt.cm.Blues):
    """
    This function plots a confusion matrix.

    Parameters:
        cm (array-like): Confusion matrix as returned by sklearn.metrics.confusion_matrix.
        classes (list): List of class names, e.g., ['Class 0', 'Class 1'].
        title (str): Title for the plot.
        cmap (matplotlib colormap): Colormap for the plot.
    """
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
        plt.text(j, i, format(cm[i, j], fmt), horizontalalignment="center", color="white" if cm[i, j] > thresh else "black")

    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    plt.tight_layout()
    plt.grid()
    plt.show()

# Plot the confusion matrix
class_names = ['Slight', 'Severe/Fatal']  # Positive class should come last
plot_confusion_matrix(cm, class_names)

#Analyze Dataset
plt.figure(figsize=(10, 6))
sns.countplot(x='Day_of_Week', data=df)
plt.title('Count of Accidents by Day of the Week')
plt.xlabel('Day of the Week')
plt.ylabel('Count')
plt.show()

plt.figure(figsize=(10, 6))
sns.countplot(x='Junction_Control', data=df)
plt.title('Count of Accidents by Junction Control')
plt.xlabel('Junction Control')
plt.ylabel('Count')
plt.xticks(rotation=90)
plt.show()