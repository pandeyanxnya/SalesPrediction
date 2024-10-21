import streamlit as st
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import matplotlib.pyplot as plt

# Load the combined dataset
@st.cache_data
def load_data():
    df = pd.read_csv('Sales_data.csv')
    df['Model Type'].fillna('Unknown', inplace=True)
    df.fillna(0, inplace=True)
    return df

# Load data
df = load_data()

# Title of the web app
st.title("Vehicle Sales Prediction System")

# Dropdowns for selections
make = st.selectbox('Select Vehicle Make', df['Make'].unique())
filtered_data = df[df['Make'] == make]
category = st.selectbox('Select Domestic or Export', filtered_data['Type'].unique())
filtered_data = filtered_data[filtered_data['Type'] == category]
subcategory = st.selectbox('Select Subcategory', filtered_data['Model Type'].unique())
filtered_data = filtered_data[filtered_data['Model Type'] == subcategory]
model = st.selectbox('Select Model', filtered_data['Model'].unique())
model_data = filtered_data[filtered_data['Model'] == model]

# Display the filtered dataset for the selected model
st.write(f"You selected {make} -> {category} -> {subcategory} -> {model}")
st.write(model_data[['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec', 'Total']])

# Model selection (placed outside the prediction block)
prediction_model = st.selectbox('Select Prediction Model', ['Linear Regression', 'Random Forest', 'Decision Tree'])

if st.button('Predict Next Month Sales'):
    # Prepare data for prediction
    X = model_data[['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']]
    y = model_data['Total']
    
    # Use the last month data as input for prediction
    last_month_data = X.iloc[-1].values.reshape(1, -1)

    # Split the data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Initialize the selected model
    if prediction_model == 'Linear Regression':
        model = LinearRegression()
    elif prediction_model == 'Random Forest':
        model = RandomForestRegressor()
    elif prediction_model == 'Decision Tree':
        model = DecisionTreeRegressor()

    # Fit the model
    model.fit(X_train, y_train)

    # Predict sales for next month based on the last month data
    next_month_prediction = model.predict(last_month_data)

    # Evaluate the model on the test set
    y_pred = model.predict(X_test)
    mae = mean_absolute_error(y_test, y_pred)
    mse = mean_squared_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)

    st.write(f"Predicted sales for next month: {next_month_prediction[0]}")
    st.write(f"Model Evaluation Metrics: \n - MAE: {mae} \n - MSE: {mse} \n - R^2: {r2}")

    # Plotting sales trends
    sales_trends = model_data[['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']]
    months = sales_trends.columns.tolist()

    plt.figure(figsize=(10, 5))
    plt.plot(months, sales_trends.loc[model_data.index].values.T, marker='o', label='Sales Trends')
    plt.axvline(x=months[-1], color='r', linestyle='--', label='Prediction Point')  # Indicate where prediction is made
    plt.title(f'Sales Trends for {model}')
    plt.xlabel('Months')
    plt.ylabel('Sales')
    plt.xticks(rotation=45)
    plt.grid()
    plt.legend()
    
    st.pyplot(plt)
