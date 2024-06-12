import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Load the datasets
travel_details_df = pd.read_csv('Travel details dataset.csv')
flights_df = pd.read_csv('Flights.csv')

# Display the first few rows of each dataframe to understand their structure and contents
travel_details_df_head = travel_details_df.head()
flights_df_head = flights_df.head()

print('Travel details dataset')
print(travel_details_df_head)

print('Flight dataset')
print (flights_df_head)

# Setting the aesthetic style of the plots
sns.set(style="whitegrid")

# Create a figure with 3 plots in one row
fig, ax = plt.subplots(1, 3, figsize=(18, 6))

# Plot 1: Age Distribution
sns.histplot(travel_details_df['Traveler age'], bins=10, kde=True, ax=ax[0], color='blue')
ax[0].set_title('Age Distribution of Travelers')
ax[0].set_xlabel('Age')
ax[0].set_ylabel('Frequency')

# Plot 2: Gender Distribution
gender_counts = travel_details_df['Traveler gender'].value_counts()
sns.barplot(x=gender_counts.index, y=gender_counts.values, ax=ax[1], palette='pastel')
ax[1].set_title('Gender Distribution of Travelers')
ax[1].set_xlabel('Gender')
ax[1].set_ylabel('Count')

# Plot 3: Nationality Distribution
nationality_counts = travel_details_df['Traveler nationality'].value_counts().head(10)  # Top 10 nationalities
sns.barplot(y=nationality_counts.index, x=nationality_counts.values, ax=ax[2], palette='Set3')
ax[2].set_title('Top 10 Nationalities of Travelers')
ax[2].set_xlabel('Count')
ax[2].set_ylabel('Nationality')

plt.tight_layout()
plt.show()

# Clean the 'Accommodation cost' by removing non-numeric characters except decimal points
travel_details_df['Accommodation cost'] = travel_details_df['Accommodation cost'].str.replace('[^\d.]', '', regex=True).astype(float)

# Setting the aesthetic style of the plots
sns.set(style="whitegrid")

# Create a single row of plots with 3 columns
fig, ax = plt.subplots(1, 3, figsize=(18, 6))

# Plot 1: Top 10 Travel Destinations
top_destinations = travel_details_df['Destination'].value_counts().nlargest(10)
sns.barplot(y=top_destinations.index, x=top_destinations.values, palette='viridis', ax=ax[0])
ax[0].set_title('Top 10 Travel Destinations')
ax[0].set_xlabel('Number of Trips')
ax[0].set_ylabel('Destination')

# Plot 2: Accommodation Types
accommodation_counts = travel_details_df['Accommodation type'].value_counts()
sns.barplot(y=accommodation_counts.index, x=accommodation_counts.values, ax=ax[1], palette='coolwarm')
ax[1].set_title('Accommodation Types')
ax[1].set_xlabel('Number of Trips')
ax[1].set_ylabel('Type of Accommodation')

# Plot 3: Simplified Accommodation Costs Visualization using a histogram
sns.histplot(travel_details_df['Accommodation cost'], bins=20, color='lightblue', ax=ax[2])
ax[2].set_title('Distribution of Accommodation Costs')
ax[2].set_xlabel('Cost in USD')
ax[2].set_ylabel('Frequency')

plt.tight_layout()
plt.show()


# Clean the 'Price' column in the flights dataset to remove the euro symbol and any spaces, then convert to float
flights_df['Price'] = flights_df['Price'].str.replace('€', '').str.strip().astype(float)

# Calculate average and median flight prices
average_price = flights_df['Price'].mean()
median_price = flights_df['Price'].median()

print('Average flight prices')
print(average_price)

print('Median flghtprices')
print(median_price)


# Create figures for busiest days and months
fig, ax = plt.subplots(1, 2, figsize=(14, 6))

# Plot 1: Busiest Days of the Month
day_of_month_counts = flights_df['DayOfMonth'].value_counts().sort_index()
sns.barplot(x=day_of_month_counts.index, y=day_of_month_counts.values, ax=ax[0], palette='coolwarm')
ax[0].set_title('Flight Counts by Day of the Month')
ax[0].set_xlabel('Day of the Month')
ax[0].set_ylabel('Number of Flights')

# Plot 2: Busiest Months
month_counts = flights_df['Month'].value_counts().reindex([
    "January", "February", "March", "April", "May", "June", "July", "August", "September", "October", "November", "December"
], fill_value=0)
sns.barplot(x=month_counts.index, y=month_counts.values, ax=ax[1], palette='viridis')
ax[1].set_title('Flight Counts by Month')
ax[1].set_xlabel('Month')
ax[1].set_ylabel('Number of Flights')
ax[1].tick_params(axis='x', rotation=45)  # Rotate month labels for better readability

plt.tight_layout()
plt.show()


# Calculate the average flight price for each day of the month
average_price_per_day = flights_df.groupby('DayOfMonth')['Price'].mean().sort_index()

# Plotting the average flight price fluctuations over the days of the month
plt.figure(figsize=(12, 6))
sns.lineplot(x=average_price_per_day.index, y=average_price_per_day.values, marker='o', color='b')
plt.title('Average Flight Price Fluctuations by Day of the Month')
plt.xlabel('Day of the Month')
plt.ylabel('Average Price (€)')
plt.grid(True)
plt.show()

# Check for missing values in the flights dataset
missing_values = flights_df.isnull().sum()

# Check data types and need for transformations
data_types = flights_df.dtypes

print('Missing values and data types')
print(missing_values, data_types)


from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder

# Initialize the OneHotEncoder
encoder = OneHotEncoder()

# One-hot encode the categorical variables
encoded_features = encoder.fit_transform(flights_df[['Origin', 'Destination', 'Month']]).toarray()

# Create a DataFrame from the encoded features
encoded_df = pd.DataFrame(encoded_features, columns=encoder.get_feature_names_out(['Origin', 'Destination', 'Month']))

# Combine the encoded features with the numerical features in the original dataframe
prepared_df = pd.concat([flights_df.drop(['Origin', 'Destination', 'Month'], axis=1), encoded_df], axis=1)

# Split the data into training and testing sets
X = prepared_df.drop('Price', axis=1)
y = prepared_df['Price']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)



# Proceed with X_train and y_train for model training
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score

# Initialize the Linear Regression model
model = LinearRegression()

# Fit the model on the training data
model.fit(X_train, y_train)

# Predict the prices on the test set
y_pred = model.predict(X_test)

# Calculate and print the model performance metrics
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print('Mean Squared Error')
print(mse)

print('R Squared')
print(r2)

# Ensure 'Accommodation cost' and 'Transportation cost' are strings before replacing non-numeric characters
travel_details_df['Accommodation cost'] = pd.to_numeric(travel_details_df['Accommodation cost'].astype(str).str.replace('[^\d.]', '', regex=True))
travel_details_df['Transportation cost'] = pd.to_numeric(travel_details_df['Transportation cost'].astype(str).str.replace('[^\d.]', '', regex=True))

# Re-normalize destinations and check for any null values after conversion
travel_details_df['Normalized Destination'] = travel_details_df['Destination'].apply(lambda x: x.split(',')[0].strip() if isinstance(x, str) else '')
flights_df['Normalized Destination'] = flights_df['Destination'].apply(lambda x: x.split(',')[0].strip() if isinstance(x, str) else '')

# Recompute average costs by destination in the travel details dataset
travel_costs = travel_details_df.groupby('Normalized Destination').agg({
    'Accommodation cost': 'mean',
    'Transportation cost': 'mean'
}).reset_index()

# Recompute average flight price by destination in the flights dataset
flight_costs = flights_df.groupby('Normalized Destination')['Price'].mean().reset_index()

# Re-merge the datasets on normalized destination
merged_costs = pd.merge(travel_costs, flight_costs, on='Normalized Destination', how='inner')

# Display the merged dataset for review
print('normalized destination')
print(merged_costs.head())


# Calculate the correlation matrix
correlation_matrix = merged_costs[['Accommodation cost', 'Transportation cost', 'Price']].corr()

# Plot the correlation matrix as a heatmap
plt.figure(figsize=(8, 6))
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', fmt=".2f", linewidths=.5)
plt.title('Correlation Matrix')
plt.show()


from sklearn.linear_model import LinearRegression

# Prepare the data for regression analysis
X = merged_costs[['Accommodation cost', 'Transportation cost']]
y = merged_costs['Price']

# Initialize and train the linear regression model
regression_model = LinearRegression()
regression_model.fit(X, y)

# Coefficients of the model
coefficients = regression_model.coef_
intercept = regression_model.intercept_

# Print the results
print('Coefficients')
print(coefficients)

print('Intercept')
print(intercept)


from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA

# Ensure all steps are performed correctly
# Select relevant features for clustering
features = travel_details_df[['Traveler age', 'Duration (days)', 'Accommodation type', 'Transportation type']]

# Encode categorical variables
encoded_features = pd.get_dummies(features, columns=['Accommodation type', 'Transportation type'])

# Convert to DataFrame if it's a NumPy array
encoded_features_df = pd.DataFrame(encoded_features)

# Impute missing values with the median of each column
encoded_features_df = encoded_features_df.fillna(encoded_features_df.median())

# Normalize the data
scaler = StandardScaler()
normalized_features = scaler.fit_transform(encoded_features_df)

# Determine the optimal number of clusters using the elbow method
sse = []
for k in range(1, 11):
    kmeans = KMeans(n_clusters=k, random_state=42)
    kmeans.fit(normalized_features)
    sse.append(kmeans.inertia_)

# Plot the elbow curve
plt.figure(figsize=(8, 6))
plt.plot(range(1, 11), sse, marker='o')
plt.xlabel('Number of Clusters')
plt.ylabel('Sum of Squared Errors (SSE)')
plt.title('Elbow Method for Determining Optimal Number of Clusters')
plt.show()

# Based on the elbow curve, choose the optimal number of clusters (k)
optimal_k = 4  # For example, based on visual inspection of the elbow curve

# Apply K-means clustering
kmeans = KMeans(n_clusters=optimal_k, random_state=42)
clusters = kmeans.fit_predict(normalized_features)

# Add the cluster labels to the original dataframe
travel_details_df['Cluster'] = clusters

# Analyze the characteristics of each cluster, only selecting numeric columns
numeric_columns = ['Traveler age', 'Duration (days)', 'Accommodation cost', 'Transportation cost']
cluster_analysis = travel_details_df.groupby('Cluster')[numeric_columns].mean().reset_index()

# Display the cluster analysis
print(cluster_analysis)


import dash
from dash import dcc, html
from dash.dependencies import Input, Output
import plotly.express as px


