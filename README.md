# Insightful-Journeys-Analyzing-Travel-Trends-and-Flight-Dynamics
This project performs an extensive analysis of travel details and flight data. The aim is to uncover insights from these datasets and visualize the findings through various plots. The analysis covers traveler demographics, trip details, and flight information. The goal is to provide actionable insights for travel companies, airlines, and other stakeholders in the travel industry.

Data Description
We used two datasets for this analysis:

**Travel Details Dataset:** Contains information about travelers, including their age, gender, nationality, travel destinations, accommodation types, and costs.
**Flights Dataset:** Provides details about flights, including flight prices, origins, destinations, and dates.

**Data Preparation**
Loading the Data:

import pandas as pd

# Load datasets
travel_details_df = pd.read_csv('Travel details dataset.csv')
flights_df = pd.read_csv('Flights.csv')
Cleaning and Transforming Data:

Normalization: Normalize destination names to ensure consistency.
Cost Cleaning: Remove non-numeric characters from cost columns and convert them to numeric types.

**Normalized Destination Analysis:**

**Amsterdam:**
Average Accommodation Cost: $966.67
Average Transportation Cost: $350.00
Average Flight Price: $46.90

**Barcelona:**
Average Accommodation Cost: $1683.33
Average Transportation Cost: $716.67
Average Flight Price: $42.90

These insights highlight the average costs associated with each destination.

travel_details_df['Accommodation cost'] = travel_details_df['Accommodation cost'].str.replace('[^\d.]', '', regex=True).astype(float)
travel_details_df['Transportation cost'] = travel_details_df['Transportation cost'].str.replace('[^\d.]', '', regex=True).astype(float)
flights_df['Price'] = flights_df['Price'].str.replace('€', '').str.strip().astype(float)

**Combining Datasets:**
Merge travel and flight data based on normalized destinations.

combined_df = pd.merge(travel_details_df, flights_df, on='Normalized Destination', how='inner')
combined_df.to_csv('prepared_data_for_dashboard.csv', index=False)

Data Analysis and Visualization

**Traveler Demographics:**

**Age Distribution:** Shows the distribution of travelers' ages.
**Gender Distribution:** Displays the proportion of male and female travelers.
**Nationality Distribution:** Lists the top 10 nationalities among travelers.


**Travel Details:**

**Top 10 Travel Destinations: Highlights the most popular travel destinations.**
Accommodation Types: Shows the distribution of different types of accommodations used by travelers.
Accommodation Costs: Visualizes the distribution of accommodation costs.


**Flight Analysis:**

Flight Counts by Day of the Month: Displays the number of flights per day.
Flight Counts by Month: Highlights the busiest months for flights.
Average Flight Price Fluctuations: Shows the fluctuation of average flight prices by day of the month.


**Correlation Analysis:**

Correlation Matrix: Displays the correlation between accommodation costs, transportation costs, and flight prices.

**Cluster Analysis:**

Elbow Method: Determines the optimal number of clusters for segmentation.

Machine Learning Model
We used a linear regression model to predict flight prices based on accommodation and transportation costs.


from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score

# Prepare the data for regression analysis
X = merged_costs[['Accommodation cost', 'Transportation cost']]
y = merged_costs['Price']

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Initialize the Linear Regression model
model = LinearRegression()
model.fit(X_train, y_train)

# Predict the prices on the test set
y_pred = model.predict(X_test)

# Calculate and print the model performance metrics
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print('Mean Squared Error:', mse)
print('R Squared:', r2)

**Results**
Mean Squared Error (MSE): 1692.73
R-Squared (R²): 0.76

The R-squared value indicates that the model explains about 76% of the variance in flight prices based on accommodation and transportation costs. 
The MSE gives an idea of the average squared difference between actual and predicted flight prices.

**Key Visualizations**

**Traveler Demographics:**
![Figure_1](https://github.com/mojutoju/Insightful-Journeys-Analyzing-Travel-Trends-and-Flight-Dynamics/assets/52916369/01d005ca-2e37-492d-bdd1-1af8b872153c)

**Travel Details:**
![Figure_2](https://github.com/mojutoju/Insightful-Journeys-Analyzing-Travel-Trends-and-Flight-Dynamics/assets/52916369/421a96a2-917f-41c6-bb5b-c0a3f1008eda)


**Flight Analysis:**
![Figure_3](https://github.com/mojutoju/Insightful-Journeys-Analyzing-Travel-Trends-and-Flight-Dynamics/assets/52916369/93ef06de-7c26-4c3e-be97-42a446d3941d)
![Figure_4](https://github.com/mojutoju/Insightful-Journeys-Analyzing-Travel-Trends-and-Flight-Dynamics/assets/52916369/d37a6bc9-95c4-4497-b3ce-0c83427a8bf3)


**Correlation Analysis:**
![Figure_5](https://github.com/mojutoju/Insightful-Journeys-Analyzing-Travel-Trends-and-Flight-Dynamics/assets/52916369/030edff8-4a87-4955-8ba1-a8362554f2ed)


**Cluster Analysis:**
![Figure_6](https://github.com/mojutoju/Insightful-Journeys-Analyzing-Travel-Trends-and-Flight-Dynamics/assets/52916369/3128426e-7f8e-46c2-b839-f054c71e13cf)


**Cluster 0:**
Average Age: 33.38
Average Duration: 7.69 days
Average Accommodation Cost: $1369.23
Average Transportation Cost: $753.85

**Cluster 1:**
Average Age: 34.43
Average Duration: 7.11 days
Average Accommodation Cost: $1500.00
Average Transportation Cost: $687.74

**Cluster 2:**
Average Age: 32.26
Average Duration: 8.22 days
Average Accommodation Cost: $627.39
Average Transportation Cost: $590.87

**Cluster 3:**
Average Age: 32.13
Average Duration: 7.85 days
Average Accommodation Cost: $1220.21
Average Transportation Cost: $593.72

These clusters represent different segments of travelers based on their behaviors and preferences, which can be useful for targeted marketing and personalized travel packages.

**Files in the Repository**
Travel details dataset.csv: Raw travel details data.
Flights.csv: Raw flight data.
project.py: Python script for data analysis and visualization.
Figures: Generated figures used in the analysis.

**Conclusion**
This project provides valuable insights into traveler demographics, trip details, and flight prices. The visualizations and predictive model can help stakeholders in the travel industry make data-driven decisions.

