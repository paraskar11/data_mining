# !pip install apyori
import pandas as pd
from apyori import apriori
import matplotlib.pyplot as plt

# Load the dataset
data = pd.read_csv('/content/apriori_dataset.csv')

# Create a list of transactions
transactions = []
for index, row in data.iterrows():
    transaction = []
    for column in data.columns[1:]:  # Start from the second column (item names)
        if row[column]:  # If the item was purchased (True)
            transaction.append(column)  # Add it to the current transaction
    transactions.append(transaction)

# Apply Apriori algorithm
association_rules = apriori(transactions, min_support=0.0045, min_confidence=0.2, min_lift=3, min_length=2)
association_results = list(association_rules)

# Store results in a DataFrame for better analysis
# Initialize an empty list to store the data for the DataFrame
results_data = []

# Iterate through association results and append data to the list
for item in association_results:
    pair = item[0]
    items = [x for x in pair]
    results_data.append({'Rule': f"{items[0]} -> {items[1]}",
                        'Support': item[1],
                        'Confidence': item[2][0][2],
                        'Lift': item[2][0][3]})

# Create the DataFrame from the list of data
results_df = pd.DataFrame(results_data)

# Print the DataFrame
print(results_df)

# # Data Visualization
# # 1. Support vs Confidence
# plt.figure(figsize=(10, 6))
# plt.scatter(results_df['Support'], results_df['Confidence'], alpha=0.5)
# plt.xlabel('Support')
# plt.ylabel('Confidence')
# plt.title('Support vs Confidence')
# plt.grid(True)
# plt.show()


# # 2.  Lift Distribution
# plt.figure(figsize=(8, 5))
# plt.hist(results_df['Lift'], bins=10, edgecolor='black')  # Adjust number of bins as needed
# plt.xlabel('Lift')
# plt.ylabel('Frequency')
# plt.title('Distribution of Lift Values')
# plt.grid(axis='y', alpha=0.75)
# plt.show()


# # 3. Top rules by lift (bar chart)
# top_rules = results_df.nlargest(10, 'Lift') # Get top 10 rules by lift
# plt.figure(figsize=(10, 6))
# plt.bar(top_rules['Rule'], top_rules['Lift'], color='skyblue')
# plt.xlabel('Rule')
# plt.ylabel('Lift')
# plt.title('Top 10 Rules by Lift')
# plt.xticks(rotation=45, ha='right')
# plt.tight_layout()
# plt.show()