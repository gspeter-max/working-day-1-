'''
Problem 1: Multi-dimensional Array Operations with NumPy (Hard)
Given a 3D NumPy array of shape (10, 10, 10), implement the following:

1.Normalize the array along the second axis (axis=1) so that all values in the array lie between 0 and 1.
2.Create a mask that extracts the elements greater than the median of the entire array.
3. Calculate the row-wise sum of the extracted elements from the mask, and then compute the mean and variance of these sums across all depth levels of the array.
''' 
import numpy as np
from sklearn.preprocessing import MinMaxScaler
# Create a random 3D NumPy array of shape (10, 10, 10)
np.random.seed(42)
array = np.random.randint(1, 100, (10, 10, 10))

# Normalize the array along axis 1
scaler = MinMaxScaler()
normalize_array = np.array([scaler.fit_transform(arrays) for arrays in array])

# Median value
median_value = np.median(array)

# Mask for values greater than the median
mask = array > median_value
mask_values = array[mask]

# Sum and mean of values greater than the median along axis 1
sum_values = np.sum(array * mask, axis=1)
mean_sum = np.mean(sum_values)

# Variance of the sum_values
variance = np.var(sum_values)

print(f"Normalized Array Shape: {normalize_array.shape}")
print(f"Sum of Masked Values: {sum_values}")
print(f"Mean of Sum Values: {mean_sum}")
print(f"Variance of Sum Values: {variance}")

'''
Problem 2: Complex Data Manipulation with Pandas (Hard)
You are given a DataFrame with the following columns:

'id': A unique identifier for each customer.
'purchase_date': Date when the customer made the purchase.
'amount': The amount the customer spent.
'product_category': The category of the product purchased.
Steps to perform:

Convert purchase_date to datetime format.
Group the data by 'id' and calculate the total spending per customer.
For each customer, determine the month and year of their first purchase.
Create a new column 'average_purchase_per_month', where you calculate the average amount spent per month for each customer.
For each product_category, calculate the most frequent product purchased by the highest-spending customer.
Calculate the overall total spending per product_category and sort it in descending order. Find the median of these totals.
''' 
import pandas as pd

# Assuming df is your DataFrame
df['purchase_date'] = pd.to_datetime(df['purchase_date'])

# Calculate total spending and first purchase date
df['total_spending'] = df.groupby("id")['amount'].transform('sum')
df['first_purchase_date'] = df.groupby('id')['purchase_date'].transform('min')

# Extract month and year
df['month'] = df['first_purchase_date'].dt.month
df['year'] = df['first_purchase_date'].dt.year

# Average spending by month
avg_amount_spending = df.groupby(['id', 'month'])['amount'].mean().reset_index()

# Most frequent product category
df['most_frequent'] = df.groupby('id')['product_category'].transform(
    lambda x: x.mode().iloc[0] if not x.mode().empty else None
)

# Highest spender and their most frequent category
highest_spending = df.loc[df['total_spending'].idxmax(), ['id', 'most_frequent']]
highest_spender_id = highest_spending['id']
most_frequent_category = highest_spending['most_frequent']

print(f"The highest spending customer: {highest_spender_id}")
print(f"The highest product category: {most_frequent_category}")

# Spending breakdown by category
category_spending = df.groupby('product_category')['amount'].sum()

# Extract spending for target categories
if most_frequent_category in category_spending:
    print(f"{most_frequent_category}: {category_spending[most_frequent_category]}")
else:
    print(f"No spending data for category: {most_frequent_category}")

'''
Problem 3: Optimizing Matrix Operations (Extremely Hard)
You are given a large matrix of shape (1000, 1000) filled with random integers between 0 and 10. Write a function that:

Scales each row of the matrix by the sum of that row (i.e., divide each element of the row by the sum of that row).
For each column, compute the cumulative sum across all rows.
Compute the element-wise product of the matrix and its transpose (i.e., A * A.T), 
but avoid explicit loops to perform the multiplication. Return the matrix of size (1000, 1000) and compute the Frobenius norm of the result.
'''

import numpy as np
import pandas as pd

# Set random seed and create a 2D array
np.random.seed(42)
array = np.random.randint(0, 11, (1000, 1000))

# Row-wise scaling (normalize rows)
row_sums = np.sum(array, axis=1, keepdims=True)
row_sums[row_sums == 0] = 1  # Avoid division by zero
scaled_array = array / row_sums

# Create a DataFrame for row-wise sums and cumulative sums
df = pd.DataFrame(array)
temp = pd.DataFrame()
temp['sums'] = df.sum(axis=1)
temp['cumulative_sum'] = temp['sums'].cumsum()

# Frobenius norm
frobenius_norm = np.linalg.norm(array, 'fro')

# Matrix multiplication (dot product of array with its transpose)
temp_1 = np.dot(array, array.T)

# Outputs
print(f"Scaled Array Shape: {scaled_array.shape}")
print(f"Cumulative Sums:\n{temp['cumulative_sum'].head()}")
print(f"Frobenius Norm: {frobenius_norm}")
print(f"Matrix Multiplication Result Shape: {temp_1.shape}")

