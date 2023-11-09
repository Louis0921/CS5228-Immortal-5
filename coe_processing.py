import pandas as pd

coe_data = pd.read_csv('auxiliary-data/sg-coe-prices.csv')

coe_data.head()
# Calculate weighted average for each category and month
coe_data['weighted_price'] = coe_data['price'] * coe_data['quota']
grouped_data = coe_data.groupby(['year', 'month', 'category']).agg(
    total_weighted_price=('weighted_price', 'sum'),
    total_quota=('quota', 'sum')
).reset_index()
grouped_data['average_price'] = grouped_data['total_weighted_price'] / grouped_data['total_quota']

# Calculate monthly COE index
monthly_coe_index = grouped_data.groupby(['year', 'month']).agg(
    coe_index=('average_price', 'sum')
).reset_index()

monthly_coe_index.head()
# Round the coe_index to integer values
monthly_coe_index['coe_index'] = (monthly_coe_index['coe_index'].round(0).astype(int))/10000

month_mapping = {
    'january': 1, 'february': 2, 'march': 3, 'april': 4,
    'may': 5, 'june': 6, 'july': 7, 'august': 8,
    'september': 9, 'october': 10, 'november': 11, 'december': 12
}
monthly_coe_index['month'] = monthly_coe_index['month'].map(month_mapping)
# Save the rounded data to a CSV file
file_path_rounded = "auxiliary-data/coe_index.csv"
monthly_coe_index.to_csv(file_path_rounded, columns=['year', 'month', 'coe_index'], index=False)

