import pandas as pd

data = pd.read_csv('sg-stock-prices.csv')
# Consider the monthly rise and fall of a stock as the percentage rise or fall on the last trading day of the month compared to the last trading day of the previous month.
# Only the mathematical average is considered. The market index for that month is the average of the monthly increase and decrease percentages of all stocks.

data.info(), data.head()
data['date'] = pd.to_datetime(data['date'])
data['year'] = data['date'].dt.year
data['month'] = data['date'].dt.month
data = data.sort_values(by=['symbol', 'date'])

last_day_of_month = data.groupby(['symbol', 'year', 'month']).last().reset_index()

last_day_of_month['last_month_close'] = last_day_of_month.groupby('symbol')['close'].shift(1)
last_day_of_month['monthly_return'] = (last_day_of_month['close'] - last_day_of_month['last_month_close']) / \
                                      last_day_of_month['last_month_close']

market_index = last_day_of_month.groupby(['year', 'month'])['monthly_return'].mean().reset_index()
market_index = market_index.rename(columns={'monthly_return': 'market_index'})
last_day_of_month = pd.merge(last_day_of_month, market_index, how='left', on=['year', 'month'])

output_data = last_day_of_month[['symbol', 'year', 'month', 'monthly_return', 'market_index']]
output_file_path = 'stock_monthly_returns_and_market_index.csv'
output_data.to_csv(output_file_path, index=False)
# Output the monthly price increases and decreases of all stocks and the market index (average price changes)

market_index['market_index'] = (market_index['market_index'] * 100).round(4)
formatted_market_index_file_path = 'formatted_market_index_by_month.csv'
market_index.to_csv(formatted_market_index_file_path, index=False)
# Output the market index percentage by year/month, in %, with two decimal places after the percentage.
