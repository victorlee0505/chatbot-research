import pandas as pd

# Assuming you have a DataFrame named df
df = pd.read_csv('./data/excel/sunshine_difference.csv')


# Print the modified DataFrame
print(df.count())
print(df['First Name'].values)