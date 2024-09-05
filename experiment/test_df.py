import pandas as pd
from langchain_community.document_loaders import DataFrameLoader

# Assuming you have a DataFrame named df
# df = pd.read_csv('./data/excel/sunshine_difference.csv')


# # Print the modified DataFrame
# print(df.count())
# print(df['First Name'].values)

df = pd.read_json('./data/doc/text-sample.json')
print(df)
loader = DataFrameLoader(data_frame=df, page_content_column="title")
documents = loader.load()
for doc in documents:
    print(f'{doc}\n')
