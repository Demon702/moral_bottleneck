import pandas as pd
import argparse

parser = argparse.ArgumentParser(description='Merge scenarios from the final experiments')

parser.add_argument('--input1', type=str, help='Path to the input file')
parser.add_argument('--input2', type=str, help='Path to the input file')

args = parser.parse_args()

df1 = pd.read_csv(args.input1, sep='\t').dropna(subset=['id'])
df2 = pd.read_csv(args.input2, sep='\t').dropna(subset=['id'])

print(df1, len(df1))
print(df2, len(df2))
df = pd.concat([df1, df2])

df_sorted = df.sort_values(by='id').drop_duplicates(subset='id', keep='first')

print(df_sorted)
df_sorted.to_csv(args.input1, sep='\t', index=False)

