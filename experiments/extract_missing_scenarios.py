import argparse
import pandas as pd
parser = argparse.ArgumentParser(description='Extract missing scenarios from the final experiments')

parser.add_argument('--input', type=str, help='Path to the input file')


args = parser.parse_args()

df = pd.read_csv(args.input, sep='\t')
df = df.dropna(subset=['aspect1_score'])
secnario_ids = set(df.dropna(subset=['aspect1_score'])['id'].unique().tolist())
all_scenario_ids = set(range(1, 650))

missing_scenario_ids = list(all_scenario_ids - secnario_ids)

print(missing_scenario_ids)

print('no of missing scenarios:', len(missing_scenario_ids))