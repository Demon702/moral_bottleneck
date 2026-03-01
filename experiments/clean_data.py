import argparse
import pandas as pd
import json
import re
parser = argparse.ArgumentParser(description='Clean data from the final experiments')

parser.add_argument('--input', type=str, help='Path to the input file')


args = parser.parse_args()

df = pd.read_csv(args.input, sep='\t')

file_name = args.input.split('/')[-1].split('.')[0] 

FILE_METADATA = json.load(open('evaluation/metadata.json'))

metadata = FILE_METADATA[file_name]
fields_to_check = metadata['feature_cols'] + [metadata['moral_score_col']]

pattern = re.compile(r'(-?\d+).*')
def extract_group(text, pattern):
    match = pattern.search(text)
    return int(match.group(1)) if match else text

for col in fields_to_check:
    df[col] = df[col].apply(lambda x: extract_group(str(x), pattern))

df.to_csv(args.input, sep='\t', index=False)

