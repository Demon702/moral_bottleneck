import openai
import os
import pandas as pd
import os

import re
from openai import OpenAI
import json
from typing import Dict, Tuple, List, Union
from dataclasses import dataclass
from tqdm import tqdm
import csv
import random
import os 
import argparse
import sys
sys.path = ['.'] + sys.path
from gptinference.openai_wrapper import OpenAIWrapper

random.seed(0)

class DyadicFrame:
    def __init__(self, circumstance: str, gen: Dict[str, str]):
        self.circumstance = circumstance
        self.agent = gen.get("Answer to Q1", "")
        self.assumption_for_agent = gen.get("Answer to Q2", "")
        self.patient = gen.get("Answer to Q3", "")
        self.assumption_for_patient = gen.get("Answer to Q4", "")
        self.vulnerable_score = self.process_num(gen.get("Answer to Q5", ""))
        self.vulnerable_expl = gen.get("Reasoning for Q5", "")
        self.intentional_score = self.process_num(gen.get("Answer to Q6", ""))
        self.intentional_expl = gen.get("Reasoning for Q6", "")
        self.harm_score = self.process_num(gen.get("Answer to Q7", ""))
        self.harm_expl = gen.get("Reasoning for Q7", "")
        self.help_score = self.process_num(gen.get("Answer to Q8", ""))
        self.help_expl = gen.get("Reasoning for Q8", "")
        self.moral_score = self.process_num(gen.get("Answer to Q9", ""))
        self.moral_expl = gen.get("Reasoning for Q9", "")

    def process_num(self, num):
        if isinstance(num, str):
            try:
                num = float(num)
            except Exception as exc:
                print(f'Exception occurred in processing num: {num}\n {exc}')
        return num

    def as_json(self):
        return {
            "agent": self.agent,
            "assumption_for_agent": self.assumption_for_agent,
            "patient": self.patient,
            "assumption_for_patient": self.assumption_for_patient,
            "vulnerable_score": self.vulnerable_score,
            "vulnerable_expl": self.vulnerable_expl,
            "intentional_score": self.intentional_score,
            "intentional_expl": self.intentional_expl,
            "harm_score": self.harm_score,
            "harm_expl": self.harm_expl,
            "help_score": self.help_score,
            "help_expl": self.help_expl,
            "moral_score": self.moral_score,
            "moral_expl": self.moral_expl
        }

    def as_tsv(self, sep: str):
        return sep.join(f"{x}" for x in [self.vulnerable_score, self.intentional_score, self.harm_score, self.help_score, self.moral_score,
                        self.vulnerable_expl, self.intentional_expl, self.harm_expl, self.help_expl, self.moral_expl,
                        self.agent, self.assumption_for_agent, self.patient, self.assumption_for_patient])

class DyadicFrameWrapper:
    def __init__(self, scenario: str, generation_json: str, circumstance: str = '', circumstance_type: str = 'orig', human_score: float = 0.0):
        self.scenario = scenario
        self.circumstance = circumstance
        self.circumstance_type = circumstance_type
        # self.generation_str = generation_str
        self.generation_json = {}
        self.moral_frames: List[DyadicFrame] = []
        self.human_score = human_score

        self.generation_json = generation_json
        if self.generation_json:
            self.moral_frames.append(DyadicFrame(circumstance=circumstance, gen=self.generation_json))
        
    def as_json(self, id_: str, dataset: str):
        return {
            "id": id_,
            "scenario": self.scenario,
            'circumstance': self.circumstance,
            "prompt": self.prompt,
            "generation_json": self.generation_json,
            "moral_frames": [x.as_json() for x in self.moral_frames],
            "dataset": dataset,
            "human_score": self.human_score
        }

    def as_tsvs(self, sep: str, scenario_id: str, participant_idx: str, dataset: str) -> List[List[str]]:
        tsvs = []
        for x in self.moral_frames:
            arr = sep.join([scenario_id, participant_idx, dataset, self.scenario, self.circumstance_type, self.circumstance]) + sep + x.as_tsv(sep) + sep + str(self.human_score)
            tsvs.append(arr)
        return tsvs

class DyadicFrameMaker:

    def __init__(self, engine: str, openai_wrapper: OpenAIWrapper, client: OpenAI):
        super().__init__()
        self.client = client
        self.engine = engine
        self.openai_wrapper = openai_wrapper
        self.question_metadata = json.load(open('../wvs_newdata/question_metadata.json'))


    def extract_largest_json(self, text):
        def find_matching_brace(text, start):
            count = 0
            for i in range(start, len(text)):
                if text[i] == '{':
                    count += 1
                elif text[i] == '}':
                    count -= 1
                if count == 0:
                    return i
            return -1

        json_objects = []
        start = 0

        while True:
            start = text.find('{', start)
            if start == -1:
                break
            end = find_matching_brace(text, start)
            if end == -1:
                break
            try:
                json_str = text[start:end+1]
                json_obj = json.loads(json_str)
                json_objects.append(json_obj)
            except json.JSONDecodeError:
                pass
            start = end + 1

        if json_objects:
            largest_json = max(json_objects, key=lambda x: len(json.dumps(x)))
            return largest_json
        else:
            print('Error in parsing json:', text)
            return {}
        
    def process_ans(self, question_id, ans, row):
        QUESTION_METADATA = self.question_metadata['QUESTION_METADATA']
        choices = QUESTION_METADATA[question_id]['choices']
        extra_choices = QUESTION_METADATA[question_id].get('extra_choices', {})
        other_choice = extra_choices.get('other_choice', None)
        other_choice_field = extra_choices.get('other_choice_field', None)
        no_choice = extra_choices.get('no_choice', None)
        if ans == other_choice:
            return row[other_choice_field]
        elif ans == no_choice:
            return None
        return choices.get(ans, None)

    def get_demographic_query(self, demographic_info: pd.Series):
        query = ''
        for idx, (question_id, question_text) in enumerate(self.question_metadata['QUESTION_ID_TO_TEXT_MAPPING'].items()):
            ans = demographic_info[question_id]
            if pd.isna(ans):
                continue
            ans = str(ans)
            if ',' in ans:
                ans = [str(int(float(k))) for k in ans.split(',')]
            else:
                ans = str(int(float(ans)))
            if question_id not in ['age', 'politics']:
                if isinstance(ans, list):
                    ans = ','.join([self.process_ans(question_id, a, demographic_info) for a in ans])
                    
                else:
                    ans = self.process_ans(question_id, ans, demographic_info)
            if ans is not None:
                query += f'D{idx + 1}. ' + question_text.replace('{}', ans) + '\n'
        return query

    def make_query(self, scenario: str, circumstance: str = '', demographic_info: pd.Series = None) -> str:
        if scenario[-1] != '.':
            scenario += '.'
        if circumstance is None:
            circumstance = ''
        if circumstance and circumstance[-1] != '.':
            circumstance += '.'
        demographic_query = f'Consider a person with the following demographics:\n'
        demographic_query += self.get_demographic_query(demographic_info)

        query1 = f'\nNow, consider this person is thinking about this "scenario": {scenario} {circumstance}'
        query2 = """

We define a "frame" for a scenario as the answers this person will provide to the following set of questions.

Q1. Who was the agent of this action (should be a person)?

Q2. What assumption did you make about the agent?

Q3. Who was the patient being affected by this action?

Q4. What assumption did the person make about the patient?

Q5. On a scale of 0-10, how vulnerable was the patient?

Q6. On a scale of 0-10, how intentional was the action?

Q7. On a scale of 0-10, how much harm was caused?

Q8. On a scale of 0-10, how much help was caused?

Q9. Based on the answers to Q5-Q8, on a scale of 1 to 7, how morally acceptable is this scenario (1 means extremely immoral, 7 means highly moral)?

Create a frame for the given scenario. The frame contains the person's answer for each question and a very short reasoning and dempgraphics that influence the person's answer for Q5-Q9. Your output format should be as follows:
{
    "Answer to Q1": "...",
    "Answer to Q2": "...",
    "Answer to Q3": "...",
    "Answer to Q4": "...",
    "Demographic used to explain person's answer for Q5": ["D1", "..", ...],
    "Reasoning for Q5": "... (maximum 30 words)",
    "Answer to Q5": "... (a number between 0 and 10)",
    "Demographic used to explain person's answer for Q6": ["D1", "..", ...],
    "Reasoning for Q6": "... (maximum 30 words)",
    "Answer to Q6": "... (a number between 0 and 10)",
    "Demographic used to explain person's answer for Q7": ["D1", "..", ...],
    "Reasoning for Q7": "... (maximum 30 words)",
    "Answer to Q7": "... (a number between 0 and 10)",
    "Demographic used to explain person's answer for Q8": ["D1", "..", ...],
    "Reasoning for Q8": "... (maximum 30 words)",
    "Answer to Q8": "... (a number between 0 and 10)",
    "Reasoning for Q9": "... (maximum 30 words)",
    "Answer to Q9": "... (a number between 1 and 7)"
}
"""
        return demographic_query + query1 + query2

    def __call__(self, scenario: str, circumstance: str, circumstance_type: str, demographic_info: pd.Series, human_score: Union[int, float], max_tokens: int = 512, temperature: float = 0.0) -> DyadicFrameWrapper:
        generation_query = self.make_query(scenario=scenario, circumstance=circumstance, demographic_info=demographic_info)
        # completion = self.client.chat.completions.create(
        #     model=self.engine,
        #     messages=[
        #         {
        #            "role": "user",
        #            "content": generation_query
        #         }
        #     ],
        #     temperature=temperature,
        #     max_tokens=max_tokens,
        #     stop=['\n\n'],
        #     n=num_generations
        # )

        # print('generation_query', generation_query)
        # # if num_generations == 1:
        # generated_str = completion.choices[0].message.content.strip()
        generated_str = self.openai_wrapper.call(
            prompt=generation_query,
            engine=self.engine,
            max_tokens=max_tokens,
            temperature=0.0
        )
        # print('generated str', generated_str)
        generation_json = self.extract_largest_json(generated_str)
        scenario = scenario.replace('\n', '')
        if not generation_json:
            print(f'Error in parsing json for scenario: {scenario}, circumstance: {circumstance}')

        return DyadicFrameWrapper(scenario=scenario, circumstance=circumstance, circumstance_type=circumstance_type, generation_json=generation_json, human_score=human_score)
        
       
def do_inference(maker: DyadicFrameMaker, items: List[Tuple[str, str]], temperature: float = 0.0) -> List[DyadicFrameWrapper]:
    outputs = []
    for item in tqdm(items):
        scenario_id = item[0]
        participant_idx = item[1]
        dataset = item[2]
        scenario = item[3]
        circumstance_type = item[4]
        circumstance = item[5]
        demographic_info = item[6]
        human_score = item[7]
        prompt = ""

        try:
            cf = maker(scenario=scenario, circumstance=circumstance, circumstance_type=circumstance_type, demographic_info=demographic_info, human_score=human_score)
            outputs.append(cf)
        except Exception as exc:
            print(f"Exception ({exc}) in input: {item}")
            outputs.append(DyadicFrameWrapper(scenario=scenario, circumstance=circumstance, circumstance_type=circumstance_type, human_score=human_score, generation_json={}))
    return outputs


def sample_random_indexes(choices, num):
    return random.sample(choices, num)

def get_match_group(string, regex):
  if not re.match(regex, string):
      return None
  
  # print(re.match(regex, string))
  return re.match(regex, string).group(1)

def get_no_of_participants(df, scenario):
    feature_cols = ['harm', 'help', 'intent', 'vuln']
    target_col = 'moral'
    sc_feature_cols = [f'X{scenario}_orig_{feature_col}' for feature_col in feature_cols ]
    sc_target_col = f'X{scenario}_orig_{target_col}'
    return df[sc_feature_cols + [sc_target_col]].dropna().shape[0]

def read_new_llm_scenarios(scenarios_path: str, df_path: str, all_circumstances: bool = False) -> List[Tuple[str, str]]:
    scenario_df = pd.read_csv(scenarios_path)
    scenario_regex = r'X(\d+)_orig_moral'
    columns = list(scenario_df.columns)
    scenarios = [get_match_group(col, scenario_regex) for col in columns ]
    scenarios = [int(sc) for sc in scenarios if sc]
    df = pd.read_csv(df_path, sep='\t')
    
    items  = []
    for id in scenarios:
        num_circumstances = 4 if all_circumstances else 1
        rows = df[df['id'] == id].iloc[:num_circumstances]
        scenario = rows['scenario'].values[0]

        num_participants = get_no_of_participants(scenario_df, id)
        for i, (_, row) in enumerate(rows.iterrows()):
            circumstance_type = 'orig' if i == 0 else f'C{i}'
            circumstance = row['circumstance'] if not pd.isna(row['circumstance']) else ''
            filtered_scenario_df = scenario_df.dropna(subset=[f'X{id}_{circumstance_type}_moral'])
            for participant_idx, participant_row in filtered_scenario_df.iterrows():
                demographic_info = participant_row.loc['gender': 'politics']
                human_score = participant_row[f'X{id}_{circumstance_type}_moral']
                items.append((str(id), str(participant_idx), row['dataset'], scenario, circumstance_type, circumstance, demographic_info, human_score))
    return items


def save_to_file(outpath_jsonl, inputs: Tuple[str, str], outputs: List[DyadicFrameWrapper], append=False):
    sep="\t"
    columns = ['scenario_id', 'participant_id', 'dataset', 'scenario', 'circumstance_type', 'circumstance', 'vulnerable_score', 'intentional_score', 'harm_score', 'help_score', 'moral_score',
            'vulnerable_expl', 'intentional_expl', 'harm_expl', 'help_expl', 'moral_expl', 'agent', 'assumption_for_agent',
            'patient', 'assumption_for_patient', 'human_score']
    
    outpath_tsv_default_circumstance = outpath_jsonl[:-4] + '.tsv'
    list_output_tsvs_def_circumstance = [output.as_tsvs(sep=sep, scenario_id=input[0], participant_idx=input[1], dataset=input[2]) for input, output in zip(inputs, outputs)]
    with open(outpath_tsv_default_circumstance, 'w' if not append else 'a') as open_file:
        open_file.write(sep.join(columns))
        open_file.write('\n')
        for output_tsvs in list_output_tsvs_def_circumstance:
            for output_tsv in output_tsvs:
                open_file.write(output_tsv)
            open_file.write('\n')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', type=str, default='gpt-3.5-turbo')
    # parser.add_argument('--use-together-api', action='store_true', default=False)
    parser.add_argument('--max-tokens', type=int, default=64)
    parser.add_argument('--all-circumstances', action='store_true', default=False)
    parser.add_argument('--temperature', type=float, default=0.0)



    args = parser.parse_args()
    # base_url = None if not args.use_together_api else 'https://api.together.xyz/v1'
    # key_env_variale = 'TOGETHER_API_KEY' if args.use_together_api else 'KEY'
    model_dir_name = args.model.replace('/', '_')
    os.makedirs(f'demographic/cache/{model_dir_name}', exist_ok=True)
    openai_wrapper = OpenAIWrapper(cache_path=f'''demographic/cache/{model_dir_name}/dyadic.jsonl''')

    # client = OpenAI(
    #     api_key=os.environ.get(key_env_variale),
    #     base_url=base_url
    # )
    client = None
    items = read_new_llm_scenarios('../wvs_newdata/wvs_spanishlang.csv', '../moral_data_circumstances/dyadic/dyadic_all.tsv', all_circumstances=args.all_circumstances)

    maker = DyadicFrameMaker(engine=args.model, openai_wrapper=openai_wrapper, client=client)

    cfs = do_inference(items=items, maker=maker, temperature=args.temperature)
    os.makedirs(f'demographic/results/{model_dir_name}', exist_ok=True)
    save_to_file(outpath_jsonl=f'''demographic/results/{model_dir_name}/dyadic{'_all_circumstances' if args.all_circumstances else ''}.tsv''', inputs=items, outputs=cfs)
