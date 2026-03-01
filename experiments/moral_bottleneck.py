import os
import pandas as pd
import os
import openai
import re
# from openai import OpenAI
import json
from typing import Dict, Tuple, List
from dataclasses import dataclass
from tqdm import tqdm
import csv
import random
import os 
import argparse
import re
from gptinference.openai_wrapper import OpenAIWrapper
from concurrent.futures import ThreadPoolExecutor, as_completed

# openai.api_key = os.environ.get('KEY')
random.seed(0)


class MoralTheoryFrame:
    def __init__(self, circumstance: str, gen: Dict[str, str], num_addn_qns: int = 0):
        self.num_questions = (len(gen) - num_addn_qns) // 2 + num_addn_qns
        self.circumstance = circumstance
        self.scores = []
        self.expls = []
        self.num_addn_qns = num_addn_qns
        if self.num_addn_qns > 0:
            self.agent = gen.get("Answer to Q1", "")
            self.agent_assumption = gen.get("Answer to Q2", "")
            self.affected_party = gen.get("Answer to Q3", "")
            self.affected_party_assumption = gen.get("Answer to Q4", "")

        for i in range(self.num_addn_qns + 1, self.num_questions):
            processed_num = self.process_num(gen.get(f"Answer to Q{i}", ""))
            self.scores.append(processed_num)
            if not isinstance(processed_num, (int, float)):
                print(f'Error in processing num in:\n\n{gen}\n\n')
            self.expls.append(gen.get(f"Reasoning for Q{i}", ""))

        procssed_moral_score = self.process_num(gen.get(f"Answer to Q{self.num_questions}", ""))
        if not isinstance(procssed_moral_score, (int, float)):
            print(f'Error in processing num in:\n\n{gen}\n\n')
        self.moral_score = procssed_moral_score
        self.moral_expl = gen.get(f"Reasoning for Q{self.num_questions}", "")

    def process_num(self, num):
        if isinstance(num, str):
            try:
                num = float(num)
            except Exception as exc:
                print(f'Exception occurred in processing num: {num}\n {exc}')
        return num

    def as_json(self):
        scores_and_expls = {}
        for i in range(self.num_addn_qns + 1, self.num_questions):
            scores_and_expls[f"Q{i}_score"] = self.scores[i - self.num_addn_qns - 1],
            scores_and_expls[f"Q{i}_expl"] = self.expls[i -self.num_addn_qns - 1]

        additional_qns = {}
        if self.num_addn_qns > 0:
            additional_qns = {
                "agent": self.agent,
                "agent_assumption": self.agent_assumption,
                "affected_party": self.affected_party,
                "affected_party_assumption": self.affected_party_assumption
            }
        return {
            **additional_qns,
            **scores_and_expls,
            "moral_score": self.moral_score,
            "moral_expl": self.moral_expl
        }

    def as_tsv(self, sep: str):

        arr = []
        if self.num_addn_qns > 0:
            arr.extend([f"{self.agent}", f"{self.agent_assumption}", f"{self.affected_party}", f"{self.affected_party_assumption}"])
        for i in range(self.num_addn_qns + 1, self.num_questions):
            arr.extend([f"{self.scores[i - self.num_addn_qns - 1]}", f"{self.expls[i - self.num_addn_qns - 1]}"])
        arr.extend([f"{self.moral_score}", f"{self.moral_expl}"])
        return sep.join(arr)

class MoralTheoryFrameWrapper:
    def __init__(self, id_: int, scenario: str, prompt: str, generation_json: str, circumstance: str = '', circumstance_type: str = 'orig', human_score: float = 0.0, num_addn_qns: int = 0):
        self.scenario = scenario
        self.circumstance = circumstance
        self.circumstance_type = circumstance_type
        self.prompt = prompt
        # self.generation_str = generation_str
        self.generation_json = {}
        self.moral_frames: List[MoralTheoryFrame] = []
        self.human_score = human_score

        self.generation_json = generation_json
        self.id_ = id_

        # try:
        #     self.generation_json = json.loads(self.generation_str)
        # except Exception as exc:
        #     print(f'Exception occurred in scenario: {scenario}\n {exc}, \nGeneration str: {self.generation_str}')

        if self.generation_json:
            self.moral_frames.append(MoralTheoryFrame(circumstance=circumstance, gen=self.generation_json, num_addn_qns=num_addn_qns))
    
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

    def as_tsvs(self, sep: str, dataset: str) -> List[List[str]]:
        tsvs = []
        for x in self.moral_frames:
            arr = sep.join([str(self.id_), dataset, self.scenario, self.circumstance_type, self.circumstance]) + sep + x.as_tsv(sep) + sep + str(self.human_score)
            tsvs.append(arr)
        return tsvs

class MoralTheoryFrameMaker:

    def __init__(self, engine: str, client = None, openai_wrapper = None, theory_prompt: str = '', num_addn_qns: int = 0, verbose: bool = False):
        super().__init__()
        self.client = client
        self.engine = engine
        self.openai_wrapper = openai_wrapper
        self.theory_prompt = theory_prompt
        self.num_addn_qns = num_addn_qns
        self.verbose = verbose

    def make_query(self, scenario: str, circumstance: str = '') -> str:
        if scenario[-1] != '.':
            scenario += '.'
        query = f'Consider a "scenario": {scenario}'
        if circumstance:
          query += f'''\nAlso consider the following unlerlying "circumstance": {circumstance}.'''
        query += f"\n\n{self.theory_prompt}"
        return query
    
    def extract_largest_json(self, text):
        try:
            js = json.loads(text)
            return js
        except json.JSONDecodeError:
            pass

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

    def post_process_generation(self, generated_str: str) -> str:
        if self.engine == "deepseek-ai/DeepSeek-R1":
            # Remove the content inside the <think> ... </think> tags
            generated_str = re.sub(r"<think>(.*)</think>", "", generated_str, flags=re.DOTALL)

        return generated_str

    def __call__(self, id_: int, scenario: str, circumstance: str, circumstance_type: str, human_score: float, max_tokens: int= 512) -> MoralTheoryFrameWrapper:

        try:
            generation_query = self.make_query(scenario=scenario, circumstance=circumstance)

            if self.verbose:
                print('generation_query', generation_query)
            if self.client is not None:
                completion = self.client.chat.completions.create(
                    model=self.engine,
                    messages=[
                        {
                        "role": "user",
                        "content": generation_query
                        }
                    ],
                    # temperature=0.0,
                    # max_tokens=max_tokens if not self.engine.startswith("o1") else None
                    # stop=['\n\n']
                )

                generated_str = completion.choices[0].message.content.strip()

            else:
                generated_str = self.openai_wrapper.call(
                    prompt=generation_query,
                    engine=self.engine,
                    max_tokens=max_tokens,
                    temperature=0.0
                )

            if self.verbose:
                print('generated str', generated_str)

            # print("\n" * 10)
            generated_str = self.post_process_generation(generated_str)

            if self.verbose:
                print('post processed generated str', generated_str)
            generation_json = self.extract_largest_json(generated_str)
            if not generation_json:
                print(f'Error in parsing json for scenario: {scenario}, circumstance: {circumstance}')
            scenario = scenario.replace('\n', '')
            return MoralTheoryFrameWrapper(id_=id_, scenario=scenario, circumstance=circumstance, circumstance_type=circumstance_type, prompt=generation_query, generation_json=generation_json, human_score=human_score, num_addn_qns=self.num_addn_qns)
        except Exception as exc:
            print(f"Exception occurred in scenario: {scenario}, circumstance: {circumstance}, Error: {exc}")
            return MoralTheoryFrameWrapper(id_=id_, scenario=scenario, circumstance=circumstance, circumstance_type=circumstance_type, prompt="", generation_json={}, human_score=human_score, num_addn_qns=self.num_addn_qns)
 

def do_inference(
        maker: MoralTheoryFrameMaker,
        items: List[Tuple[str, str]],
        max_tokens: int = 512,
        max_workers: int = 10
    ) -> List[MoralTheoryFrameWrapper]:
    outputs = []
    futures = []
    with ThreadPoolExecutor(max_workers=10) as executor:
        for item in items:
            id_ = item[0]
            scenario = item[2]
            circumstance = item[3]
            circumstance_type = item[4]
            human_score = item[5]

            args = (id_, scenario, circumstance, circumstance_type, human_score, max_tokens)
            futures.append(executor.submit(maker, *args))

        for future in tqdm(as_completed(futures), total=len(futures)):
            outputs.append(future.result())
    outputs = sorted(outputs, key=lambda x: int(x.id_))
    return outputs


def sample_random_indexes(choices, num):
    return random.sample(choices, num)


def read_new_llm_scenarios(df_path: str, all_circumstances: bool = False) -> List[Tuple[str, str]]:
    df = pd.read_csv(df_path, sep='\t')
    scenarios = df['id'].unique().tolist()
    items  = []

    for id in scenarios:
        num_circumstances = 4 if all_circumstances else 1
        rows = df[df['id'] == id].iloc[: num_circumstances]
        for i, (_, row) in enumerate(rows.iterrows()):
            circumstance_type = 'orig' if i == 0 else f'C{i}'
            circumstance = row['circumstance'] if not pd.isna(row['circumstance']) else ''
            human_score = row['human_score_normalized [-4,4]']
            items.append((str(row['id']), row['dataset'], row['scenario'], circumstance, circumstance_type, human_score))
    return items


def save_to_file(outpath_jsonl, inputs: Tuple[str, str], outputs: List[MoralTheoryFrameWrapper], append=False):
    sep="\t"

    num_questions = outputs[0].moral_frames[0].num_questions
    num_addn_qns = outputs[0].moral_frames[0].num_addn_qns
    columns = ['id', 'dataset', 'scenario', 'circumstance_type', 'circumstance']
    if num_addn_qns > 0:
        columns += ["agent", "agent_assumption", "affected_party", "affected_party_assumption"]
    for i in range(num_addn_qns + 1, num_questions):
        columns.extend([f"Q{i}_score", f"Q{i}_expl"])
    columns.extend(['moral_score', 'moral_expl', 'human_score'])

    outpath_tsv_default_circumstance = outpath_jsonl[:-4] + '.tsv'
    list_output_tsvs_def_circumstance = [output.as_tsvs(sep=sep, dataset=input[1]) for input, output in zip(inputs, outputs)]
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
    parser.add_argument('--theory-name', type=str, default=None)
    parser.add_argument('--theory-prompt-path', type=str, default=None)
    parser.add_argument('--max-tokens', type=int, default=512)
    parser.add_argument('--all-circumstances', action='store_true', default=False)
    parser.add_argument('--num-addn-qns', type=int, default=0)
    parser.add_argument('--max-workers', type=int, default=10)
    parser.add_argument('--verbose', action='store_true', default=False)

    args = parser.parse_args()
    model_dir_name = args.model.replace('/', '_')
   
    data_file_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), '../data/moral_data_circumstances/dyadic/all_dyadic.tsv')
    print('data_file_path', data_file_path)
    items = read_new_llm_scenarios(data_file_path, all_circumstances=args.all_circumstances)

    theory_name = args.theory_name

    os.makedirs(f'cache/{model_dir_name}', exist_ok=True)
    openai_wrapper = OpenAIWrapper(cache_path=f'''cache/{model_dir_name}/{theory_name}.jsonl''')

    client = None

    theory_prompt = ''
    if args.theory_prompt_path:
        with open(args.theory_prompt_path, 'r') as f:
            theory_prompt = f.read()

    MODEL_CONFIG = {
        "gpt-4o": "gpt-4o-2024-05-13",
        "gpt-3.5-turbo": "gpt-3.5-turbo",
        "meta/llama3-70b-instruct": "meta-llama/Llama-3-70b-chat-hf",
        "meta/llama3-8b-instruct": "meta-llama/Llama-3-8b-chat-hf",
        "mistralai/mixtral-8x22b-instruct-v0.1": "mistralai/Mixtral-8x22B-Instruct-v0.1",
        "mistralai/mixtral-8x7b-instruct-v0.1": "mistralai/Mixtral-8x7B-Instruct-v0.1",
        "o1": "o1-2024-12-17",
        "o3-mini": "o3-mini-2025-01-31",
        "deepseek-ai/DeepSeek-R1": "deepseek-ai/DeepSeek-R1"
    }

    print("Model engine:", MODEL_CONFIG.get(args.model, args.model))
    num_addn_qns = args.num_addn_qns
    maker = MoralTheoryFrameMaker(
        engine=MODEL_CONFIG.get(args.model, args.model),
        openai_wrapper=openai_wrapper,
        client=client,
        theory_prompt=theory_prompt,
        num_addn_qns=num_addn_qns,
        verbose=args.verbose
    )
    cfs = do_inference(items=items, maker=maker, max_tokens=args.max_tokens, max_workers=args.max_workers)
    os.makedirs(f'results/{model_dir_name}', exist_ok=True)
    save_to_file(outpath_jsonl=f'''results/{model_dir_name}/{theory_name}{'_all_circumstances' if args.all_circumstances else ''}.tsv''', inputs=items, outputs=cfs)
