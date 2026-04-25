import openai
import os
import pandas as pd
import os
import re
from openai import OpenAI
import json
from typing import Dict, Tuple, List
from dataclasses import dataclass
from tqdm import tqdm
import csv
import random
import os 
import argparse
from gptinference.openai_wrapper import OpenAIWrapper
from concurrent.futures import ThreadPoolExecutor, as_completed

random.seed(0)

class MoralFrame:
    def __init__(self, circumstance: str, gen: Dict[str, str]):
        self.circumstance = circumstance
        if isinstance(gen, Dict):
            self.moral_score = self.process_num(gen["moral acceptability score"])
            self.moral_expl = gen["explanation"]
            self.all_moral_scores = [self.moral_score]
        elif isinstance(gen, List):
            self.all_moral_scores = [self.process_num(x["moral acceptability score"]) for x in gen]
            self.moral_expl = [x["explanation"] for x in gen]
            self.moral_score = sum(self.all_moral_scores) / len(self.all_moral_scores)

    def process_num(self, num):
        if isinstance(num, str):
            try:
                num = float(num)
            except Exception as exc:
                print(f'Exception occurred in processing num: {num}\n {exc}')
        return num

    def as_json(self):
      return {
                "moral_score": self.moral_score,
                "moral_expl": self.moral_expl,
                "all_moral_scores": self.all_moral_scores
            }

    def as_tsv(self, sep: str):
        return sep.join(f"{x}" for x in [self.moral_score, self.moral_expl, self.all_moral_scores])


class MoralFrameWrapper:
    def __init__(self, id_: int, scenario: str, prompt: str, generation_json: str, circumstance: str = '', circumstance_type: str = 'orig', human_score: float = 0.0):
        self.id_ = id_
        self.scenario = scenario
        self.circumstance = circumstance
        self.circumstance_type = circumstance_type
        self.prompt = prompt
        self.generation_json = {}
        self.moral_frames: List[MoralFrame] = []
        self.human_score = human_score

        self.generation_json = generation_json
        if self.generation_json:
            self.moral_frames.append(MoralFrame(circumstance=circumstance, gen=self.generation_json))
        
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

    def as_tsvs(self, sep: str, id_: str, dataset: str) -> List[List[str]]:
        tsvs = []
        for x in self.moral_frames:
            arr = sep.join([id_, dataset, self.scenario, self.circumstance_type, self.circumstance]) + sep + x.as_tsv(sep) + sep + str(self.human_score)
            tsvs.append(arr)
        return tsvs

class MoralFrameMaker:
    def __init__(self,
            engine: str,
            openai_wrapper: OpenAIWrapper,
            client: OpenAI,
            template_path: str,
            verbose: bool = False,
            template_keyword: str = ''
        ):
        super().__init__()
        self.client = client
        self.engine = engine
        self.openai_wrapper = openai_wrapper
        self.template = self.load_template(template_path)
        self.verbose = verbose
        self.template_keyword = template_keyword

    def load_template(self, template_path: str) -> str:
        with open(template_path, 'r') as f:
            return f.read()

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

    def make_query(self, scenario: str, circumstance: str = '') -> str:
        if scenario[-1] != '.':
            scenario += '.'

        if circumstance and circumstance[-1] != '.':
            circumstance += '.'

        # Replace the {scenario} placeholder in the template
        query = self.template.format(scenario=scenario + ' ' + circumstance)
        return query

    def post_process_generation(self, generated_str: str) -> str:
        if self.engine == "deepseek-ai/DeepSeek-R1":
            # Remove the content inside the <think> ... </think> tags
            generated_str = re.sub(r"<think>(.*)</think>", "", generated_str, flags=re.DOTALL)

        if self.template_keyword in ['cot_template', 'cot_few_shot_template']:
            # Remove the content inside the <think> ... </think> tags
            generated_str = re.sub(r"<think>(.*)</think>", "", generated_str, flags=re.DOTALL)

            # Extract the content inside the <answer> ... </answer> tags
            answer_match = re.search(r"<answer>(.*)</answer>", generated_str, flags=re.DOTALL)

            if answer_match:
                generated_str = answer_match.group(1)

        if self.template_keyword in ['cot_few_shot_template_thinking', 'cot_template_thinking']:
            # Qwen-friendly tag variant: strip <thinking>...</thinking>, extract LAST <answer>...</answer>.
            # Use rfind to avoid greedy regex spanning multiple example <answer> blocks if the model
            # echoes the few-shot prompt.
            last_open = generated_str.rfind("<thinking>")
            last_close = generated_str.rfind("</thinking>")
            if last_open != -1 and last_close != -1 and last_close > last_open:
                generated_str = generated_str[:last_open] + generated_str[last_close + len("</thinking>"):]

            last_ans_open = generated_str.rfind("<answer>")
            last_ans_close = generated_str.rfind("</answer>")
            if last_ans_open != -1 and last_ans_close != -1 and last_ans_close > last_ans_open:
                generated_str = generated_str[last_ans_open + len("<answer>"):last_ans_close]

        return generated_str

    def __call__(self, id_: int, scenario: str, circumstance: str, circumstance_type: str, human_score: float, num_generations: int = 1, max_tokens: int = 64, temperature: float = 0.0) -> MoralFrameWrapper:
        try:
        
            generation_query = self.make_query(scenario=scenario, circumstance=circumstance)
            # if num_generations > 1:
            #     assert temperature > 0.0, "Temperature must be greater than 0.0 for multiple generations"

            if self.verbose:
                print(f"Generation query: {generation_query}")

            if self.client is not None:
                completion = self.client.chat.completions.create(
                    model=self.engine,
                    messages=[
                        {
                        "role": "user",
                        "content": generation_query
                        }
                    ],
                    # reasoning_effort="medium"
                    # temperature=0.0
                    max_tokens=max_tokens,
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
                print(f"Generated string: {generated_str}")

            generated_str = self.post_process_generation(generated_str)
            generation_json = self.extract_largest_json(generated_str)

            if self.verbose:
                print(f"Generation JSON: {generation_json}")

            scenario = scenario.replace('\n', '')
            if not generation_json:
                print(f'Error in parsing json for scenario: {scenario}, circumstance: {circumstance}')

            return MoralFrameWrapper(id_=id_, scenario=scenario, circumstance=circumstance, circumstance_type=circumstance_type, prompt=generation_query, generation_json=generation_json, human_score=human_score)
        except Exception as exc:
            print(f"Exception occurred in scenario: {scenario}, circumstance: {circumstance}, Error: {exc}")
            return MoralFrameWrapper(id_=id_, scenario=scenario, circumstance=circumstance, circumstance_type=circumstance_type, prompt=generation_query, generation_json={}, human_score=human_score)

def do_inference(maker: MoralFrameMaker, items: List[Tuple[str, str]], temperature: float = 0.0, max_tokens: int = 64, max_workers: int = 10) -> List[MoralFrameWrapper]:
    outputs = []
    futures = []
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        for item in items:
            id_ = item[0]
            scenario = item[2]
            circumstance = item[3]
            circumstance_type = item[4]
            human_score = item[5]
            num_generations = item[6]

            args = (id_, scenario, circumstance, circumstance_type, human_score, num_generations, max_tokens, temperature)
            futures.append(executor.submit(maker, *args))

        for future in tqdm(as_completed(futures), total=len(futures)):
            outputs.append(future.result())
    print(f"Number of outputs: {len(outputs)}")
    outputs = sorted(outputs, key=lambda x: int(x.id_))
    return outputs

def read_new_llm_scenarios(df_path: str, all_circumstances: bool = False, num_generarions: int = 1) -> List[Tuple[str, str]]:
    df = pd.read_csv(df_path, sep='\t')
    scenarios = df['id'].unique().tolist()
    items  = []
    for id in scenarios:
        num_circumstances = 4 if all_circumstances else 1
        rows = df[df['id'] == id].iloc[:num_circumstances]
        for i, (_, row) in enumerate(rows.iterrows()):
            circumstance_type = 'orig' if i == 0 else f'C{i}'
            circumstance = row['circumstance'] if not pd.isna(row['circumstance']) else ''
            human_score = row['human_score_normalized [-4,4]']
            items.append((str(row['id']), row['dataset'], row['scenario'], circumstance, circumstance_type, human_score, num_generarions))
    return items

def save_to_file(outpath_jsonl, inputs: Tuple[str, str], outputs: List[MoralFrameWrapper], append=False):
    sep="\t"
    columns = ['id', 'dataset', 'scenario', 'circumstance_type', 'circumstance', 'moral_score', 'moral_expl', 'all_moral_scores', 'human_score']

    outpath_tsv_default_circumstance = outpath_jsonl[:-4] + '.tsv'
    list_output_tsvs_def_circumstance = [output.as_tsvs(sep=sep, id_=input[0], dataset=input[1]) for input, output in zip(inputs, outputs)]
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
    parser.add_argument('--max-tokens', type=int, default=64)
    parser.add_argument('--all-circumstances', action='store_true', default=False)
    parser.add_argument('--num-generations', type=int, default=1)
    parser.add_argument('--temperature', type=float, default=0.0)
    parser.add_argument('--template-path', type=str, default='baseline_prompts/few_shot_template.txt')
    parser.add_argument('--num-addn-qns', type=int, default=0)
    parser.add_argument('--max-workers', type=int, default=10)
    parser.add_argument('--verbose', action='store_true', default=False)

    args = parser.parse_args()
    model_dir_name = args.model.replace('/', '_')


    template_keyword =  args.template_path.split('/')[-1].split('.')[0]
    
    os.makedirs(f'cache/{model_dir_name}', exist_ok=True)
    openai_wrapper = OpenAIWrapper(cache_path=f'''cache/{model_dir_name}/{template_keyword}_end_to_end.jsonl''')
    
    client = OpenAI(
        base_url=os.environ.get('OPENAI_BASE_URL', 'https://api.openai.com/v1'),
        api_key=os.environ.get('OPENAI_API_KEY')
    )

    if args.model.startswith('o'):
        client = OpenAI(
            base_url=os.environ.get('OPENAI_BASE_URL', 'https://api.openai.com/v1'),
            api_key=os.environ.get('OPENAI_API_KEY')
        )
    print(f"api key: {os.environ.get('OPENAI_API_KEY')}")

    MODEL_CONFIG = {
        "gpt-4o": "gpt-4o-2024-05-13",
        "gpt-3.5-turbo": "gpt-3.5-turbo",
        "meta/llama3-70b-instruct": "meta-llama/Llama-3-70b-chat-hf",
        "meta/llama3-8b-instruct": "meta-llama/Llama-3-8b-chat-hf",
        "mistralai/mixtral-8x22b-instruct-v0.1": "mistralai/Mixtral-8x22B-Instruct-v0.1",
        "mistralai/mixtral-8x7b-instruct-v0.1": "mistralai/Mixtral-8x7B-Instruct-v0.1",
        "o1": "o1-2024-12-17",
        "o3-mini": "o3-mini-2025-01-31",
        "deepseek-r1": "deepseek-ai/DeepSeek-R1",
        "qwen3-non-thinking": "Qwen/Qwen3-235B-A22B-fp8-tput",
        "qwen3-thinking": "Qwen/Qwen3-235B-A22B-Thinking-2507"
    }

    engine = MODEL_CONFIG.get(args.model, args.model)
    data_file_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), '../data/moral_data_circumstances/dyadic/all_dyadic.tsv')
    items = read_new_llm_scenarios(data_file_path, all_circumstances=args.all_circumstances, num_generarions=args.num_generations)

    maker = MoralFrameMaker(engine=engine, openai_wrapper=openai_wrapper, client=client, template_path=args.template_path, verbose=args.verbose)

    cfs = do_inference(items=items, maker=maker, temperature=args.temperature, max_tokens=args.max_tokens, max_workers=args.max_workers)
    os.makedirs(f'results/{model_dir_name}', exist_ok=True)
    save_to_file(outpath_jsonl=f'''results/{model_dir_name}/{template_keyword}_end_to_end_all{'_all_circumstances' if args.all_circumstances else ''}.tsv''', inputs=items, outputs=cfs) 