import openai
import os
import pandas as pd
import re
from openai import OpenAI
import json
from typing import Dict, Tuple, List
from tqdm import tqdm
import random
import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed

openai.api_key = os.environ.get('KEY')
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


def _flatten_cell(text: str) -> str:
    """Collapse newlines/tabs so a free-form string fits safely in a single TSV cell."""
    if text is None:
        return ""
    return text.replace("\r", " ").replace("\n", " ").replace("\t", " ")


class DyadicFrameWrapper:
    def __init__(self, scenario: str, prompt: str, generation_json: str, circumstance: str = '', circumstance_type: str = 'orig', human_score: float = 0.0, think_trace: str = ''):
        self.scenario = scenario
        self.circumstance = circumstance
        self.circumstance_type = circumstance_type
        self.prompt = prompt
        self.generation_json = {}
        self.moral_frames: List[DyadicFrame] = []
        self.human_score = human_score
        self.think_trace = think_trace

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
            "human_score": self.human_score,
            "think_trace": self.think_trace,
        }

    def as_tsvs(self, sep: str, id_: str, dataset: str) -> List[List[str]]:
        tsvs = []
        for x in self.moral_frames:
            arr = (sep.join([id_, dataset, self.scenario, self.circumstance_type, self.circumstance])
                   + sep + x.as_tsv(sep)
                   + sep + str(self.human_score)
                   + sep + _flatten_cell(self.think_trace))
            tsvs.append(arr)
        return tsvs


class DyadicFrameMaker:

    def __init__(self, engine: str, client: OpenAI, template_path: str, verbose: bool = False):
        super().__init__()
        self.client = client
        self.engine = engine
        self.verbose = verbose
        with open(template_path, 'r') as f:
            self.template = f.read()

    def make_query(self, scenario: str, circumstance: str = '') -> str:
        if scenario[-1] != '.':
            scenario += '.'
        if circumstance and circumstance[-1] != '.':
            circumstance += '.'
        scenario_plus = scenario + (' ' + circumstance if circumstance else '')
        return self.template.format(scenario=scenario_plus)

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

    def post_process_generation(self, generated_str: str) -> Tuple[str, str]:
        # Capture the LAST <thinking>...</thinking> block — the template has examples that
        # would otherwise be captured if the model echoes the first example tags. Using a
        # greedy match + rfind keeps us on the final pair which wraps the actual response.
        # Simpler: find the last <thinking> and last </thinking>.
        last_open = generated_str.rfind("<thinking>")
        last_close = generated_str.rfind("</thinking>")
        if last_open != -1 and last_close != -1 and last_close > last_open:
            think_trace = generated_str[last_open + len("<thinking>"):last_close].strip()
            generated_str = generated_str[:last_open] + generated_str[last_close + len("</thinking>"):]
        else:
            think_trace = ""

        # Extract content inside the LAST <answer>...</answer>
        last_ans_open = generated_str.rfind("<answer>")
        last_ans_close = generated_str.rfind("</answer>")
        if last_ans_open != -1 and last_ans_close != -1 and last_ans_close > last_ans_open:
            generated_str = generated_str[last_ans_open + len("<answer>"):last_ans_close]

        return generated_str, think_trace

    def __call__(self, scenario: str, circumstance: str, circumstance_type: str, human_score: float, max_tokens: int = 4096) -> DyadicFrameWrapper:
        generation_query = self.make_query(scenario=scenario, circumstance=circumstance)

        if self.verbose:
            print(f"\n===== Generation query (last 500 chars) =====\n{generation_query[-500:]}")

        completion = self.client.chat.completions.create(
            model=self.engine,
            messages=[
                {
                    "role": "user",
                    "content": generation_query
                }
            ],
            temperature=0.0,
            max_tokens=max_tokens,
        )

        generated_str = completion.choices[0].message.content.strip()

        if self.verbose:
            print(f"\n===== Raw generated string =====\n{generated_str}")

        answer_str, think_trace = self.post_process_generation(generated_str)
        generation_json = self.extract_largest_json(answer_str)

        if self.verbose:
            print(f"\n===== Think trace =====\n{think_trace}")
            print(f"\n===== Parsed JSON =====\n{json.dumps(generation_json, indent=2)}")

        if not generation_json:
            print(f'Error in parsing json for scenario: {scenario}, circumstance: {circumstance}')
        scenario = scenario.replace('\n', '')
        return DyadicFrameWrapper(scenario=scenario, circumstance=circumstance, circumstance_type=circumstance_type, prompt=generation_query, generation_json=generation_json, human_score=human_score, think_trace=think_trace)


def do_inference(maker: DyadicFrameMaker, items: List[Tuple[str, str]], max_tokens: int = 4096, max_workers: int = 10) -> List[DyadicFrameWrapper]:
    outputs: List[DyadicFrameWrapper] = [None] * len(items)
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        future_to_idx = {}
        for i, item in enumerate(items):
            scenario = item[2]
            circumstance = item[3]
            circumstance_type = item[4]
            human_score = item[5]
            future = executor.submit(
                maker,
                scenario=scenario,
                circumstance=circumstance,
                circumstance_type=circumstance_type,
                human_score=human_score,
                max_tokens=max_tokens,
            )
            future_to_idx[future] = i

        for future in tqdm(as_completed(future_to_idx), total=len(items)):
            idx = future_to_idx[future]
            item = items[idx]
            try:
                outputs[idx] = future.result()
            except Exception as exc:
                print(f"Exception ({exc}) in input: {item}")
                outputs[idx] = DyadicFrameWrapper(
                    scenario=item[2],
                    circumstance=item[3],
                    circumstance_type=item[4],
                    prompt=f"Failed prompt with exception {exc}",
                    generation_json={},
                    human_score=item[5],
                    think_trace="",
                )
    return outputs


def read_new_llm_scenarios(df_path: str, all_circumstances: bool = False) -> List[Tuple[str, str]]:
    df = pd.read_csv(df_path, sep='\t')
    scenarios = df['id'].unique().tolist()
    items = []

    for id in scenarios:
        num_circumstances = 4 if all_circumstances else 1
        rows = df[df['id'] == id].iloc[: num_circumstances]
        for i, (_, row) in enumerate(rows.iterrows()):
            circumstance_type = 'orig' if i == 0 else f'C{i}'
            circumstance = row['circumstance'] if not pd.isna(row['circumstance']) else ''
            human_score = row['human_score_normalized [-4,4]']
            items.append((str(row['id']), row['dataset'], row['scenario'], circumstance, circumstance_type, human_score))
    return items


def save_to_file(outpath_jsonl, inputs: Tuple[str, str], outputs: List[DyadicFrameWrapper], append=False):
    sep = "\t"
    columns = ['id', 'dataset', 'scenario', 'circumstance_type', 'circumstance', 'vulnerable_score', 'intentional_score', 'harm_score', 'help_score', 'moral_score',
               'vulnerable_expl', 'intentional_expl', 'harm_expl', 'help_expl', 'moral_expl', 'agent', 'assumption_for_agent',
               'patient', 'assumption_for_patient', 'human_score', 'think_trace']

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
    parser.add_argument('--model', type=str, default='gpt-4o')
    parser.add_argument('--max-tokens', type=int, default=4096)
    parser.add_argument('--all-circumstances', action='store_true', default=False)
    parser.add_argument('--base-url', type=str, default=os.environ.get('OPENAI_BASE_URL', 'https://api.openai.com/v1'))
    parser.add_argument('--api-key', type=str, default=os.environ.get('OPENAI_API_KEY'))
    parser.add_argument('--verbose', action='store_true', default=False)
    parser.add_argument('--max-workers', type=int, default=10)
    parser.add_argument('--template-path', type=str, default='baseline_prompts/cot_few_shot_dyadic_template.txt')

    args = parser.parse_args()
    model_dir_name = args.model.replace('/', '_')

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
        "qwen3-thinking": "Qwen/Qwen3-235B-A22B-Thinking-2507",
    }

    engine = MODEL_CONFIG.get(args.model, args.model)

    data_file_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), '../data/moral_data_circumstances/dyadic/all_dyadic.tsv')
    items = read_new_llm_scenarios(data_file_path, all_circumstances=args.all_circumstances)

    client = OpenAI(
        base_url=args.base_url,
        api_key=args.api_key,
    )

    maker = DyadicFrameMaker(engine=engine, client=client, template_path=args.template_path, verbose=args.verbose)

    cfs = do_inference(items=items, maker=maker, max_tokens=args.max_tokens, max_workers=args.max_workers)
    os.makedirs(f'results/{model_dir_name}', exist_ok=True)
    save_to_file(outpath_jsonl=f'''results/{model_dir_name}/cot_few_shot_dyadic{'_all_circumstances' if args.all_circumstances else ''}.tsv''', inputs=items, outputs=cfs)
