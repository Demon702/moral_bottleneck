
import pandas as pd
import numpy as np
from openai import OpenAI
import os

THEORIES = [
    "end_to_end",
    "deontology",
    "dyadic",
    "cot",
    "mft_two_step",
    "mft",
    "morality_as_cooperation",
    "utilitarianism",
    "virtue_ethics"
]

MODELS = [
    # "gpt-3.5-turbo",
    # "gpt-4o",
    # "meta-llama/Llama-3-8b-chat-hf",
    # "meta-llama/Llama-3-70b-chat-hf",
    "mistralai/Mixtral-8x7B-Instruct-v0.1",
    "mistralai/Mixtral-8x22B-Instruct-v0.1"
]



SYSTEM_PROMPT = """You are a moral AI."""
PROMPT = """
You will be presented with a series of scenarios. You have to identify what's common between the scenarios. Here are the scenarios:
{scenarios}

Whenver you mention a common theme among a group of scenarios, add one or two examples from that group.
Your answer should contain a paragraph each for the common themes identified between the scenarios.
"""

def get_common_themes(scenarios):

    client = OpenAI(
        api_key=os.environ["OPENAI_PERSONAL_API_KEY"]
    )

    scenario_prompt = PROMPT.format(scenarios="\n - ".join(scenarios))
    completion = client.chat.completions.create(
        model="gpt-4o",
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {
                "role": "user",
                "content": scenario_prompt
            }
        ],
        temperature=0.0,
        max_tokens=1500,
        seed=0
    )

    return completion.choices[0].message.content



for model in MODELS:
    for theory in THEORIES:

        print(f"\n\n\n\nProcessing {theory} theory for {model} model")

        model_dir_name = model.replace("/", "_")

        if not os.path.exists(f"../results/{model_dir_name}/{theory}.tsv"):
            print(f"Skipping {model} {theory} as the file does not exist")
            continue
        data = pd.read_csv(f"../results/{model_dir_name}/{theory}.tsv", sep="\t")

        # Extract the test data
        if len(data) > 550:
            data = data[147:]

        print(f"Total number of scenarios: {len(data)}")

        if len(data) < 450:
            raise Exception(f"Error in {model} {theory} as it has less than 500 scenarios")


        data["squared_error"] = (data["human_score"] - data["moral_score"])  ** 2

        good_scenarios = data[data["squared_error"] < 1.0]["scenario"].tolist()
        mid_scenarios = data[(data["squared_error"] >= 1.0) & (data["squared_error"] < 4.0)]["scenario"].tolist()
        bad_scenarios = data[data["squared_error"] >= 4.0]["scenario"].tolist()


        os.makedirs(f"{model_dir_name}/{theory}", exist_ok=True)

        print("\n" * 5)
        print(f"No of good scenarios: {len(good_scenarios)}")
        print("Getting common themes for good scenarios...")

        with open(f"{model_dir_name}/{theory}/good_scenarios.txt", "w") as f:
            f.write(get_common_themes(good_scenarios))


        print("\n" * 5)
        print(f"No of mid scenarios: {len(mid_scenarios)}")
        print("Getting common themes for mid scenarios...")


        with open(f"{model_dir_name}/{theory}/mid_scenarios.txt", "w") as f:
            f.write(get_common_themes(mid_scenarios))


        print("\n" * 5)

        print(f"No of bad scenarios: {len(mid_scenarios)}")
        print("Getting common themes for bad scenarios...")


        print("\n" * 5)

        with open(f"{model_dir_name}/{theory}/bad_scenarios.txt", "w") as f:
            f.write(get_common_themes(bad_scenarios))

