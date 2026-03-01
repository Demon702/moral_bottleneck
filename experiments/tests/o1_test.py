
from openai import OpenAI
import os
client = OpenAI(api_key=os.environ["OPENAI_UMASS_API_KEY"])

prompt = '''Consider a "scenario": {scenario}.

On a scale of -4 to 4, how morally acceptable is this scenario (-4 means extremely immoral, 4 means highly moral)?

Your output format should be:
{{
    "explanation": "... (maximum 20 words)",
    "moral acceptability score": "... (a number between -4 and 4)"
}}
'''

messages = [
    {
        "role": "user",
        "content": ""
    }
]

completion = client.chat.completions.create(
    model="o1",
    messages=messages
)

print(completion.choices[0].message.content)