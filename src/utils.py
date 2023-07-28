# language model imports
import re
import torch
from transformers import pipeline
import networkx as nx
import openai



language_model = pipeline(model="declare-lab/flan-alpaca-xl", device="cuda:0")

key_path = "openai_api_key"
#openai.organization = "org-D3T7qkglEsZGgYNCoTz3Uocx"
with open(key_path, "r") as f:
    api_key = f.readline()
openai.api_key = api_key



def generate_text(prompt_background, prompt_text, use_openai = False):
    if use_openai:
        print("using openai gpt-3.5-turbo")
        completion = openai.ChatCompletion.create(
            model="gpt-3.5-turbo",
            messages=[
            {"role": "system", "content": prompt_background},
            {"role": "user", "content": prompt_text}
            ]
            )

        return completion.choices[0].message['content']
    else:
        # merge background and question for simple model
        prompt = prompt_background + prompt_text
        return language_model(prompt, do_sample=True, min_length=10, max_length=len(prompt)+128)[0]["generated_text"]
    
    

def get_rating(rating_str):
    """
    extracts a rating from a string.
    
    rating_str: The string to extract the rating from.
    
    returns:
    int: The rating extracted from the string, or None if no rating is found.
    """
    nums = [int(i) for i in re.findall(r'(?<![A-Za-z0-9.])[0-9](?![0-9:])', rating_str)]
    
    if len(nums)>0:
        return min(nums)
    else:
        return None