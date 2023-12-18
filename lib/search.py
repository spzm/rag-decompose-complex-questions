import json
import numpy
import os
import requests

from openai import OpenAI
from dotenv import load_dotenv

load_dotenv()


def do_search(query):
    data = json.dumps({
        "q": query,
        "num": 10
    })

    headers = {
        'X-API-KEY': os.getenv('SERPER_API'),
        'Content-Type': 'application/json'
    }

    response = requests.post('https://google.serper.dev/search', headers=headers, data=data)
    response.raise_for_status()  # Raises a HTTPError if the HTTP request returned an unsuccessful status code
    return response.json()


def cosine_similarity(a, b):
    return numpy.dot(a, b) / (numpy.linalg.norm(a) * numpy.linalg.norm(b))


def get_embedding(text, model="text-embedding-ada-002"):
    openai = OpenAI(api_key=os.getenv('OPENAI_API_KEY'))
    text = text.replace("\n", " ")
    return openai.embeddings.create(input=[text], model=model).data[0].embedding
