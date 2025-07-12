# apis/gpt_timestamp_parser.py

from openai import OpenAI
import os
import json

client = OpenAI()

def extract_timestamps(scraped_text: str) -> list:
    """
    Sends scraped fight description text to GPT and returns list of timestamp dicts:
    [
        {
            "episode": 163,
            "start_time": "07:15",
            "end_time": "end"
        },
        ...
    ]
    """
    
    system_prompt = """
    You are an assistant that extracts structured timestamp data from text about anime fights.

    Input: messy text describing episodes and timestamps where a fight happens.

    Your goal:
    - Identify each episode number involved in the fight.
    - Extract start and end times if available.
      - If times are missing, use "start" or "end" as placeholders.

    Output ONLY JSON as an array of objects:
    [
        {
            "episode": EPISODE_NUMBER,
            "start_time": "HH:MM" or "start",
            "end_time": "HH:MM" or "end"
        }
    ]

    Do not include any explanation or text — output only JSON.
    """

    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": scraped_text}
    ]

    completion = client.chat.completions.create(
        model="gpt-4-turbo",
        messages=messages,
        temperature=0
    )

    response_text = completion.choices[0].message.content

    try:
        result = json.loads(response_text)
        return result
    except json.JSONDecodeError:
        return []
