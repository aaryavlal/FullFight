# apis/gpt_fight_parser.py

from openai import OpenAI
import os
import json

client = OpenAI()

def parse_fight_request(user_input: str) -> dict:
    """
    Sends user fight request to GPT and returns structured JSON:
    {
        "anime_name": "Attack on Titan",
        "fighter_names": ["Levi", "Beast Titan"]
    }
    """
    
    system_prompt = """
    You are an assistant that extracts structured information from user requests about anime fights.

    Given a user’s message, identify:
    - The anime series name
    - The fighter names

    Output ONLY JSON:
    {
        "anime_name": "...",
        "fighter_names": ["...", "..."]
    }

    If the anime is not mentioned, return "anime_name": null.
    If fighter names are not mentioned, return an empty array [].
    Do not include any explanation — output only JSON.
    """

    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_input}
    ]

    completion = client.chat.completions.create(
        model="gpt-3.5-turbo",
        messages=messages,
        temperature=0
    )

    # GPT returns a JSON string
    response_text = completion.choices[0].message.content

    try:
        result = json.loads(response_text)
        return result
    except json.JSONDecodeError:
        return {
            "anime_name": None,
            "fighter_names": []
        }
