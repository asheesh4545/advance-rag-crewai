# sound_utils.py

import asyncio
import aiohttp
import base64
import pygame
import io
import json
import os 
from dotenv import load_dotenv


load_dotenv()
SARVAM_API_KEY = os.getenv('SARVAM_API_KEY')


async def text_to_speech(session, text):
    url = "https://api.sarvam.ai/text-to-speech"
    
    payload = {
        "inputs": [text],
        "target_language_code": "hi-IN",
        "speaker": "meera",
        "pitch": 0,
        "pace": 1,
        "loudness": 1.5,
        "speech_sample_rate": 16000,
        "enable_preprocessing": True,
        "model": "bulbul:v1"
    }
    
    headers = {
        "api-subscription-key": SARVAM_API_KEY,
        "Content-Type": "application/json"
    }
    
    async with session.post(url, json=payload, headers=headers) as response:
        return await response.json()

def play_audio(audio_data):
    audio_io = io.BytesIO(audio_data)
    pygame.mixer.music.load(audio_io)
    pygame.mixer.music.play()
    while pygame.mixer.music.get_busy():
        pygame.time.Clock().tick(10)

async def process_and_play_text(session, text):
    print(f"Processing: {text}")
    
    try:
        response = await text_to_speech(session, text)
        #print(f"API Response: {json.dumps(response, indent=2)}")
        
        if 'audios' in response and response['audios']:
            audio_base64 = response['audios'][0]
            audio_data = base64.b64decode(audio_base64)
            
            print(f"Playing audio")
            await asyncio.to_thread(play_audio, audio_data)
        else:
            print("Error: 'audios' key not found in the API response or it's empty.")
            print(f"Full API response: {response}")
    except Exception as e:
        print(f"An error occurred: {str(e)}")

def initialize_audio():
    pygame.mixer.init()