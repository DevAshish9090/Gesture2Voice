import os
from gtts import gTTS
import playsound

def build_sentence(word_list):
    if len(word_list) == 0:
        return ""
    cleaned = [w.replace('_', ' ') for w in word_list]
    sentence = " ".join(cleaned)
    sentence = sentence.capitalize() + "."
    return sentence

def speak(sentence, voice='en'):
    if not sentence or sentence == ".":
        return
    print(f"Speaking: {sentence}")
    tts = gTTS(text=sentence, lang='en', tld=get_tld(voice))
    audio_path = "output.mp3"
    tts.save(audio_path)
    playsound.playsound(audio_path)
    os.remove(audio_path)

def get_tld(voice):
    voices = {
        'en-us': 'com',
        'en-gb': 'co.uk',
        'en-in': 'co.in',
        'en-au': 'com.au'
    }
    return voices.get(voice, 'com')