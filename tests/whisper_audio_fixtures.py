"""Synthetic signal fixtures: no recorded user audio."""
import array
import math
import random
import sys
from types import SimpleNamespace


def pcm(seconds, *, amplitude=180, noise=False):
    rng = random.Random(719)
    values = array.array('h', (rng.randint(-amplitude, amplitude) if noise else
        round(amplitude * math.sin(2*math.pi*220*i/16000)) for i in range(round(seconds*16000))))
    if sys.byteorder != 'little':
        values.byteswap()
    return values.tobytes()


def transcriber(text='はい', segments=None):
    from server.openai_whisper import OpenAIWhisperTranscriber
    adapter = OpenAIWhisperTranscriber(api_key='synthetic-key', base_url='https://synthetic.invalid', model='synthetic-whisper')
    adapter.client.close()
    requests = []

    def create(**kwargs):
        requests.append(kwargs['file'].getvalue())
        return SimpleNamespace(text=text, segments=segments)

    client = SimpleNamespace(audio=SimpleNamespace(transcriptions=SimpleNamespace(create=create)), close=lambda: None)
    client.with_options = lambda **kwargs: client
    adapter.client = client
    adapter.multi_pass_enabled = False
    return adapter, requests
