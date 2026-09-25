import tempfile
import time

import gradio as gr
import spaces

import lfm_audio

lfm_audio.load()  # load weights once at startup; ZeroGPU attaches the GPU per call


def tmp_wav():
    return tempfile.NamedTemporaryFile(suffix=".wav", delete=False).name


@spaces.GPU
def tts(text, voice):
    return lfm_audio.run(f"Perform TTS. Use the {voice} voice.", user_text=text, out=tmp_wav(),
                         audio_temperature=0.8, audio_top_k=64)[1]


@spaces.GPU
def asr(wav):
    return lfm_audio.run("Perform ASR.", wav_path=wav)[0]


def chat(wav):
    recv = time.time()
    out, t = gpu_chat(wav)
    t["recv"] = recv
    print("TIMING " + " ".join(f"{k}={v:.3f}" for k, v in t.items()), flush=True)  # read by talk_space.py
    return out


@spaces.GPU
def gpu_chat(wav):
    gpu = time.time()
    _, out, t = lfm_audio.run("Respond with interleaved text and audio.", wav_path=wav, out=tmp_wav(),
                                  mode="interleaved", audio_temperature=1.0, audio_top_k=4)
    t["first_audio"] = t["first_audio"] or time.time()  # no audio: treat as "at the end"
    return out, {**t, "gpu": gpu, "done": time.time()}


mic = lambda: gr.Audio(sources=["microphone", "upload"], type="filepath")
gr.TabbedInterface([
    gr.Interface(tts, [gr.Textbox(label="Text"),
                       gr.Dropdown(["US female", "US male", "UK female", "UK male"], value="US female")],
                 gr.Audio()),
    gr.Interface(asr, mic(), gr.Textbox()),
    gr.Interface(chat, mic(), gr.Audio()),
], ["TTS", "ASR", "Voice chat"], title="LFM2.5-Audio-1.5B").launch()
