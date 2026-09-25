"""Voice agent: one persistent LFM2.5-Audio conversation that streams replies.

Used by server.py (over the WebSocket) and by `lfm_audio.py voice` (local mic/speakers).
"""
import time

import numpy as np
import torch

from lfm_audio import DEVICE, MIC_SR, ChatState, LFMModality, load, trim_pauses


class VoiceAgent:
    def __init__(self, system="Respond with interleaved text and audio."):
        self.processor, self.model = load()
        self.mimi = self.processor.mimi.eval()  # streaming decoder: one 8-code frame -> 80ms of 24kHz audio
        with torch.no_grad(), self.mimi.streaming(1):  # warm up, the first decode takes ~2.4s on mps
            for _ in range(5):
                self.mimi.decode(torch.randint(2048, (1, 8, 1), device=DEVICE))
        self.chat = ChatState(self.processor)
        self.chat.new_turn("system")
        self.chat.add_text(system)
        self.chat.end_turn()
        self.marks, self.last_text = {}, ""

    @torch.no_grad()
    def reply(self, audio=None, note=None):
        """Stream one assistant turn: yields str (text piece) and np.ndarray (float32 24kHz ~80ms chunk).

        audio: float32 16kHz mono user speech; note: extra text in the same user turn.
        The turn is kept in history even if the caller stops early. After it ends, self.marks holds
        perf_counter() times of "first text token", "first audio frame" and "generation done".
        """
        chat, proc = self.chat, self.processor
        chat.new_turn("user")
        if audio is not None:
            chat.add_audio(torch.from_numpy(audio).unsqueeze(0), MIC_SR)
        if note:
            chat.add_text(note)
        chat.end_turn()
        chat.new_turn("assistant")

        self.marks = {}
        text, codes, modality = [], [], []

        def mark(event):
            self.marks.setdefault(event, time.perf_counter())

        def pieces():
            with self.mimi.streaming(1):
                for t in self.model.generate_interleaved(**chat, max_new_tokens=512,
                                                         audio_temperature=1.0, audio_top_k=4):
                    if t.numel() == 1:  # text token: the model's script for the speech
                        mark("first text token")
                        text.append(t)
                        modality.append(LFMModality.TEXT)
                        if piece := proc.text.decode(t, skip_special_tokens=True):
                            yield piece
                    else:
                        mark("first audio frame")
                        codes.append(t)
                        modality.append(LFMModality.AUDIO_OUT)
                        if not (t == 2048).any():  # 2048 marks end of audio
                            yield self.mimi.decode(t[None, :, None])[0].float().cpu().numpy().ravel()
            mark("generation done")

        try:
            yield from trim_pauses(pieces())
        finally:  # keep the reply in history so the next turn has context (same as liquid_audio's demo)
            if modality:
                empty = torch.empty((8, 0), dtype=torch.long, device=proc.device)
                chat.append(text=torch.stack(text, 1) if text else empty[:1],
                            audio_out=torch.stack(codes, 1) if codes else empty,
                            modality_flag=torch.tensor(modality))
            chat.end_turn()
            self.last_text = proc.text.decode(torch.cat(text), skip_special_tokens=True) if text else ""
