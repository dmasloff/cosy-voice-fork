import os
import sys
import pathlib
sys.path.append('third_party/Matcha-TTS')

import torch
import torchaudio
from tqdm import trange

import onnxruntime as ort
from hyperpyyaml import load_hyperpyyaml
from cosyvoice.cli.cosyvoice import CosyVoice2, CosyVoice3
from cosyvoice.utils.file_utils import load_wav


model_dir = 'pretrained_models/CosyVoice2-0.5B'
model = CosyVoice2(model_dir)


def reconstruct_wav(wav: str) -> torch.Tensor:
    prompt_speech_token, _= model.frontend._extract_speech_token(wav)
    prompt_speech_feat, _ = model.frontend._extract_speech_feat(wav)
    embedding = model.frontend._extract_spk_embedding(wav)

    model.model.hift_cache_dict[0] = None
    model.model.vc_job(
        source_speech_token=prompt_speech_token,
        uuid=0
    )
    speech_token = torch.tensor(model.model.tts_speech_token_dict[0]).unsqueeze(dim=0)
    speech = model.model.token2wav(
        token=speech_token,
        prompt_token=prompt_speech_token,
        prompt_feat=prompt_speech_feat,
        embedding=embedding,
        token_offset=0,
        uuid=0,
        finalize=True
    )

    return speech

for i in trange(1, 245):
    directory = pathlib.Path(f'../buldjat_stripped/{i}')

    try:
        tracks = os.listdir(directory)
        for track in tracks:
            if not track.endswith('.mp3') or track.endswith('_r.mp3') or track.endswith('_r2.mp3'):
                continue

            reconstructed_audio = reconstruct_wav(directory / pathlib.Path(track))
            torchaudio.save(directory / pathlib.Path(track.rsplit('.mp3', 1)[0] + '_r2.mp3'), reconstructed_audio.cpu(), model.sample_rate)

    except FileNotFoundError:
        continue
