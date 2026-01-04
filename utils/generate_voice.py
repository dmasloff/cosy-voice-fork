import sys
sys.path.append('third_party/Matcha-TTS')
from cosyvoice.cli.cosyvoice import AutoModel
import torchaudio

prompt_text = """Прошлый ролик по чит-кодам вам прям зашёл и это замотивировало меня ещё более усиленно копать интернет, чтобы найти для вас самые интересные и самые смешные чит-коды, которые встречались в видеоиграх."""

text = """<|ru|>Я могу вам сказать только одно: никто, никогда не вернётся в 2007 год. Потому что на дворе 2011 год."""
text1 = """<|ru|>Моя не реплика уже, а приговор. Реплики у вас, а все, что я говорю, — в граните отливается!"""
text2 = """<|ru|>Когда Маргарет Тэтчер привела к власти консерваторов в 1979 году, они были 18 лет у власти, никто же не говорил о том, что это противоречит какому-то демократическому тренду."""

cosyvoice = AutoModel(model_dir='pretrained_models/Fun-CosyVoice3-0.5B')

# for i, j in enumerate(cosyvoice.inference_zero_shot(text, prompt_text, '../buldjat_stripped/1/1_0.mp3')):
#         torchaudio.save('2007_{}.wav'.format(i + 1), j['tts_speech'], cosyvoice.sample_rate)

# for i, j in enumerate(cosyvoice.inference_zero_shot(text, prompt_text, './asset/trump_prompt.wav', stream=False)):
#     torchaudio.save('zero_shot_{}.wav'.format(i + 1), j['tts_speech'], cosyvoice.sample_rate)

for i, j in enumerate(cosyvoice.inference_cross_lingual(text2, './asset/trump_prompt.wav', stream=False)):
        torchaudio.save('2007_7.wav', j['tts_speech'], cosyvoice.sample_rate)