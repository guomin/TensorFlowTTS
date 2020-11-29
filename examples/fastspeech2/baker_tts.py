import os
import sys
import tensorflow as tf
import yaml
import numpy as np
import matplotlib.pyplot as plt
import IPython.display as ipd
import scipy
import scipy.io.wavfile

from tensorflow_tts.inference import AutoConfig
from tensorflow_tts.inference import TFAutoModel
from tensorflow_tts.inference import AutoProcessor

# 来源：https://colab.research.google.com/drive/1YpSHRBRPBI7cnTkQn1UcVTWEQVbsUm1S?usp=sharing#scrollTo=dXC1c-0qZyg3

tacotron2_config = AutoConfig.from_pretrained('../tacotron2/conf/tacotron2.baker.v1.yaml')
tacotron2 = TFAutoModel.from_pretrained(
    config=tacotron2_config,
    pretrained_path='models/tacotron2_gm_20k.h5',
    training=False,
    name='tacotron2'
)

fs2_model_path = 'models/fastspeech2_gm_140k.h5'
fs2_model_path = 'models/fastspeech2-200k.h5'
fs2_model_path = 'models/fastspeech2_gm_200k.h5'
fastspeech2_config = AutoConfig.from_pretrained('conf/fastspeech2.baker.v2.yaml')
fastspeech2 = TFAutoModel.from_pretrained(
    config=fastspeech2_config,
    pretrained_path=fs2_model_path,
    name='fastspeech2'
)

mb_melgan_config = AutoConfig.from_pretrained(
    '../multiband_melgan/conf/multiband_melgan.baker.v1.yaml')
mb_melgan = TFAutoModel.from_pretrained(
    config=mb_melgan_config,
    pretrained_path='models/mb.melgan-920k.h5',
    name='mb_melgan'
)
processor = AutoProcessor.from_pretrained(pretrained_path="./models/baker_mapper.json")


def do_synthesis(input_text, text2mel_model, vocoder_model, text2mel_name, vocoder_name, speed_r=1, f0_r=1, energy_r=1):
    input_ids = processor.text_to_sequence(input_text, inference=True)

    # text2mel part
    if text2mel_name == "TACOTRON":
        _, mel_outputs, stop_token_prediction, alignment_history = text2mel_model.inference(
            tf.expand_dims(tf.convert_to_tensor(input_ids, dtype=tf.int32), 0),
            tf.convert_to_tensor([len(input_ids)], tf.int32),
            tf.convert_to_tensor([0], dtype=tf.int32)
        )
    elif text2mel_name == "FASTSPEECH2":
        mel_before, mel_outputs, duration_outputs, _, _ = text2mel_model.inference(
            tf.expand_dims(tf.convert_to_tensor(input_ids, dtype=tf.int32), 0),
            speaker_ids=tf.convert_to_tensor([0], dtype=tf.int32),
            speed_ratios=tf.convert_to_tensor([speed_r], dtype=tf.float32),
            f0_ratios=tf.convert_to_tensor([f0_r], dtype=tf.float32),
            energy_ratios=tf.convert_to_tensor([energy_r], dtype=tf.float32),
        )
    else:
        raise ValueError("Only TACOTRON, FASTSPEECH2 are supported on text2mel_name")

    # vocoder part
    if vocoder_name == "MB-MELGAN":
        # tacotron-2 generate noise in the end symtematic, let remove it :v.
        if text2mel_name == "TACOTRON":
            # remove_end = 1024
            remove_end = int(8192*2.5)
        else:
            remove_end = 1
        print(len(mel_outputs[0]))
        audio = vocoder_model.inference(mel_outputs)[0, :-remove_end, 0]
    else:
        raise ValueError("Only MB_MELGAN are supported on vocoder_name")

    if text2mel_name == "TACOTRON":
        return mel_outputs.numpy(), alignment_history.numpy(), audio.numpy()
    else:
        return mel_outputs.numpy(), audio.numpy()


def visualize_attention(alignment_history):
    import matplotlib.pyplot as plt

    fig = plt.figure(figsize=(8, 6))
    ax = fig.add_subplot(111)
    ax.set_title(f'Alignment steps')
    im = ax.imshow(
        alignment_history,
        aspect='auto',
        origin='lower',
        interpolation='none')
    fig.colorbar(im, ax=ax)
    xlabel = 'Decoder timestep'
    plt.xlabel(xlabel)
    plt.ylabel('Encoder timestep')
    plt.tight_layout()
    plt.show()
    plt.close()


def visualize_mel_spectrogram(mels):
    mels = tf.reshape(mels, [-1, 80]).numpy()
    fig = plt.figure(figsize=(10, 8))
    ax1 = fig.add_subplot(311)
    ax1.set_title(f'Predicted Mel-after-Spectrogram')
    im = ax1.imshow(np.rot90(mels), aspect='auto', interpolation='none')
    fig.colorbar(mappable=im, shrink=0.65, orientation='horizontal', ax=ax1)
    plt.show()
    plt.close()


# input_text = "这是一个开源的端到端中文语音合成系统"
input_texts = [
    '通过研究生院审核并组织专家评审',
    '旧水',
    '理科楼',
    '谢谢',
    '再见',
    '对不起',
    '请拿好杯子'
]
for input_text in input_texts:
    speed_r = 0.7
    f0_r = 1
    energy_r = 1
    mels, audios = do_synthesis(input_text, fastspeech2, mb_melgan, "FASTSPEECH2", "MB-MELGAN", speed_r=speed_r, f0_r=f0_r, energy_r=energy_r)
    # visualize_mel_spectrogram(mels[0])
    # ipd.Audio(audios, rate=24000, autoplay=True)
    scipy.io.wavfile.write(os.path.join('results', input_text + '_' + str(speed_r) + '_' + str(f0_r) + "_" + '.wav'), 24000, audios)
    # with open('test.pcm', 'wb') as fout:
    #     fout.write(audios)

# for input_text in input_texts:
#     mels, alignment_history, audios = do_synthesis(input_text, tacotron2, mb_melgan, "TACOTRON", "MB-MELGAN")
#     visualize_attention(alignment_history[0])
#     visualize_mel_spectrogram(mels[0])
#     scipy.io.wavfile.write(input_text + "-tacotron2.wav", 24000, audios)
