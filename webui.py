# Copyright (c) 2024 Alibaba Inc (authors: Xiang Lyu, Liu Yue)
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#   http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
import os
import sys
import argparse
import gradio as gr
import numpy as np
import torch
import torchaudio
import random
import librosa
ROOT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append('{}/third_party/Matcha-TTS'.format(ROOT_DIR))
from cosyvoice.cli.cosyvoice import CosyVoice, CosyVoice2
from cosyvoice.utils.file_utils import load_wav, logging
from cosyvoice.utils.common import set_all_random_seed

inference_mode_list = ['사전 학습 음색', '3초 초고속 복제', '다국어 복제', '자연어 제어']
instruct_dict = {'사전 학습 음색': '1. 사전 학습 음색 선택\n2. 오디오 생성 버튼 클릭',
                 '3초 초고속 복제': '1. prompt 오디오 파일 선택 또는 prompt 오디오 녹음, 30초 이하 주의, 동시 제공 시 prompt 오디오 파일 우선 선택\n2. prompt 텍스트 입력\n3. 오디오 생성 버튼 클릭',
                 '다국어 복제': '1. prompt 오디오 파일 선택 또는 prompt 오디오 녹음, 30초 이하 주의, 동시 제공 시 prompt 오디오 파일 우선 선택\n2. 오디오 생성 버튼 클릭',
                 '자연어 제어': '1. 사전 학습 음색 선택\n2. instruct 텍스트 입력\n3. 오디오 생성 버튼 클릭'}
stream_mode_list = [('아니오', False), ('예', True)]
max_val = 0.8


def generate_seed():
    seed = random.randint(1, 100000000)
    return {
        "__type__": "update",
        "value": seed
    }


def postprocess(speech, top_db=60, hop_length=220, win_length=440):
    speech, _ = librosa.effects.trim(
        speech, top_db=top_db,
        frame_length=win_length,
        hop_length=hop_length
    )
    if speech.abs().max() > max_val:
        speech = speech / speech.abs().max() * max_val
    speech = torch.concat([speech, torch.zeros(1, int(cosyvoice.sample_rate * 0.2))], dim=1)
    return speech


def change_instruction(mode_checkbox_group):
    return instruct_dict[mode_checkbox_group]


def generate_audio(tts_text, mode_checkbox_group, sft_dropdown, prompt_text, prompt_wav_upload, prompt_wav_record, instruct_text,
                   seed, stream, speed, temperature, n_timesteps):
    if prompt_wav_upload is not None:
        prompt_wav = prompt_wav_upload
    elif prompt_wav_record is not None:
        prompt_wav = prompt_wav_record
    else:
        prompt_wav = None
    # instruct 모드인 경우, 모델이 iic/CosyVoice-300M-Instruct이고 cross_lingual 모드가 아닌지 확인
    if mode_checkbox_group in ['자연어 제어']:
        if cosyvoice.instruct is False:
            gr.Warning('자연어 제어 모드를 사용 중입니다. {} 모델은 이 모드를 지원하지 않습니다. iic/CosyVoice-300M-Instruct 모델을 사용하세요.'.format(args.model_dir))
            yield (cosyvoice.sample_rate, default_data)
        if instruct_text == '':
            gr.Warning('자연어 제어 모드를 사용 중입니다. instruct 텍스트를 입력하세요.')
            yield (cosyvoice.sample_rate, default_data)
        if prompt_wav is not None or prompt_text != '':
            gr.Info('자연어 제어 모드를 사용 중입니다. prompt 오디오/prompt 텍스트는 무시됩니다.')
    # cross_lingual 모드인 경우, 모델이 iic/CosyVoice-300M이고 tts_text와 prompt_text가 다른 언어인지 확인
    if mode_checkbox_group in ['다국어 복제']:
        if cosyvoice.instruct is True:
            gr.Warning('다국어 복제 모드를 사용 중입니다. {} 모델은 이 모드를 지원하지 않습니다. iic/CosyVoice-300M 모델을 사용하세요.'.format(args.model_dir))
            yield (cosyvoice.sample_rate, default_data)
        if instruct_text != '':
            gr.Info('다국어 복제 모드를 사용 중입니다. instruct 텍스트는 무시됩니다.')
        if prompt_wav is None:
            gr.Warning('다국어 복제 모드를 사용 중입니다. prompt 오디오를 제공하세요.')
            yield (cosyvoice.sample_rate, default_data)
        gr.Info('다국어 복제 모드를 사용 중입니다. 합성 텍스트와 prompt 텍스트가 다른 언어인지 확인하세요.')
    # zero_shot cross_lingual인 경우, prompt_text와 prompt_wav가 요구사항을 만족하는지 확인
    if mode_checkbox_group in ['3초 초고속 복제', '다국어 복제']:
        if prompt_wav is None:
            gr.Warning('prompt 오디오가 비어있습니다. prompt 오디오를 입력하는 것을 잊으셨나요?')
            yield (cosyvoice.sample_rate, default_data)
        if torchaudio.info(prompt_wav).sample_rate < prompt_sr:
            gr.Warning('prompt 오디오 샘플링 레이트 {}가 {}보다 낮습니다.'.format(torchaudio.info(prompt_wav).sample_rate, prompt_sr))
            yield (cosyvoice.sample_rate, default_data)
    # sft 모드는 sft_dropdown만 사용
    if mode_checkbox_group in ['사전 학습 음색']:
        if instruct_text != '' or prompt_wav is not None or prompt_text != '':
            gr.Info('사전 학습 음색 모드를 사용 중입니다. prompt 텍스트/prompt 오디오/instruct 텍스트는 무시됩니다!')
        if sft_dropdown == '':
            gr.Warning('사용 가능한 사전 학습 음색이 없습니다!')
            yield (cosyvoice.sample_rate, default_data)
    # zero_shot 모드는 prompt_wav와 prompt 텍스트만 사용
    if mode_checkbox_group in ['3초 초고속 복제']:
        if prompt_text == '':
            gr.Warning('prompt 텍스트가 비어있습니다. prompt 텍스트를 입력하는 것을 잊으셨나요?')
            yield (cosyvoice.sample_rate, default_data)
        if instruct_text != '':
            gr.Info('3초 초고속 복제 모드를 사용 중입니다. 사전 학습 음색/instruct 텍스트는 무시됩니다!')

    if mode_checkbox_group == '사전 학습 음색':
        logging.info('get sft inference request')
        logging.info('temperature: {}, n_timesteps: {}, seed: {}'.format(temperature, n_timesteps, seed))
        set_all_random_seed(seed)
        for i in cosyvoice.inference_sft(tts_text, sft_dropdown, stream=stream, speed=speed, temperature=temperature, n_timesteps=n_timesteps):
            yield (cosyvoice.sample_rate, i['tts_speech'].numpy().flatten())
    elif mode_checkbox_group == '3초 초고속 복제':
        logging.info('get zero_shot inference request')
        logging.info('temperature: {}, n_timesteps: {}, seed: {}'.format(temperature, n_timesteps, seed))
        prompt_speech_16k = postprocess(load_wav(prompt_wav, prompt_sr))
        set_all_random_seed(seed)
        for i in cosyvoice.inference_zero_shot(tts_text, prompt_text, prompt_speech_16k, stream=stream, speed=speed, temperature=temperature, n_timesteps=n_timesteps):
            yield (cosyvoice.sample_rate, i['tts_speech'].numpy().flatten())
    elif mode_checkbox_group == '다국어 복제':
        logging.info('get cross_lingual inference request')
        prompt_speech_16k = postprocess(load_wav(prompt_wav, prompt_sr))
        set_all_random_seed(seed)
        for i in cosyvoice.inference_cross_lingual(tts_text, prompt_speech_16k, stream=stream, speed=speed, temperature=temperature, n_timesteps=n_timesteps):
            yield (cosyvoice.sample_rate, i['tts_speech'].numpy().flatten())
    else:
        logging.info('get instruct inference request')
        set_all_random_seed(seed)
        for i in cosyvoice.inference_instruct(tts_text, sft_dropdown, instruct_text, stream=stream, speed=speed, temperature=temperature, n_timesteps=n_timesteps):
            yield (cosyvoice.sample_rate, i['tts_speech'].numpy().flatten())


def main():
    with gr.Blocks() as demo:
        gr.Markdown("### 코드 저장소 [CosyVoice](https://github.com/FunAudioLLM/CosyVoice) \
                    사전 학습 모델 [CosyVoice-300M](https://www.modelscope.cn/models/iic/CosyVoice-300M) \
                    [CosyVoice-300M-Instruct](https://www.modelscope.cn/models/iic/CosyVoice-300M-Instruct) \
                    [CosyVoice-300M-SFT](https://www.modelscope.cn/models/iic/CosyVoice-300M-SFT)")
        gr.Markdown("#### 합성할 텍스트를 입력하고, 추론 모드를 선택한 후 안내 단계에 따라 작업하세요.")

        tts_text = gr.Textbox(label="합성 텍스트 입력", lines=1, value="<|ko|>저는 통의 연구소 음성 팀이 새롭게 출시한 생성형 음성 대형 모델로, 편안하고 자연스러운 음성 합성 능력을 제공합니다.", placeholder="한국어: <|ko|>텍스트, 중국어: <|zh|>텍스트, 영어: <|en|>텍스트")
        with gr.Row():
            mode_checkbox_group = gr.Radio(choices=inference_mode_list, label='추론 모드 선택', value=inference_mode_list[0])
            instruction_text = gr.Text(label="작업 단계", value=instruct_dict[inference_mode_list[0]], scale=0.5)
            sft_dropdown = gr.Dropdown(choices=sft_spk, label='사전 학습 음색 선택', value=sft_spk[0], scale=0.25)
            stream = gr.Radio(choices=stream_mode_list, label='스트리밍 추론 여부', value=stream_mode_list[0][1])
            speed = gr.Number(value=1, label="속도 조절(비스트리밍 추론만 지원)", minimum=0.5, maximum=2.0, step=0.1)
        with gr.Row():
            with gr.Column(scale=0.3):
                seed_button = gr.Button(value="\U0001F3B2", size="sm")
                seed = gr.Number(value=0, label="랜덤 추론 시드 (0=랜덤, 같은 시드=같은 결과)")
            with gr.Column(scale=0.35):
                temperature = gr.Number(value=0.85, label="Temperature (0.7-0.9: 자연스러움, 1.0: 기본값, 권장: 0.3-2.0)", minimum=0.1, maximum=10.0, step=0.05)
            with gr.Column(scale=0.35):
                n_timesteps = gr.Number(value=15, label="디퓨전 스텝 수 (10: 빠름, 15-20: 고품질, 권장: 10-30)", minimum=1, maximum=100, step=1)

        with gr.Row():
            prompt_wav_upload = gr.Audio(sources='upload', type='filepath', label='prompt 오디오 파일 선택, 샘플링 레이트는 16khz 이상이어야 합니다')
            prompt_wav_record = gr.Audio(sources='microphone', type='filepath', label='prompt 오디오 파일 녹음')
        prompt_text = gr.Textbox(label="prompt 텍스트 입력", lines=1, placeholder="prompt 텍스트를 입력하세요. prompt 오디오 내용과 일치해야 하며, 자동 인식은 아직 지원하지 않습니다...", value='오늘은 날씨가 좋아서 산책하기에 딱 좋은 하루입니다. 새로운 기술을 배우는 과정은 때로 어렵지만 매우 보람 있습니다. 따뜻한 커피 한 잔이 하루를 조금 더 특별하게 만들어 줍니다.')
        instruct_text = gr.Textbox(label="instruct 텍스트 입력", lines=1, placeholder="instruct 텍스트를 입력하세요.", value='')

        generate_button = gr.Button("오디오 생성")

        audio_output = gr.Audio(label="합성 오디오", autoplay=True, streaming=True)

        seed_button.click(generate_seed, inputs=[], outputs=seed)
        generate_button.click(generate_audio,
                              inputs=[tts_text, mode_checkbox_group, sft_dropdown, prompt_text, prompt_wav_upload, prompt_wav_record, instruct_text,
                                      seed, stream, speed, temperature, n_timesteps],
                              outputs=[audio_output])
        mode_checkbox_group.change(fn=change_instruction, inputs=[mode_checkbox_group], outputs=[instruction_text])
    demo.queue(max_size=4, default_concurrency_limit=2)
    demo.launch(server_name='0.0.0.0', server_port=args.port)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--port',
                        type=int,
                        default=8000)
    parser.add_argument('--model_dir',
                        type=str,
                        # 보이스 클로닝이 필요한 모델 
                        default='pretrained_models/CosyVoice2-0.5B',
                        help='local path or modelscope repo id'

                        # 사전 학습 음성 제공되는데 부자연스러움 반응이랑 그런거
                        # default='pretrained_models/CosyVoice-300M-SFT',
                        # help='local path or modelscope repo id (CosyVoice-300M-SFT for SFT voices, CosyVoice2-0.5B for zero-shot)'
                        )
    args = parser.parse_args()
    try:
        cosyvoice = CosyVoice(args.model_dir)
    except Exception:
        try:
            cosyvoice = CosyVoice2(args.model_dir)
        except Exception:
            raise TypeError('no valid model_type!')

    sft_spk = cosyvoice.list_available_spks()
    if len(sft_spk) == 0:
        sft_spk = ['']
    prompt_sr = 16000
    default_data = np.zeros(cosyvoice.sample_rate)
    main()
