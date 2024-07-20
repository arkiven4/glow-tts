import torch
from text import text_to_sequence, cmudict, cleaned_text_to_sequence
from text.symbols import symbols
import commons
import models
import json
import sys
from collections import defaultdict
import gradio as gr
from gradio import components
import os
import torch
import utils
import numpy as np
from scipy.io.wavfile import write
import random
import time


from BigVGAN_.env import AttrDict
from BigVGAN_.meldataset import MAX_WAV_VALUE
from BigVGAN_.models import BigVGAN as Generator

def load_checkpoint(filepath, device):
    assert os.path.isfile(filepath)
    print("Loading '{}'".format(filepath))
    checkpoint_dict = torch.load(filepath, smap_location=device)
    print("Complete.")
    return checkpoint_dict

language_dict = {
    'English': 0,
    'Indonesian': 2,
    'Japanese': 3
}

sys.path.append('./BigVGAN_/')
device = torch.device('cuda')

#MasterArray_Emotion = defaultdict((lambda: defaultdict(list)))
MasterArray_Emotion = defaultdict(list)
MasterArray_Speaker = defaultdict(list)
hps = None
model = None
generator = None

database_index = 8
root_database =  "/run/media/arkiven4/Other/Kuliah/Thesis/Datasets/dataset_name"

def get_text(text, hps):
    text_norm = text_to_sequence(text, hps.data.text_cleaners)
    if hps.data.add_blank:
        text_norm = commons.intersperse(text_norm, 0)
    text_norm = torch.LongTensor(text_norm)
    return text_norm

def tts(language, emotion, speaker, pitch_intensity, pitch_noise_scale, energy_intensity, energy_noise_scale, duration_scale, text_input):
    tst_stn = text_input
    language = language_dict[language]

    text_norm = text_to_sequence(tst_stn.strip(), [hps.data.text_cleaners[int(language)]])
    text_norm = commons.intersperse(text_norm, len(symbols))
    sequence = np.array(text_norm)[None, :]
    x_tst = torch.autograd.Variable(torch.from_numpy(sequence)).to(device).long()
    x_tst_lengths = torch.tensor([x_tst.shape[1]]).to(device)   

    # MasterArray_Emotion[str(language)][emotion]
    emo_filepick = random.choice(MasterArray_Emotion[emotion])
    emo_database_name = emo_filepick.split("/")[database_index]
    emo_filename = emo_filepick.split("/")[-1].split(".")[0]

    print(emo_filepick)
    print(emo_database_name)
    print(emo_filename)
    
    spk_filepick = random.choice(MasterArray_Speaker[speaker])
    spk_database_name = spk_filepick.split("/")[database_index]
    spk_filename = spk_filepick.split("/")[-1].split(".")[0]
    
    spk_emb_src = torch.Tensor(np.load(f"{root_database.replace('dataset_name', spk_database_name)}/spk_embeds/{spk_filename}.npy")).reshape(1,-1).to(device)
    emo_emb = torch.Tensor(np.load(f"{root_database.replace('dataset_name', emo_database_name)}/emo_embeds/{emo_filename}.npy")).reshape(1,-1).to(device)

    lid = torch.IntTensor([int(language)]).to(device)

    with torch.no_grad():
        noise_scale = .667
        noise_scale_w = 0.8
        (y_gen_tst, *_), *_, (attn_gen, *_) = model.infer(x_tst, x_tst_lengths, y=None, y_lengths=None, g=spk_emb_src, emo=emo_emb, l=lid, noise_scale=noise_scale, noise_scale_w=noise_scale_w, f0_noise_scale=pitch_noise_scale, energy_noise_scale=energy_noise_scale, length_scale=duration_scale, pitch_scale=pitch_intensity, energy_scale=energy_intensity)

    with torch.no_grad():
        x = y_gen_tst.cpu().detach().numpy()
        x = torch.FloatTensor(x).to(device)
        y_g_hat = generator(x)
        audio = y_g_hat.squeeze()
        audio = audio * MAX_WAV_VALUE
        audio = audio.cpu().numpy().astype('int16')

    
    output_wav_path = f"webui_generated/{int(time.time())}.wav"
    write(output_wav_path, hps.data.sampling_rate, audio)

    return output_wav_path

if __name__ == "__main__":
    # Generator
    config_file = os.path.join("BigVGAN_/cp_model", 'config.json')
    with open(config_file) as f:
        data = f.read()
    json_config = json.loads(data)
    h = AttrDict(json_config)
    generator = Generator(h).to(device)
    state_dict_g = torch.load("BigVGAN_/cp_model/g_05000000", map_location="cpu")
    generator.load_state_dict(state_dict_g['generator'])
    generator.eval()
    generator.remove_weight_norm()

    # Model
    model_dir = "logs/base_blank_emo_lang_pitch/"
    hps = utils.get_hparams_from_dir(model_dir)
    checkpoint_path = utils.latest_checkpoint_path(model_dir)
    model = models.FlowGenerator(
        len(symbols) + getattr(hps.data, "add_blank", False),
        out_channels=hps.data.n_mel_channels,
        n_lang=hps.data.n_lang,
        **hps.model).to(device)
    utils.load_checkpoint(checkpoint_path, model)
    model.decoder.store_inverse() # do not calcuate jacobians for fast decoding
    _ = model.eval()

    # Data
    with open("filelists/runpod_test_filelist.txt", "r", encoding="utf-8") as txt_file:
        lines = txt_file.readlines()

    for line in lines:
        line_metadata = line.rstrip().split("|")
        path_audio = line_metadata[0]
        language = line_metadata[1]

        if int(language) == 0:
            emotion = path_audio.split("/")[11]
            #MasterArray_Emotion[language][emotion].append(path_audio)
            MasterArray_Emotion[emotion].append(path_audio)

            speaker = path_audio.split("/")[-1].split("_")[0]
            MasterArray_Speaker[speaker].append(path_audio)
        elif int(language) == 2:
            emotion = path_audio.split("/")[10]
            #MasterArray_Emotion[language][emotion].append(path_audio)
            MasterArray_Emotion[emotion].append(path_audio)

            speaker = path_audio.split("/")[-1].split("_")[1]
            MasterArray_Speaker[speaker].append(path_audio)
        else:
            speaker = path_audio.split("/")[-1].split("_")[0]
            MasterArray_Speaker[speaker].append(path_audio)

    MasterArray_Emotion = dict(MasterArray_Emotion)
    MasterArray_Emotion = json.loads(json.dumps(MasterArray_Emotion))

    MasterArray_Speaker = dict(MasterArray_Speaker)
    MasterArray_Speaker = json.loads(json.dumps(MasterArray_Speaker))

    def onLanguangeChange(language):
        if language == 'English':
            return gr.Dropdown(choices=MasterArray_Emotion['0'].keys(), interactive=True)
        elif language == 'Indonesian':
            return gr.Dropdown(choices=MasterArray_Emotion['2'].keys(), interactive=True)
        else:
            return gr.Dropdown(choices=MasterArray_Emotion['0'].keys(), interactive=True)
        

    # [y for x in [MasterArray_Emotion['0'].keys(), MasterArray_Emotion['2'].keys()] for y in x]

    inputs_gr = [
        components.Dropdown(language_dict.keys(), value='English', label="Language", info="Will add more animals later!"), 
        components.Dropdown(MasterArray_Emotion.keys(), value='', label="Emotion"), 
        components.Dropdown(MasterArray_Speaker.keys(), value='', label="Speaker"), 
        components.Slider(0.1, 10, step=0.2, value=1, label="Pitch Intensity"), 
        components.Slider(0.1, 1, step=0.1, value=0.667, label="Pitch Noise Scale"), 
        components.Slider(0.1, 10, step=0.2, value=1, label="Energy Intensity"), 
        components.Slider(0.1, 1, step=0.1, value=0.667, label="Energy Noise Scale"), 
        components.Slider(0.1, 1.8, step=0.1, value=1, label="Duration Scale"), 
        components.Textbox(label="Text Input")]

    with gr.Blocks() as app:
        gr.Interface( 
            fn=tts,
            inputs=inputs_gr,
            outputs=components.Audio(type='filepath', label="Generated Speech"),
            examples=[
                ['English', "Angry", "fyat", 3.5, 0.5, 2.9, 0.8, 0.9, "Don't Sit Under That Tree, It is Dangerous"],
                ['Indonesian', "Sedih", "fena", 5.7, 0.9, 1.1, 0.9, 1.5, "Film Itu Sedih Sekali."],
                ['Indonesian', "Marah", "fyat", 6.5, 0.8, 2.9, 0.8, 0.9, "Dilarang Merokok Di Tempat ini."],
            ],
            live=False
        )

        #inputs_gr[0].select(fn=onLanguangeChange, inputs=inputs_gr[0], outputs=inputs_gr[1])

    app.launch()
    


