import matplotlib
import matplotlib.pylab as plt

import IPython.display as ipd
from num2words import num2words
from unicodedata import normalize
import sys
sys.path.append('waveglow/')
import numpy as np
from vinorm import TTSnorm
from text import *
from hparams import create_hparams
from train import load_model

import re
from unicodedata import normalize
from flask import Flask, send_file
import pickle
from flask import Flask, render_template, request
import requests
import speech_recognition as sr
from uuid import uuid4
import torch
from model import Tacotron2
import os
#os.environ["CUDA_VISIBLE_DEVICES"]=""
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

hparams = create_hparams()
#hparams.sampling_rate = 22050
checkpoint_path = "checkpoint_338000"
model = load_model(hparams)
model.load_state_dict(torch.load(checkpoint_path)['state_dict'])
_ = model.cuda(device=device).eval().half()

waveglow_path = 'waveglow_256channels_universal_v5.pt'
waveglow = torch.load(waveglow_path)['model']
waveglow = waveglow.remove_weightnorm(waveglow)
waveglow.cuda(device=device).eval().half()
for k in waveglow.convinv:
    k.float()
from waveglow.denoiser import Denoiser
denoiser = Denoiser(waveglow)

def lowercase(text):
  return text.lower()
def collapse_whitespace(text):
    _whitespace_re = re.compile(r'\s+')
    return re.sub(_whitespace_re, ' ', text)
def TTS_norm(text):
    text = TTSnorm(text) 
    return text

def basic_cleaners(text):
    '''Basic pipeline that lowercases and collapses whitespace without transliteration.'''
    text = lowercase(text)
    text = collapse_whitespace(text)
    return text

def vi_num2words(num):
    return num2words(num, lang='vi')

def convert_time_to_text(time_string):
    #Support only hh:mm format
    try:
        h, m = time_string.split(":")
        time_string = vi_num2words(int(h)) + " giờ " + vi_num2words(int(m)) + " phút" 
        return time_string
    except:
        return None

def replace_time(text):
    # Define regex to time hh:mm
    result = re.findall(r'\d{1,2}:\d{1,2}|', text)
    match_list = list(filter(lambda x : len(x), result))

    for match in match_list:
        if convert_time_to_text(match):
            text = text.replace(match, convert_time_to_text(match))
    return text

def replace_number(text):
    return re.sub('(?P<id>\d+)', lambda m: vi_num2words(int(m.group('id'))), text)

def normalize_text(text):
    text = normalize("NFC", text)
    text = text.lower()
    text = replace_time(text)
    text = replace_number(text)
    return text

def preprocess_text(text):
    text=basic_cleaners(text)
    #text=TTS_norm(text)
    return text
def tts(text):    
    text = normalize("NFKC", text).lower()
    sequence = np.array(text_to_sequence(text, ['basic_cleaners']))[None, :]

    sequence = torch.autograd.Variable(
        torch.from_numpy(sequence)).cuda().long()

    with torch.no_grad():
        mel_outputs, mel_outputs_postnet, _, alignments = model.inference(sequence)
        audio = waveglow.infer(mel_outputs_postnet, sigma=0.666)
    audio_denoised = denoiser(audio, strength=0.01)[:, 0].cpu().numpy()
    return audio_denoised


def pts(para, filename):
    audio = np.zeros((1,0))
    para = para.replace('!', '.')
    para = para.replace('?', '.')
    para = para.replace(';', '.')
    para = para.replace(':', '.')
    sentence_ls = para.split(".")
    for sen in sentence_ls:
        sen = sen.replace('-', ',')
        sen = sen.replace('(', ',')
        sen = sen.replace(')', ',')
        sub_stn_ls = re.split(",", sen)
        for sub_stn in sub_stn_ls:
            audio = np.append(audio, tts(sub_stn), axis=1)
            audio = np.append(audio, np.zeros((1, int(hparams.sampling_rate/8)), dtype=np.uint8) , axis=1)
        audio = np.append(audio, np.zeros((1, int(hparams.sampling_rate/2)), dtype=np.uint8) , axis=1)

    a = ipd.Audio(audio, rate=hparams.sampling_rate)
    with open(f"generated_file/{filename}", "wb") as file:
        file.write(a.data)

filename = str(uuid4()) + '.wav'
# Khởi tạo Flask Server Backend
app = Flask(__name__)
# Apply Flask CORS

@app.route(f'/generated-file/{filename}')
def generateFile():
	try:
		return send_file(f"generated_file/{filename}", download_name=filename)
	except Exception as e:
		return str(e)

@app.route('/', methods=['GET', 'POST'])
def generate():
    if request.method == "POST":
        # Lấy file gửi lên
        text = request.json['text']
        
        text = preprocess_text(text)
        text = normalize_text(text)

        
        pts(text, filename)
    
        return {'status':200,'path':'/generated-file/'+ filename}
    
if __name__ == "__main__":
    app.run(host='0.0.0.0', debug=True)
