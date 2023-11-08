import torch
import torchaudio
import random
import os
import math
import numpy as np
import librosa
from scipy.signal import convolve

SAMPLE_RATE = 16000
bg_dir = '../../LibriVox_Kaggle/BGnoise/'
rir_dir = '../../RIR/MIT_IR_Survey/Audio/'

bg_files = os.listdir(bg_dir)
rir_files = os.listdir(rir_dir)[1:]


def get_seconds(audio):

    duration = (int)(audio.shape[0]/SAMPLE_RATE)
    audio_list = []
    
    for i in range(0, duration*SAMPLE_RATE, SAMPLE_RATE):
        audio_list.append(audio[i:i+SAMPLE_RATE])
    return audio_list

def round_up_audio(audio):
    
    rem = audio.shape[0]%SAMPLE_RATE
    zero_len = SAMPLE_RATE-rem
    added_arr = np.zeros(zero_len, audio.dtype)
    ext_audio = np.concatenate((audio, added_arr), axis=None)

    return ext_audio

def add_echo_from_file(filename, audio):

    rir_wav,sr_rir = librosa.load(filename, sr=SAMPLE_RATE)
    echo_audio = convolve(audio, rir_wav, mode='full')

    return echo_audio[0:SAMPLE_RATE]


def get_noise_from_sound(signal,noise,SNR):
    
    RMS_s=math.sqrt(np.mean(signal**2))
    #required RMS of noise
    RMS_n=math.sqrt(RMS_s**2/(pow(10,SNR/10)))
    
    #current RMS of noise
    RMS_n_current=math.sqrt(np.mean(noise**2))
    noise=noise*(RMS_n/RMS_n_current)
    
    return noise

def add_noise(audio, noise):

    SNR_list = [i for i in range(0,10)]
    SNR_choice = random.choice(SNR_list)
    
    noise = get_noise_from_sound(audio, noise, SNR_choice)
    noisy_audio = audio + noise
    return noisy_audio, noise

def get_random_audio_sec(audio_filename):
    
    # Choosing a random background and echo filename
    bg_file = bg_dir + random.choice(bg_files)
    rir_file = rir_dir + random.choice(rir_files)

    # Extracting audio data
    wav, sr = librosa.load(audio_filename, sr=SAMPLE_RATE)
    bg_wav,sr =librosa.load(bg_file, sr=SAMPLE_RATE)

    # Randomising and normalising audio data
    wav = round_up_audio(wav)
    wav /= np.max(np.abs(wav), axis=0)
    bg_wav /= np.max(np.abs(bg_wav), axis=0)

    # Getting a random audio and bg second
    wav_duration = (int)(wav.shape[0]/SAMPLE_RATE)
    sec_choice = random.choice([i for i in range(0, wav_duration-2)])
    rand_audio_sec = wav[sec_choice*SAMPLE_RATE:(sec_choice+2)*SAMPLE_RATE]
    bg_duration = (int)(bg_wav.shape[0]/SAMPLE_RATE)
    temp = [i for i in range(0, bg_duration-2)]
    bg_random_sec = random.choice(temp)
    bg_random_wav = bg_wav[bg_random_sec*SAMPLE_RATE:((bg_random_sec+2)*SAMPLE_RATE)]
    
    # Adding echo and bg noise to the audio
    #echo_audio = add_echo_from_file(rir_file, rand_audio_sec)
    #print(echo_audio.shape, bg_wav.shape)
    #noisy_audio, noise = add_noise(echo_audio, bg_random_wav)
    noisy_audio, noise = add_noise(rand_audio_sec, bg_random_wav)

    noisy_audio /= np.max(np.abs(noisy_audio), axis=0)
    noise /= np.max(np.abs(noise), axis=0)

    noisy_audio = torch.from_numpy(noisy_audio).unsqueeze(0)
    noise = torch.from_numpy(noise).unsqueeze(0)

    return noisy_audio, noise
