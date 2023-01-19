import re
import os
import os.path as osp
import sys
import glob
import json
import tqdm
from typing import List

import cv2
import librosa
import soundfile
import skvideo.io
import numpy as np
from numpy import ndarray

from vocabulary.utils import char2grp


def preprocess_source_data(
        root_dir=".", save_dir="./save",
        transcription_label_path="./label.txt",
        vocabulary_path="vocabulary/kor_grapheme.csv",
        src_dir="Source", label_dir="Label", 
        src_id='TS', label_id='TL',
        resize_shape=(224,224), fps=30):
    if not osp.exists(save_dir): os.makedirs(save_dir)
    fs = open(transcription_label_path, 'w', encoding='utf8')
    
    print(f"""
        Data Preprocessing Starts.
        Load Data from "{root_dir}" and Processed Data will be placed at "{save_dir}"
        """)
    
    char2id, id2char = load_label(vocabulary_path)
    label_paths = glob.glob(f"{root_dir}/{label_dir}/{label_id}*/*/*/*/*/*.json")
    print(f"""Cropping Data and Generating Label at "{transcription_label_path}".""")
    for label_path in tqdm.tqdm(label_paths):
        with open(label_path, encoding='utf8') as f:
            label_contents = json.load(f)[0]
        src_path = label_path.replace(f"/{label_dir}/{label_id}", f"/{src_dir}/{src_id}")
        audio_path = src_path.replace('.json','.wav')
        video_path = src_path.replace('.json','.mp4')
            
        output_save_dir = re.sub(root_dir, save_dir, video_path, count=1).replace(".mp4","")
        if not osp.exists(output_save_dir): os.makedirs(output_save_dir)

        # Only one audio sequence is provided per view set, with labeled 'A'
        pass_audio = re.search('.*[^A]_\d\d\d.wav', audio_path) is not None
        if pass_audio:
            audio_path = re.sub('(.*)[^A]_(\d\d\d).wav','(\1)A_(\2).wav', audio_path) ; print(audio_path)
        else: 
            audio_array, sr = load_audio(audio_path)

        sentence_info = label_contents['Sentence_info']
        bbox_sequence = label_contents['Bounding_box_info']['Lip_bounding_box']['xtl_ytl_xbr_ybr']
        
        # Spatial Crop
        video_array = _load_and_crop_video(video_path, bbox_sequence, resize_shape=resize_shape)
        # Temporal Crop
        for sentence_idx, sentence_label in tqdm.tqdm(enumerate(sentence_info), leave=False):
            sentence, sentence_start_sec, sentence_end_sec = _get_sentence_info(sentence_label)
            
            audio_save_path = osp.join(output_save_dir, str(sentence_idx))+'.wav'
            if not pass_audio:
                _crop_and_save_audio_segment(
                    audio_save_path, audio_array, sr, sentence_start_sec, sentence_end_sec)
            
            video_save_path = osp.join(output_save_dir, str(sentence_idx))+'.mp4'
            _crop_and_save_video_segment(
                video_save_path, video_array, fps, 
                sentence_start_sec, sentence_end_sec, resize_shape=resize_shape)
            
            sentence = refine_transcription(sentence)
            sentence_id = sentence_to_target(sentence, char2id)

            fs.write('\t'.join([video_save_path, audio_save_path, sentence, sentence_id])+'\n')


def _get_sentence_info(label:dict):
    return (label['sentence_text'], label['start_time'], label['end_time'])


def load_audio(audio_path):
    y, sr = librosa.load(audio_path, sr = None)
    return y, sr


def _load_and_crop_video(video_path:str, bboxes:List[List[int]], resize_shape=(224,224)):
    reader = skvideo.io.FFmpegReader(video_path)
    cropped_frame_list = []
    for i, frame in enumerate(reader.nextFrame()):
        left, top, right, bottom = bboxes[i]
        cropped_img = frame[int(left):int(right), int(top):int(bottom)]
        cropped_img = cv2.resize(cropped_img, resize_shape, interpolation = cv2.INTER_LINEAR)
        cropped_img = cv2.cvtColor(cropped_img, cv2.COLOR_RGB2BGR)
        cropped_frame_list.append(cropped_img)
    return np.stack(cropped_frame_list, axis=0)


def _crop_and_save_audio_segment(path:str, array:ndarray, sr:int, start:float, end:float):
    cropped_array = array[int(sr*start):int(sr*end)]
    soundfile.write(path, data=cropped_array, samplerate=sr)


def _crop_and_save_video_segment(
        path:str, array:ndarray, fps:float, start:float, end:float, 
        resize_shape:tuple=(224,224)):
    cropped_array = array[int(fps*start):int(fps*end)]
    save_video_mp4(path, cropped_array, fps=fps, resize_shape=resize_shape)


def crop_video(frames:ndarray, bboxes:List[List[int]], crop_shape=(224,224)):
    cropped_frame_list = []
    for i, frame in enumerate(frames):
        [left, top, right, bottom] = bboxes[i]
        cropped_img = frame[int(left):int(right), int(top):int(bottom)]
        cropped_img = cv2.resize(cropped_img, crop_shape, interpolation = cv2.INTER_LINEAR)
        cropped_img = cv2.cvtColor(cropped_img, cv2.COLOR_RGB2BGR)
        cropped_frame_list.append(cropped_img)
    return np.stack(cropped_frame_list, axis=0)


def save_video_mp4(save_path, video_frames, fps=30, resize_shape=(224,224)):
    out = cv2.VideoWriter(
            save_path,
            cv2.VideoWriter_fourcc(*'mp4v'),
            fps,
            resize_shape,
        )
    for video_frame in video_frames:
        out.write(video_frame)
    out.release()


def load_label(filepath, encoding='utf-8'):
    import pandas as pd

    char2id = dict()
    id2char = dict()

    ch_labels = pd.read_csv(filepath, encoding=encoding)
    id_list = ch_labels["id"]
    char_list = ch_labels["char"]
    #freq_list = ch_labels.get("freq", id_list)
    #ord_list = ch_labels["ord"]

    for (id_, char) in zip(id_list, char_list):
        char2id[char] = id_
        id2char[id_] = char
    return char2id, id2char


def sentence_to_target(sentence, char2id, unit='grapheme'):
    # tokenize
    _sentence = char2grp(sentence)
    target = str()
    
    for ch in _sentence:
        try:
            target += (str(char2id[ch]) + ' ')
        except KeyError:
            print(f"KeyError Occured, Key : '{ch}', sentence : '{sentence}'")
            continue

    return target[:-1]

    
def num2kor(num):
    num = int(num)
    unit = ['일','만','억','조']
    sub_unit = ['일','십','백','천']
    nums = '영일이삼사오육칠팔구'
    string = ''
    
    if num == 0:
        return nums[num]
    if num == 10000:
        return unit[1]
        
    for i in range(len(unit)-1, -1, -1):
        k, num = divmod(num, 10**(4*i))
        if k==0: continue
        for j in range(3, -1, -1):
            l, k = divmod(k, 10**j)
            if l > 0:
                if l > 1 or j == 0:
                    string += nums[l]
                if j > 0: string += sub_unit[j]
        if i > 0:
            string += unit[i]
            string += ' '
    return string
    

def refine_transcription(transcript:str):
    transcript = re.sub('\xa0',' ',transcript) # \xa0 : space
    transcript = re.sub('[Xx]',' ',transcript) # x : mute
    transcript = re.sub('[%]','퍼센트',transcript) # % : percent
    transcript = re.sub('\d+', lambda x: num2kor(x.group(0)), transcript) # number
    transcript = unzip_groups(transcript) # (아기씨)/(애기씨) (안돼써)(안됐어) (그런)게/(그러)게
    transcript = re.sub('[^ㄱ-ㅎ가-힣\s]', '', transcript)
    return transcript

 
def unzip_groups(transcript):
    # Use the former one, while not grammatically correct
    pattern = '(\(([^(/)]+)\)?([^(/)]*))\/?(\(?([^(/)]+)\)(\3)?)'
    if re.search(pattern, transcript):
        _transcript = re.sub(pattern, f"\{1}", transcript)
        _transcript = re.sub('[(/)]', "", _transcript)
        result = unzip_groups(_transcript)
    else:
        result = transcript
    return result


def get_args():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--root_dir', type=str, default='./src_data')
    parser.add_argument('--src_dir', type=str, default='Source')
    parser.add_argument('--src_id', type=str, default='TS')
    parser.add_argument('--label_dir', type=str, default='Label')
    parser.add_argument('--label_id', type=str, default='TL')
    parser.add_argument('--save_dir', type=str, default='./data')
    parser.add_argument('--vocab_path', type=str, default='vocabulary/kor_grapheme.csv')
    parser.add_argument('--transcript_save_path', type=str, default='./label.txt')

    parser.add_argument('--resize_video', type=int, nargs='+', default=(224,224))
    parser.add_argument('--fps', type=float, default=30)
    args = parser.parse_args()
    return args


if __name__=='__main__':
    # assert len(sys.argv) > 1, "You Should Pass the Arguments."
    args = get_args()
    preprocess_source_data(
        root_dir=args.root_dir, save_dir=args.save_dir, 
        src_dir=args.src_dir, src_id=args.src_id,
        label_dir=args.label_dir, label_id=args.label_id,
        transcription_label_path=args.transcript_save_path,
        vocabulary_path=args.vocab_path,
        resize_shape=args.resize_video,
        fps=args.fps
    )