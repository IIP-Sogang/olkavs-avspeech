#-*-coding:utf-8-*-

import re
import csv
import pdb
import torch
import torch.nn.functional as F

# 유니코드 한글 시작 : 44032, 끝 : 55203
BASE_CODE, CHOSUNG, JUNGSUNG = 44032, 588, 28
END_CODE = 55203

# 초성 리스트. 00 ~ 18
CHOSUNG_LIST = ['ㄱ', 'ㄲ', 'ㄴ', 'ㄷ', 'ㄸ', 'ㄹ', 'ㅁ', 'ㅂ', 'ㅃ', 'ㅅ', 'ㅆ', 'ㅇ', 'ㅈ', 'ㅉ', 'ㅊ', 'ㅋ', 'ㅌ', 'ㅍ', 'ㅎ']

# 중성 리스트. 00 ~ 20
JUNGSUNG_LIST = ['ㅏ', 'ㅐ', 'ㅑ', 'ㅒ', 'ㅓ', 'ㅔ', 'ㅕ', 'ㅖ', 'ㅗ', 'ㅘ', 'ㅙ', 'ㅚ', 'ㅛ', 'ㅜ', 'ㅝ', 'ㅞ', 'ㅟ', 'ㅠ', 'ㅡ', 'ㅢ', 'ㅣ']

# 종성 리스트. 00 ~ 27 + 1(1개 없음)
JONGSUNG_LIST = ['<unk>', 'ㄱ', 'ㄲ', 'ㄳ', 'ㄴ', 'ㄵ', 'ㄶ', 'ㄷ', 'ㄹ', 'ㄺ', 'ㄻ', 'ㄼ', 'ㄽ', 'ㄾ', 'ㄿ', 'ㅀ', 'ㅁ', 'ㅂ', 'ㅄ', 'ㅅ', 'ㅆ', 'ㅇ', 'ㅈ', 'ㅊ', 'ㅋ', 'ㅌ', 'ㅍ', 'ㅎ']


def char2ord(x):
    if len(x)!=1:
        return 0
    else:
        return ord(x)
    
def ord_labeling(df):
    df['ord'] = df['char'].apply(char2ord)
    return df

def findGraphemeTokens(text):
    result = list()
    isJS = False
    for char in text:
        if isJS:
            result.append('\\'+char)
            isJS = False
        elif char != '\\':
            result.append(char)
        else:
            isJS = True
    return result

def char2grp(test_keyword):
    split_keyword_list = list(test_keyword)

    result = list()
    for keyword in split_keyword_list:
        # 한글 여부 check 후 분리
        if re.match('.*[가-힣]+.*', keyword) is not None: # '.*[ㄱ-ㅎㅏ-ㅣ가-힣]+.*'
            char_code = ord(keyword) - BASE_CODE
            char1 = int(char_code / CHOSUNG)
            result.append(CHOSUNG_LIST[char1])
            char2 = int((char_code - (CHOSUNG * char1)) / JUNGSUNG)
            result.append(JUNGSUNG_LIST[char2])
            char3 = int((char_code - (CHOSUNG * char1) - (JUNGSUNG * char2)))
            if char3==0:
                #result.append('<unk>')
                pass
            else:
                result.append(f'\{JONGSUNG_LIST[char3]}')
        else:
            result.append(keyword)
    return result


def grp2char(sentence):

    JASOlist = findGraphemeTokens(sentence)

    id2KR = {code-BASE_CODE+5:chr(code)
             for code in range(BASE_CODE, END_CODE + 1)}
    id2KR[0] = '<pad>'
    id2KR[1] = '<sos>'
    id2KR[2] = '<eos>' 
    id2KR[3] = '<unk>'
    id2KR[4] = ' '
    # 5 ~ ... => 가 ~ .. 힣
    KR2id = {key:value for value, key in id2KR.items()}
    
    def reset_count():
        return 0, 5
    
    result = list()
    chr_count, chr_id = reset_count()

    jaso_lists = [CHOSUNG_LIST, JUNGSUNG_LIST, JONGSUNG_LIST]
    nums = [CHOSUNG, JUNGSUNG, 1]
       
    i = 0
    while JASOlist:
        i += 1
        if i>1000:
            pdb.set_trace()
        JS = JASOlist.pop(0)

        # character ended
        ## end token
        is_starting = '\\' not in JS and JS in jaso_lists[0]
        ## start + middle + end
        full_character = chr_count == 3
        ## start + middle
        half_character = chr_count == 2 and '\\' not in JS

        JS = JS.replace('\\','') 

        ## not jaso
        special_character = re.match('.*[ㄱ-ㅎㅏ-ㅣ가-힣]+.*', JS) is None
        
        # "ㄱㅣ\\ㄴ", "ㄱㅏㄱ"
        if full_character or half_character:
            result.append(id2KR[chr_id])
            JASOlist.insert(0, JS)
            chr_count, chr_id = reset_count()
        # "ㄱ?", "!"
        elif special_character:
            result.append(JS)
            chr_count, chr_id = reset_count()
        # "ㄱㄴ", "ㄱㄱ"
        elif JS not in jaso_lists[chr_count]:
            if is_starting:
                JASOlist.insert(0, JS)
            else:
                pass
            chr_count, chr_id = reset_count()
        else:
            chr_id += jaso_lists[chr_count].index(JS) * nums[chr_count]
            chr_count += 1

        if len(JASOlist)==0:
            if chr_count >= 2:
                result.append(id2KR[chr_id])
            
    result = ''.join(result)
    return result


class Vocabulary(object):
    """
    Note:
        Do not use this class directly, use one of the sub classes.
    """
    def __init__(self, *args, **kwargs):
        self.sos_id = None
        self.eos_id = None
        self.pad_id = None
        self.unk_id = None

    def label_to_string(self, labels):
        raise NotImplementedError


class KsponSpeechVocabulary(Vocabulary):
    def __init__(self, unit='character', encoding='utf-8'):
        super(KsponSpeechVocabulary, self).__init__()
        
        self.unit = unit
        self.vocab_dict, self.id_dict = self.load_vocab(encoding=encoding)
        self.sos_id = int(self.vocab_dict['<sos>'])
        self.eos_id = int(self.vocab_dict['<eos>'])
        self.pad_id = int(self.vocab_dict['<pad>'])
        self.unk_id = int(self.vocab_dict['<unk>'])
      

    def __len__(self):
        return len(self.vocab_dict)

    def label_to_string(self, labels, tolist=False):
        """
        Converts label to string (number => Hangeul)
        Args:
            labels (numpy.ndarray): number label
        Returns: sentence
            - **sentence** (str or list): symbol of labels
        """

        if len(labels.shape) == 1:
            sentence = list() if tolist else str()
            for label in labels:
                if label.item() == self.eos_id:
                    break
                elif label.item() == self.unk_id:
                    continue
                if tolist:
                    sentence.append(self.id_dict[label.item()])
                else:
                    sentence += self.id_dict[label.item()]
            return sentence

        sentences = list()
        for batch in labels:
            sentence = list() if tolist else str()
            for label in batch:
                if label.item() == self.eos_id:
                    break
                if tolist:
                    sentence.append(self.id_dict[label.item()])
                else:
                    sentence += self.id_dict[label.item()]
            sentences.append(sentence)
        return sentences

    def load_vocab(self, label_path="./vocabulary/kor_grapheme.csv", encoding='utf-8'):
        """
        Provides char2id, id2char
        Args:
            label_path (str): csv file with character labels
            encoding (str): encoding method
        Returns: unit2id, id2unit
            - **unit2id** (dict): unit2id[unit] = id
            - **id2unit** (dict): id2unit[id] = unit
        """
        unit2id = dict()
        id2unit = dict()
        
        try:
            with open(label_path, 'r', encoding=encoding) as f:
                labels = csv.reader(f, delimiter=',')
                next(labels)
                
                for row in labels:
                    unit2id[row[1]] = row[0]
                    id2unit[int(row[0])] = row[1]
                
                #unit2id['<blank>'] = len(unit2id)
                #id2unit[len(unit2id)] = '<blank>'

            return unit2id, id2unit
        except IOError:
            raise IOError("Character label file (csv format) doesn`t exist : {0}".format(label_path))