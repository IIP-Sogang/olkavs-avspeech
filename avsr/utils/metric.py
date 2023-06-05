import Levenshtein as Lev
import pdb
import os

from .korean_funcs import grp2char

class Metric:
    def __init__(self, vocab, log_path, unit='character', error_type='cer'):
        super().__init__()
        if error_type=='cer':
            self.metric = CharacterErrorRate(
                vocab,
                log_path = log_path,
                unit = unit,
            )
        elif error_type=='wer':
            self.metric = WordErrorRate(
                vocab,
                log_path = None,
                unit = unit,
            )
        elif error_type=='swer':
            self.metric = SpaceWordErrorRate(
                vocab,
                log_path = None,
                unit = unit,
            )
        elif error_type=='ger':
            self.metric = CharacterErrorRate(
                vocab,
                log_path = None,
                convert=False
            )
            
    def reset(self):
        self.metric.reset()
    
    def __call__(self, targets, outputs, target_lengths=None, output_lengths=None, show=False, file_path=None):
        y_hats = outputs
        if output_lengths is not None:
            y_hats = [output[:output_lengths[i].item()] for i, output in enumerate(y_hats)]
        if target_lengths is not None:
            targets = [target[:target_lengths[i].item() - 1] for i, target in enumerate(targets)] # Minus the <end> token.
        return self.metric(targets, y_hats, show=show, file_path=file_path)
    

class ErrorRate(object):
    """
    Provides inteface of error rate calcuation.
    Note:
        Do not use this class directly, use one of the sub classes.
    """

    def __init__(self, vocab, log_path : str = None, unit:str = 'grapheme') :
        self.total_dist = 0.0
        self.total_length = 0.0
        self.vocab = vocab
        self.log_path = log_path
        self.unit = unit

    def reset(self):
        self.total_dist = 0.0
        self.total_length = 0.0

    def __call__(self, targets, y_hats, show=False, file_path=None):
        """ Calculating character error rate """
        dist, length = self._get_distance(targets, y_hats, show=show, file_path=file_path)
        try:
            return dist, length
        except:
            pdb.set_trace()

    def _get_distance(self, targets, y_hats, show=False, file_path=None):
        """
        Provides total character distance between targets & y_hats
        Args:
            targets (torch.Tensor): set of ground truth
            y_hats (torch.Tensor): predicted y values (y_hat) by the model
        Returns: total_dist, total_length
            - **total_dist**: total distance between targets & y_hats
            - **total_length**: total length of targets sequence
        """
        total_dist = 0
        total_length = 0

        for i, (target, y_hat) in enumerate(zip(targets, y_hats)):
            s1 = self.vocab.label_to_string(target)
            s2 = self.vocab.label_to_string(y_hat)

            # Print Results
            if show:
                print(f"Tar: {s1}")
                print(f"Out: {s2}")
                print('==========')
            # Record Results
            elif self.log_path:
                save_folder = f'results/metric_log'
                if not os.path.exists(save_folder):
                    os.makedirs(save_folder)
                with open(f'{save_folder}/{self.log_path}', 'a') as f:
                    f.write(f"Tar: {s1}\n")
                    f.write(f"Out: {s2}\n")
                    if file_path is not None:
                        f.write(f'{file_path[i]}\n')
                    else:
                        f.write(f'==========\n')
            
            if self.unit=='grapheme' and getattr(self, 'convert', True):
                s1 = grp2char(s1)
                s2 = grp2char(s2)
            dist, length = self.metric(s1, s2)
            total_dist += dist
            total_length += length

        return total_dist, total_length

    def metric(self, *args, **kwargs):
        raise NotImplementedError


class CharacterErrorRate(ErrorRate):
    
    def __init__(self, vocab, log_path:str = None, unit:str='grapheme', convert:bool=True):
        super(CharacterErrorRate, self).__init__(vocab, log_path, unit)
        self.convert = convert

    def metric(self, s1: str, s2: str):
        # if '_' in sentence, means subword-unit, delete '_'
        if '_' in s1:
            s1 = s1.replace('_', '')

        if '_' in s2:
            s2 = s2.replace('_', '')

        s1 = s1.strip()
        s2 = s2.strip()
        dist = Lev.distance(s2, s1)
        length = len(s1)

        return dist, length


class WordErrorRate(ErrorRate):
    """
    Computes the Word Error Rate, defined as the edit distance between the
    two provided sentences after tokenizing to words.
    """
    def __init__(self, vocab, log_path:str = None, unit:str='grapheme'):
        super(WordErrorRate, self).__init__(vocab, log_path, unit)

    def metric(self, s1, s2):
        """
        Computes the Word Error Rate, defined as the edit distance between the
        two provided sentences after tokenizing to words.
        Arguments:
            s1 (string): space-separated sentence
            s2 (string): space-separated sentence
        """

        # build mapping of words to integers
        b = set(s1.split() + s2.split())
        word2char = dict(zip(b, range(len(b))))

        # map the words to a char array (Levenshtein packages only accepts
        # strings)
        w1 = [chr(word2char[w]) for w in s1.split()]
        w2 = [chr(word2char[w]) for w in s2.split()]

        dist = Lev.distance(''.join(w1), ''.join(w2))
        length = len(s1.split())

        return dist, length


class SpaceWordErrorRate(ErrorRate):
    """
    Computes the Word Error Rate, defined as the edit distance between the
    two provided sentences after tokenizing to words.
    """
    def __init__(self, vocab, log_path:str = None, unit:str='grapheme'):
        super(SpaceWordErrorRate, self).__init__(vocab, log_path, unit)

    def get_space_normalized_text(self, ref, hyp):
        import numpy as np

        # Return original texts if hypothesis has an empty sequence
        if hyp=='': return ref, hyp

        refs = self.get_space_tokenized_text(ref)
        hyps = self.get_space_tokenized_text(hyp)
        
        # calculate levenshtein distance between unspaced sequences
        scores = np.zeros((len(hyps)+1, len(refs)+1))
        scores[0,:] = range(len(refs)+1)
        for h in range(1,len(hyps)+1):
            scores[h,0] = scores[h-1,0]
            for r in range(1,len(refs)+1):
                hyp_nosp = self.get_no_spaced_text(hyps[h-1])
                ref_nosp = self.get_no_spaced_text(refs[r-1])
                sub_or_cor = scores[h-1, r-1] + int(hyp_nosp!=ref_nosp)
                _ins = scores[h-1, r] + 1
                _del = scores[h, r-1] + 1
                scores[h,r] = min(sub_or_cor, _ins, _del)
                
        # alignment
        r = len(refs)
        h = len(hyps)
        last_r = r
        last_h = h
        ref_norm = []
        hyp_norm = []
        while r>0 and h>0:
            if h==0:
                last_r = r-1
            elif r==0:
                last_h = h-1
            else:
                hyp_nosp = self.get_no_spaced_text(hyps[h-1])
                ref_nosp = self.get_no_spaced_text(refs[r-1])
                sub_or_cor = scores[h-1, r-1] + int(hyp_nosp!=ref_nosp)
                _ins = scores[h-1, r] + 1
                _del = scores[h, r-1] + 1
                if sub_or_cor <= min(_ins, _del):
                    last_r, last_h = r-1, h-1
                elif _ins<_del:
                    last_r, last_h = r, h-1
                else:
                    last_r, last_h = r-1, h
            
            c_hyp = hyps[last_h] if last_h >= 0 and last_h < len(hyps) else ""
            c_ref = refs[last_r] if last_r >= 0 and last_r < len(refs) else ""
            h, r = last_h, last_r
            
            if self.get_no_spaced_text(c_hyp)==self.get_no_spaced_text(c_ref):
                c_hyp = c_ref
            
            ref_norm.insert(0,c_ref)
            hyp_norm.insert(0,c_hyp)
            
        return "".join(ref_norm), "".join(hyp_norm)

    def get_no_spaced_text(self, text):
        return text.replace(" ","")

    def get_space_tokenized_text(self, text):
        tokens = []
        text = list(text)
        while text:
            token = text.pop(0)
            if token==' ':
                if len(text)!=0:
                    tokens.append(token+text.pop(0))
                else: pass
            else:
                tokens.append(token)
        return tokens

    def metric(self, s1, s2):
        """
        Computes the Word Error Rate, defined as the edit distance between the
        two provided sentences after tokenizing to words.
        Arguments:
            s1 (string): space-separated sentence
            s2 (string): space-separated sentence
        """
        s1, s2 = self.get_space_normalized_text(ref=s1, hyp=s2)

        # build mapping of words to integers
        b = set(s1.split() + s2.split())
        word2char = dict(zip(b, range(len(b))))

        # map the words to a char array (Levenshtein packages only accepts
        # strings)
        w1 = [chr(word2char[w]) for w in s1.split()]
        w2 = [chr(word2char[w]) for w in s2.split()]

        dist = Lev.distance(''.join(w1), ''.join(w2))
        length = len(s1.split())

        return dist, length

