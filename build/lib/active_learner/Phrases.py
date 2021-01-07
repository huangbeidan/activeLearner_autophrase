#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from collections import defaultdict

class Phrase:
    def __init__(self):
        self.idx = -1
        self.quality = -1
        self.tokens = ""
        self.words = ""
        self.label = ""

    def add_phrase(self, idx, quality, tokens, words, label):
        self.idx = idx
        self.quality = quality
        self.tokens = tokens
        self.words = words
        self.label = label


class Phrases:

    def __init__(self, token_mapping_dir="input/token_mapping.txt", intermediate_labels_dir="input/intermediate_labels.txt"):
        """
        :param token_mapping_dir: Immediate output from AutoPhrase
        :param intermediate_labels_dir: Intermediate labels produced by AL-Autophrase: https://github.com/huangbeidan/AutoPhrase

        Example files have been put under input/ folder
        """
        self.phrases = []
        self.phrases_num = []
        self.phrase_labels_dict=defaultdict()
        self.word2phrase = defaultdict()
        self.token2word = defaultdict()

        self.token_mapping_dir = token_mapping_dir
        self.intermediate_labels_dir = intermediate_labels_dir

        self.load_content_v2(self.intermediate_labels_dir)

    def _get_phrases(self):
        return self.phrases

    def load_tokens_mapping(self):
        tokens_dict = defaultdict(lambda:' ')
        with open(self.token_mapping_dir) as content:
            for line in content:
                line = line.strip()
                cans = line.split('\t')
                if len(cans) > 1:
                    token = cans[0]
                    word = cans[1]
                    tokens_dict[token] = word
        return tokens_dict


    def load_content_v2(self, sentence_file):
        tokens_dict = self.load_tokens_mapping()

        with open(sentence_file) as content:
            for line in content:
                line = line.strip()
                cans = line.split('\t')
                phrase = ""
                if len(cans) > 3:
                    idx = cans[0]
                    label = cans[1]
                    score = cans[2]
                    tokens_raw = cans[3]
                for tk in tokens_raw.split(' '):
                    # assert tk in tokens_dict.keys(), "tokens should be in tokens mapping dictionary"
                    phrase += tokens_dict[tk]
                    phrase += " "

                phrase_clean = phrase.replace(",", "")
                phrase_clean = phrase_clean.replace(":", "")
                phrase_clean = phrase_clean.lower()
                phrase_clean = phrase_clean.strip()

                if len(phrase_clean.split(' ')) < 2: continue

                pp = Phrase()
                pp.add_phrase(idx, score, tokens_raw, phrase_clean, label)
                self.phrases.append(pp)
                self.phrases_num.append(tokens_raw)
                self.phrase_labels_dict[phrase_clean] = label
                self.word2phrase[phrase_clean] = pp
                self.token2word[tokens_raw] = pp

