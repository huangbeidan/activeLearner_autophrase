#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse
from collections import defaultdict

from active_learner.Phrases import Phrases

SEED = 3

import py_entitymatching as em
from snapy import MinHash, LSH
import pandas


# class Phrase:
#     def __init__(self):
#         self.idx = -1
#         self.quality = -1
#         self.tokens = ""
#         self.words = ""
#         self.label = ""
#
#     def add_phrase(self, idx, quality, tokens, words, label):
#         self.idx = idx
#         self.quality = quality
#         self.tokens = tokens
#         self.words = words
#         self.label = label

class ActiveLearner:

    def __init__(self, phrases):
        self.phrases = phrases


    def load_tokens_mapping(self):
        tokens_dict = defaultdict(lambda:' ')
        with open('/home/beidan/AutoPhrase/tmp/token_mapping.txt') as content:
            for line in content:
                line = line.strip()
                cans = line.split('\t')
                if len(cans) > 1:
                    token = cans[0]
                    word = cans[1]
                    tokens_dict[token] = word
        return tokens_dict


    # def load_content_v2(self, sentence_file):
    #     phrases = []
    #     tokens_dict = self.load_tokens_mapping()
    #
    #     with open(sentence_file) as content:
    #         for line in content:
    #             line = line.strip()
    #             cans = line.split('\t')
    #             phrase = ""
    #             if len(cans) > 3:
    #                 idx = cans[0]
    #                 label = cans[1]
    #                 score = cans[2]
    #                 tokens_raw = cans[3]
    #             for tk in tokens_raw.split(' '):
    #                 # assert tk in tokens_dict.keys(), "tokens should be in tokens mapping dictionary"
    #                 phrase += tokens_dict[tk]
    #                 phrase += " "
    #
    #             phrase_clean = phrase.replace(",", "")
    #             phrase_clean = phrase_clean.lower()
    #             phrase_clean = phrase_clean.strip()
    #
    #             pp = Phrase()
    #             pp.add_phrase(idx, score, tokens_raw, phrase_clean, label)
    #             phrases.append(pp)
    #
    #     return phrases


    def query_high_score_neg_label_items(self, phrases):
        outputs = [p for p in phrases if float(p.quality) > 0.85 and p.label == '0']
        return outputs


    def query_low_score_pos_label_items(self, phrases):
        outputs = [p for p in phrases if float(p.quality) < 0.85 and p.label == '1']
        return outputs


    def phrases_to_dataframe(self, candidates):
        df = pandas.DataFrame(columns=['idx', "words", "score", "label"])
        i = 0
        for p in candidates:
            print("words: ", p.words, " score: ", p.quality, " label: ", p.label)
            row = [p.idx, p.words, p.quality, p.label]
            df.loc[i] = row
            i += 1
        return df


    def user_labeling(self, candidates):
        df = self.phrases_to_dataframe(candidates)
        df.to_csv(r'temp.csv', index=False)

        # df_csv = em.read_csv_metadata('temp.csv', key='idx')
        # em.label_table(df_csv, 'gold_label')
        df_cc = em._init_label_table(df, 'gold_label')
        # df_cc['gold_label'] = df['label']
        df_cc['gold_label'] = 1

        # Invoke the GUI
        try:
            from PyQt5 import QtGui
        except ImportError:
            raise ImportError('PyQt5 is not installed. Please install PyQt5 to use '
                              'GUI related functions in py_entitymatching.')

        from py_entitymatching.gui.table_gui import edit_table
        edit_table(df_cc)

        df_cc = self._post_process_labelled_table(df_cc, 'gold_label')

        print(df_cc.head())

        with open('/home/beidan/AutoPhrase/tmp/labeled_patterns.txt', 'w', encoding='utf-8') as f:
            for rec_idx, rec in df_cc.iterrows():
                f.write(rec['idx'] + '\t' + str(rec['gold_label']) + '\n')

        print("successfully write {} records to Autophrase".format(len(df_cc)))

        # S = em.sample_table(df_csv, 5)
    def _post_process_labelled_table(self, labeled_table, col_name):
        labeled_table[col_name] = labeled_table[col_name].astype(int)

        # Check if the table contains only 0s and 1s
        label_value_with_1 = labeled_table[col_name] == 1
        label_value_with_0 = labeled_table[col_name] == 0
        sum_of_labels = sum(label_value_with_1 | label_value_with_0)

        # If they contain column values other than 0 and 1, raise an error
        if not sum_of_labels == len(labeled_table):
            em.logger.error('The label column contains values other than 0 and 1')
            raise AssertionError(
                'The label column contains values other than 0 and 1')
        # Return the label table
        return labeled_table


    def main(self):
        query = "/home/beidan/AutoPhrase/tmp/intermediate_labels.txt"

        # load sentences from file (v2 is the autophrase version)
        # phrases = self.load_content_v2(query)

        print("get total number of {} phrases".format(len(self.phrases)))

        # query_outputs_1 = self.query_high_score_neg_label_items(self.phrases)
        # query_outputs_2 = self.query_low_score_pos_label_items(self.phrases)

        # integrated = query_outputs_2 + query_outputs_1
        # print(integrated)

        self.user_labeling(self.phrases)


if __name__ == "__main__":
    #phrases = Phrases().phrases
    #ActiveLearner(phrases).main()
    print("Warning: please start from the lsh_autophrase step!")