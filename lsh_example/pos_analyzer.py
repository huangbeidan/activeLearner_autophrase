import itertools
import os
import pickle
import re
from collections import defaultdict
import matplotlib.pyplot as plt
plt.style.use('seaborn-white')
import numpy as np
import pandas as pd
import ast

import dill
from tqdm import tqdm

from lsh_example.Phrases import Phrases

class PosTag_Query_Fetcher:
    def __init__(self):
        self.phrase_interface = Phrases()
        self.phrases = self.phrase_interface.phrases
        self.token2phrase_dict = self.phrase_interface.token2word


    def get_all_tokens(self):
        tokens = []
        with open('/home/beidan/AutoPhrase/tmp/tokenized_train.txt') as content:
            cnt = 0
            for line in content:
                vector = line.split(' ')
                tokens += vector
            print("total tokens: ", len(tokens))
            return tokens


    def get_all_tags(self):
        tags = []
        with open('/home/beidan/AutoPhrase/tmp/pos_tags_tokenized_train.txt') as content:
            for line in content:
                tags.append(line.strip().replace("\n", ""))
            print("total tags: ", len(tags))
            return tags


    def build_index(self, plist):
        inverted = defaultdict(lambda: list())
        for idx, token in enumerate(plist):
            inverted[token].append(idx)
        return inverted


    def find_one_v2(self, target, inverted_idx, plist):
        targets = target.split(" ")
        indices = []

        start_pos = inverted_idx[targets[0]]
        for idx, pos in enumerate(start_pos):
            flag = False
            for j in range(len(targets)):
                if plist[pos+j] != targets[j]:
                    flag = False
                    break
                flag = True
            if flag:
                indices.append(pos)
        return indices, len(targets)


    def find_one(self, target, plist):
        targets = target.split(" ")
        indices = []
        i = 0

        while i < (len(plist)):
            flag = False
            if plist[i] == targets[0]:
                for j in range(len(targets)):
                    if plist[i + j] != targets[j]:
                        flag = False
                        break
                    flag = True

            if flag:
                indices.append(i)
                i += len(targets)
            else:
                i += 1

        # print("indices: ", indices)
        return indices, len(targets)


    def find_pos_tag_patterns(self):
        tokens = self.get_all_tokens()
        tags = self.get_all_tags()

        phrases = self.phrases

        pos_tags_dict = defaultdict(lambda: list())
        scores_dict = defaultdict()

        inverted_idx = self.build_index(tokens)


        for phr_raw in tqdm(phrases):
            phr = phr_raw.tokens
            indices, len_target = self.find_one_v2(phr, inverted_idx, tokens)
            # print("phr: ", phr, "indices: ", indices, "phrase length: ", len_target)
            if(len(indices)==0):
                continue
            for idx in indices:
                pattern = ""
                for l in range(len_target):
                    pattern += (tags[idx+l] + " ")
                pos_tags_dict[phr].append(pattern)
            if phr not in scores_dict:
                scores_dict[phr] = phr_raw.quality

        print("hello")
        return pos_tags_dict, scores_dict



    def pos_pattern_generator(self):
        if os.path.isfile("pos_tags_patterns_backup"):
            pos_tags_patterns = dill.load(open('pos_tags_patterns_backup', 'rb'))
        else:
            pos_tags_dict, scores_dict = self.find_pos_tag_patterns()
            # pos_tags_patterns_backup should be like: posTag : [score1, score2, score3 .... ]
            pos_tags_patterns = defaultdict(lambda: list())

            # initialize the first time and then save to pickle
            for idx, phr in tqdm(enumerate(pos_tags_dict)):
                #phr in token: 563 564
                score = scores_dict[phr]
                for pos in pos_tags_dict[phr]:
                    pos_tags_patterns[pos].append(score)
            dill.dump(pos_tags_patterns, open('pos_tags_patterns_backup', 'wb'))

        # 2nd time and onwards, load from pickle
        return pos_tags_patterns

    def analyzer(self):
        pos_tags_statistics = dict()
        pos_tags_patterns = self.pos_pattern_generator()

        for idx, pattern in enumerate(pos_tags_patterns):
            scores = (pos_tags_patterns[pattern])
            scores = list(map(float, scores))
            weighted_mean = np.mean(scores)
            freq = len(scores)
            minVal = min(scores)
            maxVal = max(scores)
            sd = np.std(scores)
            pos_tags_statistics[pattern] = [weighted_mean, freq, minVal, maxVal, sd]

        pos_tags_statistics_df = pd.DataFrame.from_dict(pos_tags_statistics, orient='index')
        pos_tags_statistics_df.columns = ['weighted_mean','freq','min','max','std']
        pos_tags_statistics_df.to_csv('pos_tags_statistics.csv', index=True)
        return pos_tags_patterns


    def get_pos_tag_unique_count(self):
        pos_tags_dict, scores_dict = self.find_pos_tag_patterns()
        unique_set = dict()
        for phr in pos_tags_dict:
            unique_set[phr] = len(set(pos_tags_dict[phr]))
        return unique_set


    def query_pos_tags_1(self):
        unique_set = self.get_pos_tag_unique_count()
        unique_set = sorted(unique_set.items(), key=lambda x:x[1], reverse=True)
        output = [phr[0] for phr in unique_set if phr[1] > 5]
        return output

    def query_pos_tags_2(self):
        # find sub-chunks whose score differs a lot from parents'
        tokens = [p.tokens for p in self.phrases]
        tokens.sort()

        # i = 0
        # res = []
        # while i < len(tokens):
        #     j = i
        #     tmp = []
        #     if i < len(tokens) - 1 and tokens[j] in tokens[i + 1]:
        #         tmp.append(tokens[j])
        #         while i < len(tokens) - 1 and tokens[j] in tokens[i + 1]:
        #             tmp.append(tokens[i + 1])
        #             i += 1
        #     if len(tmp) > 0 and abs(
        #             float(self.token2phrase_dict[tmp[0]].quality) - float(self.token2phrase_dict[tmp[-1]].quality)) > 0.2:
        #         tmp = [self.token2phrase_dict[t] for t in tmp]
        #         res.append(tmp)
        #     i += 1
        # return res

        i = 0
        res = defaultdict(lambda: 0)
        while i < len(tokens):
            j = i
            tmp = []
            if i < len(tokens) - 1 and tokens[j] in tokens[i + 1]:
                tmp.append(tokens[j])
                while i < len(tokens) - 1 and tokens[j] in tokens[i + 1]:
                    tmp.append(tokens[i + 1])
                    i += 1
            if len(tmp) > 0:
                diff = abs(
                    float(self.token2phrase_dict[tmp[0]].quality) - float(
                        self.token2phrase_dict[tmp[-1]].quality))
                # TODO: Threshold can be set here
                if diff > 0.1:
                    #tmp = [self.token2phrase_dict[t] for t in tmp]
                    res[str(tmp)] = diff
            i += 1
        res = sorted(res, key=lambda x:x[1], reverse=True)
        #convert back to list
        res = [ast.literal_eval(phr) for phr in res]
        res = [[self.token2phrase_dict[token] for token in group] for group in res]
        return res





if __name__ == "__main__":
    # get_all_tokens()
    # get_all_tags()
    # phrases_raw = Phrases().phrases_num
    # phrases = Phrases().phrases
    # tokens = get_all_tokens()
    # indices, len_target = find_one("2 865 4 433", tokens)
    # # for idx in indices:
    # #     print(tokens[idx] + " " + tokens[idx+1]+ " " + tokens[idx+2]+ " " + tokens[idx+3])
    # print("target has a length of: ", len_target)

    #find_pos_tag_patterns()

    pos_interface = PosTag_Query_Fetcher()
    # pos_tags_patterns_backup = pos_interface.analyzer()
    # query_pos_tags_1()
    res = pos_interface.query_pos_tags_2()



    print("hello")