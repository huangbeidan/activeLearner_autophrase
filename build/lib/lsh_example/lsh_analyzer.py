from collections import defaultdict
from itertools import chain
from statistics import stdev
import matplotlib.pyplot as plt
import heapq
import numpy as np
import math

from active_learner.Phrases import Phrases
from active_learner.al_autophrase import ActiveLearner
from active_learner.pos_analyzer import PosTag_Query_Fetcher
from active_learner.lsh_autophrase import LSH_Autophrase

class LSHAnalyzer:
    def __init__(self, num_queries=5, threshold_nlargest=0.1):
        """
        Active Learner for phrase extraction, integrated with LSH, pos-tagging algorithms

        :param num_queries:
           The number of phrases for presenting users for labeling
        """

        #initialzing phrase interface
        #TODO: these two files need to be dynamically replaced by Autophrase immediate outputs
        self.token_mapping_dir = "input/token_mapping.txt"
        self.intermediate_labels_dir = "input/intermediate_labels.txt"
        self.tokenized_train_dir = "input/tokenized_train.txt"
        self.pos_tags_tokenized_train_dir = "input/pos_tags_tokenized_train.txt"

        ## Paramters for threshold
        self.thres_unique_counts = 5
        self.thres_parent_chil_diff = 0.1

        self.phrases_interface = Phrases(self.token_mapping_dir, self.intermediate_labels_dir)
        self.word2phrase_dict = self.phrases_interface.word2phrase
        self.phrase_labels_dict = self.phrases_interface.phrase_labels_dict
        self.token2word_dict = self.phrases_interface.token2word
        self.phrases_input = self.phrases_interface.phrases

        self.lsh_output_file = LSH_Autophrase(self.phrases_input).main()
        self.lines = []
        self.scores_lines = []
        self.load_lsh_groups()
        self.ind = self.get_nlargest(int(len(self.scores_lines) * threshold_nlargest), self.scores_lines)
        # ind = get_nsmallest(int(len(scores_lines)*0.1), scores_lines)

        self.pos_interface = PosTag_Query_Fetcher(self.phrases_interface, self.tokenized_train_dir,
                                                  self.pos_tags_tokenized_train_dir, self.thres_unique_counts,
                                                  self.thres_parent_chil_diff)

        ## array to collect query candidates
        self.query_group_1 = []

        # this idx is the idx in lsh_autophrase_output.txt! not the original index
        # idx for finding the top variance groups after lsh result
        for idx in self.ind:
            temp = ""
            lowest_phrase = ""
            lowest_score = 100
            for pair in str(self.lines[idx]).split(','):
                if ":" not in pair: continue
                tuple = pair.split(':')
                phrase = tuple[0]
                score = tuple[1]
                if phrase not in self.phrase_labels_dict: continue
                label = self.phrase_labels_dict[phrase]
                temp = temp + str(phrase) + ":" + str(score) + ":" + str(label) + ","
                if float(score) > 0.9:
                    self.query_group_1.append(self.word2phrase_dict[phrase])
                if float(score) < lowest_score:
                    lowest_phrase = phrase
            self.query_group_1.append(self.word2phrase_dict[lowest_phrase])

            print(temp + "\n")

        ## query 2: get phrases with weird pos tag patterns (unique count > 4)
        self.group_2_raw = self.pos_interface.query_pos_tags_1()
        self.query_group_2 = [self.token2word_dict[token] for token in self.group_2_raw]

        ## Query_3: get phrases whose subchunk has a diverse score with parents'
        # TODO: delete pos_tag_patterns in the directory if changing corpus in AutoPhrase!
        self. query_group_3 = self.pos_interface.query_pos_tags_2()
        self.query_group_3_flatten = list(chain.from_iterable(self.query_group_3))

        # Compile two query groups
        # Testing
        query_test_1 = self.query_group_2

        # Add sampling technique - sample @ K
        len1 = len(self.query_group_1)
        len2 = len(self.query_group_2)
        len3 = len(self.query_group_3_flatten)
        total_len = len1 + len2 + len3

        if total_len < num_queries:
            self.query_group_combined = list(set(self.query_group_1 + self.query_group_2 + self.query_group_3_flatten))
        else:
            nlen1 = math.ceil(len1 / total_len * num_queries)
            nlen2 = math.ceil(len2 / total_len * num_queries)
            nlen3 = math.ceil(len3 / total_len * num_queries)

            self.query_group_combined = list(
                set(self.query_group_1[:nlen1] + self.query_group_2[:nlen2] + self.query_group_3_flatten[:nlen3]))

        # check query group1
        print("Successfully get {} candidate phrases \n".format(len(self.query_group_combined)))
        print("Get {} from group 1, {} from group 2, {} from group 3 with K = {} \n".format(nlen1, nlen2, nlen3, num_queries))

        ## Call ActiveLearner for human labeling
        ALearner = ActiveLearner(self.query_group_combined)
        ALearner.main()

    def load_lsh_groups(self):

        with open(self.lsh_output_file) as content:
            for line in content:
                line = line.strip()
                self.lines.append(line)
                temp = []
                count = 0
                for pair in line.split(','):
                    if ':' not in pair:
                        continue
                    tt = pair.split(':')
                    phrase = tt[0]
                    score = tt[1]
                    print(pair)
                    temp.append((float(score)))
                    count += 1
                if count == 0:
                    print("not sufficient data samples")
                    continue
                line_std = stdev(temp)
                self.scores_lines.append(line_std)


    def get_nlargest(self, n, iter):
        ind = np.argpartition(self.scores_lines, -n)[-n:]
        return ind


    def get_nsmallest(self, n, iter):
        ind = np.argpartition(self.scores_lines, -n)[:n]
        return ind


if __name__ == "__main__":
    analyzer = LSHAnalyzer()

