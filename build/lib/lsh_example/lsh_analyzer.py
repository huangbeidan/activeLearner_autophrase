from collections import defaultdict
from itertools import chain
from statistics import stdev
import matplotlib.pyplot as plt
import heapq
import numpy as np
import math

from lsh_example.Phrases import Phrases
from lsh_example.al_autophrase import ActiveLearner
from lsh_example.pos_analyzer import PosTag_Query_Fetcher


def load_lsh_groups():

    with open('lsh_autophrase_output.txt') as content:
        for line in content:
            line = line.strip()
            lines.append(line)
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
            scores_lines.append(line_std)


def get_nlargest(n, iter):
    ind = np.argpartition(scores_lines, -n)[-n:]
    return ind


def get_nsmallest(n, iter):
    ind = np.argpartition(scores_lines, -n)[:n]
    return ind


if __name__ == "__main__":
    lines = []
    scores_lines = []
    load_lsh_groups()
    # print(scores_lines)

    #TODO: threshold can be set here
    ind = get_nlargest(int(len(scores_lines)*0.1), scores_lines)
    # ind = get_nsmallest(int(len(scores_lines)*0.1), scores_lines)

    phrases_interface = Phrases()
    word2phrase_dict = phrases_interface.word2phrase
    phrase_labels_dict = phrases_interface.phrase_labels_dict
    token2word_dict = phrases_interface.token2word

    pos_interface = PosTag_Query_Fetcher()

    ## array to collect query candidates
    query_group_1 = []

    # this idx is the idx in lsh_autophrase_output.txt! not the original index
    # idx for finding the top variance groups after lsh result
    for idx in ind:
        temp = ""
        lowest_phrase = ""
        lowest_score = 100
        for pair in str(lines[idx]).split(','):
            if ":" not in pair: continue
            tuple = pair.split(':')
            phrase = tuple[0]
            score = tuple[1]
            if phrase not in phrase_labels_dict: continue
            label = phrase_labels_dict[phrase]
            temp = temp + str(phrase) + ":" + str(score) + ":" + str(label) + ","
            if float(score) > 0.9:
                query_group_1.append(word2phrase_dict[phrase])
            if float(score) < lowest_score:
                lowest_phrase = phrase
        query_group_1.append(word2phrase_dict[lowest_phrase])

        print(temp + "\n")


    ## query 2: get phrases with weird pos tag patterns (unique count > 4)
    group_2_raw = pos_interface.query_pos_tags_1()
    query_group_2 = [token2word_dict[token] for token in group_2_raw]

    ## Query_3: get phrases whose subchunk has a diverse score with parents'
    # TODO: delete pos_tag_patterns in the directory if changing corpus in AutoPhrase!
    query_group_3 = pos_interface.query_pos_tags_2()
    query_group_3_flatten = list(chain.from_iterable(query_group_3))

    # Compile two query groups
    #Testing
    query_test_1 = query_group_2


    # Add sampling technique - sample @ K
    len1 = len(query_group_1)
    len2 = len(query_group_2)
    len3 = len(query_group_3_flatten)
    total_len = len1 + len2 + len3

    #TODO: K is another parameter that can be set
    K = 5
    if total_len < K:
        query_group_combined = list(set(query_group_1 + query_group_2 + query_group_3_flatten))
    else:
        nlen1 = math.ceil(len1/total_len * K)
        nlen2 = math.ceil(len2/total_len * K)
        nlen3 = math.ceil(len3/total_len * K)

        query_group_combined = list(set(query_group_1[:nlen1] + query_group_2[:nlen2] + query_group_3_flatten[:nlen3]))


    # check query group1
    print("Successfully get {} candidate phrases \n".format(len(query_group_combined)))
    print("Get {} from group 1, {} from group 2, {} from group 3 with K = {} \n".format(nlen1,nlen2,nlen3,K))

    ## Call ActiveLearner for human labeling
    ALearner = ActiveLearner(query_group_combined)
    ALearner.main()

