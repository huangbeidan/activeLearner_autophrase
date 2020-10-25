from collections import defaultdict
from statistics import stdev
import matplotlib.pyplot as plt
import heapq
import numpy as np

from lsh_example.Phrases import Phrases
from lsh_example.al_autophrase import ActiveLearner


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

    ind = get_nlargest(int(len(scores_lines)*0.1), scores_lines)
    # ind = get_nsmallest(int(len(scores_lines)*0.1), scores_lines)

    phrases_interface = Phrases()
    word2phrase_dict = phrases_interface.word2phrase
    phrase_labels_dict = phrases_interface.phrase_labels_dict

    ## array to collect query candidates
    query_group_1 = []

    # this idx is the idx in lsh_autophrase_output.txt! not the original index
    # idx for finding the top variance groups after lsh result
    for idx in ind:
        temp = ""
        for pair in str(lines[idx]).split(','):
            if ":" not in pair: continue
            tuple = pair.split(':')
            phrase = tuple[0]
            score = tuple[1]
            label = phrase_labels_dict[phrase]
            temp = temp + str(phrase) + ":" + str(score) + ":" + str(label) + ","
            if float(score) > 0.9:
                query_group_1.append(word2phrase_dict[phrase])

        print(temp + "\n")

    ## check query group1
    print("Successfully get {} candidate phrases \n".format(len(query_group_1)))

    ## Call ActiveLearner for human labeling
    ALearner = ActiveLearner(query_group_1)
    ALearner.main()

