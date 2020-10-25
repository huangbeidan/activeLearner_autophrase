#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse
from collections import defaultdict

from snapy import MinHash, LSH

from lsh_example.Phrases import Phrases

SEED = 3

class LSH_Autophrase:

    def __init__(self, phrases_input):
        self.tokens_dict, self.words_to_tokens_dict = self.load_tokens_mapping()
        self.scores_dict = defaultdict()
        self.phrases_lsh = phrases_input

    def load_tokens_mapping(self):
        tokens_dict = defaultdict(lambda : ' ')
        words_to_tokens_dict = defaultdict()
        with open('/home/beidan/AutoPhrase/tmp/token_mapping.txt') as content:
            for line in content:
                line = line.strip()
                cans = line.split('\t')
                if len(cans) > 1:
                    token = cans[0]
                    word = cans[1]
                    tokens_dict[token] = word
                    words_to_tokens_dict[word] = token
        return tokens_dict, words_to_tokens_dict

    def load_content_v3(self, phrases):
        sentences = {}
        for phrase in phrases:
            sentences[phrase.words] = phrase.words+'\t'+phrase.quality

        return sentences


    def load_content_v2(self, sentence_file):
        sentences = {}

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
                        phrase += self.tokens_dict[tk]
                        phrase += " "

                    phrase_clean = phrase.replace(",", "")
                    phrase_clean = phrase_clean.lower()
                    phrase_clean = phrase_clean.strip()

                    if len(phrase_clean.split(' ')) > 1:
                        sentences[phrase_clean] = phrase + '\t' + score

                        self.scores_dict[phrase_clean] = score

        return sentences





    def create_lsh(self, content, no_of_bands, n_permutations, n_gram):
        """Create Minhash and Locality Sensitive Hashing (LSH) to detect near duplicate texts.

        Args:
            content (list): List with string to build LSH.
            no_of_bands (int): Number of bands to break minhash signature into before hashing into buckets.
            n_permutations (int): Number of permutations used to create minhash signatures used in LSH model.
            n_gram (int): Size of each overlapping text shingle to break text into prior to hashing.
            no_of_bands(int): Number of bands to break minhash signature into before hashing into buckets.

        Returns:
            class 'snapy.lsh.LSH':  Snapy LSH object.

        """
        labels = range(len(content))

        # Create MinHash object.
        minhash = MinHash(content, n_gram=n_gram, permutations=n_permutations, hash_bits=64, seed=SEED)

        # Create LSH model.
        lsh = LSH(minhash, labels, no_of_bands=no_of_bands)

        return lsh


    def find_near_duplicate(self, query_sentences, sentences, min_jaccard_value, no_of_bands, n_permutations, n_gram):
        """Using LSH object finds the near duplicate strings.

        Args:
            query_sentences (dict): Dict with query strings and version of string in lower case and without comma.
            sentences (dict): Dict with target strings and version of string in lower case and without comma.
            min_jaccard_value (float): Minimum value for the Jaccard Distance.
            no_of_bands (int): Number of bands to break minhash signature into before hashing into buckets.
            n_permutations (int): Number of permutations used to create minhash signatures used in LSH model.
            n_gram (int): Size of each overlapping text shingle to break text into prior to hashing.

        """

        content = list(sentences.keys())
        lsh = self.create_lsh(content, no_of_bands, n_permutations, n_gram)

        i = 0
        seen = set()
        with open('lsh_autophrase_output.txt', 'w') as f1:
            for index_query, search_string in enumerate(query_sentences):
                closest_results = lsh.query(i, min_jaccard=min_jaccard_value)
                if not i in seen and len(closest_results) > 1:
                    print("{} QUERY: {}".format(index_query + 1, query_sentences[search_string]))
                    # f1.write("{} QUERY: {} \n".format(index_query + 1, query_sentences[search_string]))
                    score0 = query_sentences[search_string].split('\t')[1]
                    tuple0 = (search_string, score0)
                    f1.write(':'.join(tuple0))
                    f1.write(',')

                    for content_index in closest_results:
                        result = content[content_index]
                        if result != search_string:
                            print(sentences[result])
                            # f1.write(sentences[result] + "\n")
                            score_i = sentences[result].split('\t')[1]
                            tuple_i = (result, score_i)
                            f1.write(':'.join(tuple_i))
                            f1.write(',')
                        seen.add(content_index)
                    print()
                    f1.write("\n")
                    seen.add(i)
                i += 1


    def parse_args(self):
        """Parse args entered by the user.

        Returns:
            argparse.Namespace: Parsed arguments.

        """
        parser = argparse.ArgumentParser(
            description="Detect near duplicate texts using Minhash and Locality Sensitive Hashing.",
            epilog="example > python3 find_near_duplicate.py  -q INPUT -t TARGERS")
        parser.add_argument("-q", "--query", help="Path to file with sentences to query", required=True)
        parser.add_argument("-t", "--targets", help="Path to file with sentences be matched against", required=True)
        parser.add_argument("-g", "--n_gram", help="Size of each overlapping text shingle to break text into "
                                                   "prior to hashing", default=2)
        parser.add_argument("-p", "--n_permutations", help="Number of permutations used to create minhash signatures used "
                                                           "in LSH model.", default=100)
        parser.add_argument("-j", "--min_jaccard", help="Jaccard similarity threshold texts have to exceed to be "
                                                        "returned as similar.", default=0.25)
        parser.add_argument("-b", "--no_of_bands", help="Number of bands to break minhash signature into "
                                                        "before hashing into buckets..", default=50)
        return parser.parse_args()


    def main(self):
        args = self.parse_args()

        query = args.query
        targets = args.targets
        min_jaccard_value = float(args.min_jaccard)
        n_gram = int(args.n_gram)
        n_permutations = int(args.n_permutations)
        no_of_bands = int(args.no_of_bands)

        # load sentences from file (v2 is the autophrase version)
        query_sentences = self.load_content_v3(self.phrases_lsh)
        targets_sentences = self.load_content_v3(self.phrases_lsh)

        # find near duplicate sequences to `search_string`
        self.find_near_duplicate(query_sentences, targets_sentences, min_jaccard_value, no_of_bands, n_permutations, n_gram)


if __name__ == "__main__":
    phrases_input = Phrases().phrases
    LSH_Autophrase(phrases_input).main()
