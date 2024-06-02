#!/usr/bin/env python3
"""Write the function that calculates the n-gram BLEU score for a sentence"""


import numpy as np


def transform_grams(references, sentence, n):
    """
    Transforms references and sentence based on grams
    """
    if n == 1:
        return references, sentence

    ngram_sentence = []
    sentence_length = len(sentence)

    for i, word in enumerate(sentence):
        count = 0
        w = word
        for j in range(1, n):
            if sentence_length > i + j:
                w += " " + sentence[i + j]
                count += 1
        if count == j:
            ngram_sentence.append(w)

    ngram_references = []

    for ref in references:
        ngram_ref = []
        ref_length = len(ref)

        for i, word in enumerate(ref):
            count = 0
            w = word
            for j in range(1, n):
                if ref_length > i + j:
                    w += " " + ref[i + j]
                    count += 1
            if count == j:
                ngram_ref.append(w)
        ngram_references.append(ngram_ref)

    return ngram_references, ngram_sentence


def ngram_bleu(references, sentence, n):
    """
    Calculates the n-gram BLEU score for a sentence

    parameters:
        references [list]:
            contains reference translations
        sentence [list]:
            contains the model proposed sentence
        n [int]:
            the size of the n-gram to use for evaluation

    returns:

