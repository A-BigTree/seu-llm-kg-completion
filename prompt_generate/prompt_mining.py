from datasets import load_dataset
from tqdm import tqdm
import random
import nltk
import os

# nltk.download("words", download_dir="./data/nltk")
# words = set(nltk.corpus.words.words())


def get_wiki_corpus():
    wiki_dataset = load_dataset(
        "wikipedia",
        "20220301.en",
        cache_dir="./data/datasets",
        trust_remote_code=True,
    )
    wiki_dataset = wiki_dataset['train']

    corpus_text = ""
    sum = 0

    w_text = open("E:/wiki/dataset/corpus_text.txt", 'w', encoding='utf-8')
    print(corpus_text, file=w_text)
    w_text = open("E:/wiki/dataset/corpus_text.txt", 'a', encoding='utf-8')

    for context in tqdm(wiki_dataset):
        split_text = context['text'].split('.')
        for t in split_text:
            line = t + '.\n'
            corpus_text = corpus_text + line
        if len(corpus_text) > 10000:
            print(corpus_text, file=w_text)
            del corpus_text
            corpus_text = ""
            sum += 1

    print(f"Write count: {sum}")


if __name__ == '__main__':
    get_wiki_corpus()
