import gc
import random
import re

from datasets import load_dataset, load_from_disk
from enum import Enum
from tqdm import tqdm
from solr_request import update_data, query_data, MultiSolrReqeust
import csv
import os

FB60K_NYT10_RELATIONS = [
    '/people/person/nationality',
    '/location/location/contains',
    '/people/person/place_lived',
    '/people/person/place_of_birth',
    '/people/deceased_person/place_of_death',
    '/people/person/ethnicity',
    '/people/ethnicity/people',
    '/business/person/company',
    '/people/person/religion',
    '/location/neighborhood/neighborhood_of',
    '/business/company/founders',
    '/people/person/children',
    '/location/administrative_division/country',
    '/location/country/administrative_divisions',
    '/business/company/place_founded',
    '/location/us_county/county_seat'
]

UMLS_RELATIONS = [
    'gene_associated_with_disease',
    'disease_has_associated_gene',
    'gene_mapped_to_disease',
    'disease_mapped_to_gene',
    'may_be_treated_by',
    'may_treat',
    'may_be_prevented_by',
    'may_prevent',
]

BASE_PATH = "./data/datasets"


class DataSet(Enum):
    WIKI_PEDIA = "wikipedia"
    FB60K_NYT10 = "FB60K-NYT10"
    UMLS_PUB_MED = "UMLS-PubMed"


def get_wiki_dataset():
    """Download wikipedia datasets"""
    dataset = load_dataset("wikipedia", "20220301.en", trust_remote_code=True)
    dataset.save_to_disk("./data/datasets/wikipedia")


def get_wiki_corpus_to_text():
    """Save wikipedia datasets into txt file"""
    wiki_dataset = load_dataset(
        "wikipedia",
        "20220301.en",
        cache_dir="./data/datasets",
        trust_remote_code=True,
    )
    wiki_dataset = wiki_dataset['train']

    corpus_text = ""

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
    # Write count: 163316426


def get_wiki_corpus_to_csv():
    """Save wikipedia dataset into csv file"""
    data = load_from_disk("./data/datasets/wikipedia")
    sum_ = 0
    cache = []
    csv_file = "E:/wiki/dataset/data.csv"
    with open(csv_file, "w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f)
        header = ["id", "url", "title",  "text"]
        writer.writerow(header)
    for content in tqdm(data["train"]):
        cache.append([content['id'], content['url'], content['title'], content['text']])
        if len(cache) > 10000:
            with open(csv_file, "a", encoding="utf-8", newline="") as f:
                writer = csv.writer(f)
                writer.writerows(cache)
            del cache
            cache = []
        if sum_ % 100000 == 0:
            gc.collect()
        sum_ += 1
    if len(cache) > 0:
        with open(csv_file, "a", encoding="utf-8", newline="") as f:
            writer = csv.writer(f)
            writer.writerows(cache)
        del cache
        cache = []


def get_wiki_corpus_to_solr():
    """Save wikipedia datasets into solr database"""
    solr_request = MultiSolrReqeust(queue_size=10, consumer_thread=20)
    solr_request.run_init()


def query_data_from_solr(query: dict) -> list:
    """Query data from solr database"""
    task = MultiSolrReqeust()
    return task.run_query(query)


def get_triples_for_relation(relation: str, dataset: DataSet) -> str:
    """Get triples for relation"""
    if dataset == DataSet.FB60K_NYT10:
        origin_triples_path = BASE_PATH + "/triples_nyt10.txt"
    elif dataset == DataSet.UMLS_PUB_MED:
        origin_triples_path = BASE_PATH + "/triples_umls.txt"
    else:
        raise ValueError(f"Dataset {dataset} is not supported!")

    with open(origin_triples_path, "r", encoding="utf-8") as f:
        origin_triples = f.readlines()

    count = 0
    random.shuffle(origin_triples)
    random_selected_triples = ""
    for triple in origin_triples:
        if relation in triple:
            random_selected_triples += triple
            count += 1
    print(f"Total {count} triples for relation {relation}")
    relation_ = relation.replace("/", "_")
    with open(f"./data/datasets/relation_triples/{dataset.value}/{relation_}.txt", "w", encoding="utf-8") as f:
        f.write(random_selected_triples)
    return random_selected_triples


def get_entity_tokens(dataset: DataSet) -> dict:
    if dataset == DataSet.FB60K_NYT10:
        origin_triples_path = BASE_PATH + "/triples_nyt10.txt"
    elif dataset == DataSet.UMLS_PUB_MED:
        origin_triples_path = BASE_PATH + "/triples_umls.txt"
    else:
        raise ValueError(f"Dataset {dataset} is not supported!")
    with open(origin_triples_path, "r", encoding="utf-8") as f:
        origin_triples = f.readlines()
    entity_tokens = set()
    for triple in origin_triples:
        triple = triple.replace("\n", "").strip()
        if len(triple) == 0:
            continue
        entity_tokens.add(triple.split("\t")[0])
        entity_tokens.add(triple.split("\t")[2])
    entity_list = list(entity_tokens)
    result = {}
    for i in range(len(entity_list)):
        result[entity_list[i]] = i
    return result


def get_triple_text_from_corpus(triple: str, relation: str, dataset: DataSet, max_lines: int = 50000):
    lines = triple.split("\n")
    relation_ = relation.replace("/", "_")
    path = f"./data/datasets/mine_text/{dataset.value}/mined_text{relation_}.txt"
    file = open(path, "w", encoding="utf-8")
    print("", file=file)
    file = open(path, "a", encoding="utf-8")
    num_lines = 0

    task = MultiSolrReqeust()

    for line in tqdm(lines):
        if num_lines >= max_lines:
            break
        line = line.replace("\n", "").strip()
        if len(line) == 0:
            continue
        head = line.split("\t")[0].replace("_", " ").strip()
        tail = line.split("\t")[2].replace("_", " ").strip()
        query = {
            "q": f"value:(+\"{head}\" +\"{tail}\")",
            "fl": "value",
            "start": 0,
            "rows": 100
        }
        results = task.run_query(query)
        res = ""
        for result in results:
            if len(result["value"].replace("\n", "").strip()) == 0:
                continue
            doc = result["value"].replace("\n", " ").strip().lower()
            doc = re.sub(r"\b%s\b" % head, "[X]", doc)
            doc = re.sub(r"\b%s\b" % tail, "[Y]", doc)
            res += doc + "\n"
            num_lines += 1
        if len(res) > 0:
            print(res, file=file)
    print(f'Sum mined {num_lines} sentences')


def mine_text_from_corpus(dataset: DataSet):
    if dataset == DataSet.FB60K_NYT10:
        relations = FB60K_NYT10_RELATIONS
    elif dataset == DataSet.UMLS_PUB_MED:
        relations = UMLS_RELATIONS
    else:
        raise ValueError(f"Dataset {dataset} is not supported!")

    for relation in relations:
        print(f"Start mining relation {relation}")
        triples = get_triples_for_relation(relation, dataset)
        get_triple_text_from_corpus(triples, relation, dataset)


if __name__ == '__main__':
    # get_wiki_dataset()
    # get_wiki_corpus_to_csv()
    # get_wiki_corpus_to_solr()
    mine_text_from_corpus(DataSet.FB60K_NYT10)
