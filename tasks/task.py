import json
import os
import re
import string
from queue import Empty

from common.config import *
from common.base import *
from util.data_util import read_relation_from_id, read_entity_from_id
from datasets import load_from_disk
from tqdm import tqdm
import gc
import requests
import urllib.parse
from collections import Counter
from random import choice


class SolrInitTask(MultiThreadRequest):
    """Solr init task."""

    def __init__(self):
        super().__init__(name_="Solr Data InitTask",
                         queue_size=SOLR_UPDATE_QUEUE_SIZE,
                         produce_thread=SOLR_UPDATE_PRODUCE_THREAD,
                         consumer_thread=SOLR_UPDATE_CONSUMER_THREAD)

    def produce(self, *args, **kwargs):
        data = load_from_disk(SOLR_UPDATE_DATA_DIR)
        sum_ = 1
        cache = []
        index_ = 0
        split = re.compile(r"[.;?!\n]")
        sub = re.compile(r"[()\[\]\"\\/]")
        for content in tqdm(data["train"]):
            for t1 in split.split(content["text"]):
                t = sub.sub(" ", t1)
                t = t.strip(r"[ '\",:/\\\(\)\[\]]")
                if len(t.split(" ")) < 4:
                    continue
                cache.append({"id": sum_, "entity_id": int(content['id']), "value": t})
                sum_ += 1
                if len(cache) > 10000:
                    self.queue.put((cache, index_ % 5))
                    cache = []
                    index_ += 1
                if sum_ % 100000 == 0:
                    gc.collect()
        if len(cache) > 0:
            self.queue.put((cache, index_ % 5))

    def consume(self, *args, **kwargs):
        while True:
            try:
                docs, index = self.queue.get(timeout=10)
                url = SOLR_HOST + SOLR_CORES[index] + SOLR_UPDATE_URL
                headers = {'Content-Type': 'application/json'}
                requests.post(url, json=docs, headers=headers)
            except Empty:
                break


class SolrQueryTask(MultiThreadRequest):
    """Solr query task."""

    def __init__(self):
        super().__init__(name_="Solr Data Query Task",
                         input_=True,
                         cost_time=True,
                         queue_size=len(SOLR_CORES),
                         produce_thread=0,
                         consumer_thread=len(SOLR_CORES))

    def exec_input(self, *args, **kwargs):
        query_str = urllib.parse.urlencode(kwargs["query"])
        for core in SOLR_CORES:
            self.queue.put((core, query_str))

    def produce(self, *args, **kwargs):
        pass

    def consume(self, *args, **kwargs):
        while not self.queue.empty():
            core, query = self.queue.get()
            url = SOLR_HOST + core + SOLR_QUERY_URL % query
            response = requests.get(url)
            response_json = response.json()
            response_docs = response_json["response"]["docs"]
            self.result_queue.put((core, response_docs))

    def exec_output(self) -> list:
        result = []
        while not self.result_queue.empty():
            core, response_docs = self.result_queue.get()
            result.extend(response_docs)
        return result


class PreProcessTask(BaseModel):
    """Pre process task."""

    def __init__(self):
        super().__init__(name_="Pre Process Task",
                         input_=False,
                         output_=False,
                         cost_time=True)
        self.datasets = DATASETS_TYPE
        self.path = DATASETS_PATH

    def exec_process(self, *args, **kwargs):
        error_data = set()
        for dataset in self.datasets:
            path = self.path + dataset + "/"
            relation = read_relation_from_id(path)
            wiki_croup = dict()
            reg = re.compile(r"/([^/>]+)>$")
            if dataset == DataSet.FB15K.value or dataset == DataSet.FB15K_237.value:
                with open(self.path + "entity2wikidata.json", "r", encoding="utf-8") as f:
                    wiki_croup = json.load(f)
            with open(self.path + dataset + f"/{dataset}_data.txt", "r", encoding="utf-8") as f:
                lines = f.readlines()
            entity2id = read_entity_from_id(path)
            result = []
            for key, _ in relation.items():
                result.append([])
            LOG_TASK.info(f"Start process {dataset} data.")
            for line in tqdm(lines):
                line = line.strip(" |.|\n")
                if line == "":
                    continue
                elems = line.split(" ")
                index = [entity2id[elems[0]], entity2id[elems[2]]]
                if dataset == DataSet.FB15K.value or dataset == DataSet.FB15K_237.value:
                    if elems[0] in wiki_croup:
                        elems[0] = wiki_croup[elems[0]]["label"].replace(" ", "_")
                    else:
                        error_data.add(elems[0] + "\n")
                        continue
                    if elems[2] in wiki_croup:
                        elems[2] = wiki_croup[elems[2]]["label"].replace(" ", "_")
                    else:
                        error_data.add(elems[2] + "\n")
                        continue
                else:
                    e1 = reg.search(elems[0])
                    elems[0] = e1.group(1)
                    e2 = reg.search(elems[2])
                    elems[2] = e2.group(1)
                result[int(relation[elems[1]])].append(" ".join(elems + index) + "\n")
            for i in range(len(result)):
                with open(self.path + dataset + f"/relation/{i}_relation2entity.txt", "w", encoding="utf-8") as f:
                    f.writelines(result[i])
        with open(self.path + "error_data.txt", "w", encoding="utf-8") as f:
            f.writelines(error_data)


class GPTRequestTask(MultiThreadRequest):
    """Url request task."""

    def __init__(self):
        super().__init__(name_="GPT Request Task",
                         input_=False,
                         queue_size=GPT_REQUEST_THREAD,
                         produce_thread=0,
                         consumer_thread=GPT_REQUEST_THREAD)
        self.url = GPT_URL
        self.api_key = GPT_API_KEY
        self.model = GPT_MODEL
        self.proxies = GPT_PROXIES
        self.path = DATASETS_PATH
        self.datasets = DATASETS_TYPE
        self.relation_example_num = RELATION_EXAMPLE_NUM
        self.header = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json"
        }

    def __choose_example(self, examples: list):
        tails = []
        data = []
        for example in examples:
            elems = example.split(" ")
            tails.append(elems[2])
            data.append(elems[:5])
        counter = Counter(tails)
        choose_tails = counter.most_common(self.relation_example_num)
        result = []
        i = 0
        for tail, _ in choose_tails:
            heads = []
            for da in data:
                if da[2] == tail:
                    heads.append(da[0])
            short_head = min(heads, key=lambda x: len(x))
            result.append((i, short_head, tail))
            i += 1
        return result

    def produce(self, *args, **kwargs):
        for dataset in self.datasets:
            LOG_TASK.info(f"Start process {dataset} data.")
            path = self.path + dataset + "/"
            relation = read_relation_from_id(path)
            for _, value in tqdm(relation.items()):
                examples = []
                with open(path + f"relation/{value}_relation2entity.txt", "r", encoding="utf-8") as f:
                    lines = f.readlines()
                for line in lines:
                    examples.append(line.strip(" |\n"))
                choose_examples = self.__choose_example(examples)
                self.result_queue.put((dataset, value, len(choose_examples)))
                for example in choose_examples:
                    self.queue.put((dataset, value, example[0], example[1], example[2]))

    def consume(self, *args, **kwargs):
        while True:
            try:
                dataset, relation, index, head, tail = self.queue.get(timeout=5)
            except Empty:
                break
            boby = {
                "model": self.model,
                "messages": [{
                    "role": "user",
                    "content": GPT_PROMPT % (head, tail, head, tail)
                }]
            }
            try:
                response = requests.post(self.url,
                                         headers=self.header,
                                         json=boby,
                                         proxies=self.proxies,
                                         timeout=GPT_REQUEST_TIMEOUT)
                if response.status_code != 200:
                    LOG_TASK.error(f"Error status code: {response.status_code}")
                    self.cache_queue.put((dataset, relation, index, head, tail))
                    time.sleep(5)
                    continue
                response_json = response.json()
                result = response_json["choices"][0]["message"]["content"]
                with open(self.path + dataset + f"/txt/{relation}_{index}_relation.txt", "w", encoding="utf-8") as f:
                    f.write(result + "\n")
            except Exception as e:
                print(e)
                self.cache_queue.put((dataset, relation, index, head, tail))
                time.sleep(5)
                continue

    def exec_output(self):
        while not self.result_queue.empty():
            dataset, relation, num = self.result_queue.get()
            result = []
            for i in range(num):
                with open(self.path + dataset + f"/txt/{relation}_{i}_relation.txt", "r", encoding="utf-8") as f:
                    for line in f.readlines():
                        line = line.strip(" |\n|.").strip(string.digits).strip(" |\n|.")
                        if line == "":
                            continue
                        result.append(line + "\n")
            with open(self.path + dataset + f"/txt/{relation}_all_relation.txt", "w", encoding="utf-8") as f:
                f.writelines(result)


class PostProcessTask(BaseModel):
    def __init__(self, dataset="DB15K"):
        super().__init__(name_="Pre Process Task",
                         input_=True,
                         output_=False,
                         cost_time=True)
        self.path = DATASETS_PATH
        self.dataset = dataset
        self.target_path = self.path + self.dataset + "/txt/"
        self.files = []

    def exec_input(self):
        files = os.listdir(self.target_path)
        for fi in files:
            if fi.endswith("_all_relation.txt"):
                self.files.append(fi)
        for fi in self.files:
            result = []
            with open(self.target_path + fi, "r", encoding="utf-8") as f:
                lines = f.readlines()
                for line in lines:
                    if line.find("<H>") != -1 and line.find("<T>") != -1:
                        result.append(line)
            result.sort(key=lambda x: len(x), reverse=False)
            with open(self.target_path + "post/" + fi, "w", encoding="utf-8") as f:
                f.writelines(result)

    def exec_process(self, *args, **kwargs):
        path = self.path + self.dataset + "/"
        relation_id = read_relation_from_id(path)
        entity_id = read_entity_from_id(path)
        id_relation = dict()
        relation_txt = dict()
        for k, v in relation_id.items():
            id_relation[v] = k
            relation_txt[v] = []
        id_entity = dict()
        entity_txt = dict()
        for k, v in entity_id.items():
            id_entity[v] = k
            entity_txt[v] = []
        for i in tqdm(range(len(self.files))):
            entities = []
            with open(path + f"relation/{i}_relation2entity.txt", "r", encoding="utf-8") as f:
                lines = f.readlines()
                for line in lines:
                    elems = line.strip().split(" ")
                    entities.append((elems[0], elems[2], int(elems[3]), int(elems[4])))
            templates = dict()
            templates["head"] = []
            templates["tail"] = []
            templates["relation"] = []
            with open(self.target_path + f"post/{i}_all_relation.txt", "r", encoding="utf-8") as f:
                lines = f.readlines()
                for line in lines:
                    if line.strip() != "":
                        templates["head"].append(line.strip())
                        templates["tail"].append(line.strip())
                        templates["relation"].append(line.strip())
            if len(templates["head"]) == 0:
                with open(self.target_path + f"{i}_all_relation.txt", "r", encoding="utf-8") as f:
                    lines = f.readlines()
                    for line in lines:
                        if line.strip() == "":
                            continue
                        if line.find("<H>") != -1:
                            templates["head"].append(line.strip())
                            templates["relation"].append(line.strip())
                        if line.find("<T>") != -1:
                            templates["tail"].append(line.strip())
                            templates["relation"].append(line.strip())
            if len(templates["head"]) == 0:
                templates["head"].append("default")
                print(f"relation: {i} head template is empty")
            if len(templates["tail"]) == 0:
                templates["tail"].append("default")
                print(f"relation: {i} tail template is empty")
            for head, tail, h_i, t_i in entities:
                head_template = choice(templates["head"])
                entity_txt[h_i].append(head_template
                                       .replace("<H>", head
                                                .replace("_", " "))
                                       .replace("<T>", tail.
                                                replace("_", " ")) + ". ")
                tail_template = choice(templates["tail"])
                entity_txt[t_i].append(tail_template
                                       .replace("<H>", head
                                                .replace("_", " "))
                                       .replace("<T>", tail
                                                .replace("_", " ")) + ". ")
                relation_template = choice(templates["head"])
                relation_txt[i].append(relation_template
                                       .replace("<H>", head
                                                .replace("_", " "))
                                       .replace("<T>", tail
                                                .replace("_", " ")) + ". ")
        entities = []
        for i in range(len(entity_txt)):
            s = ""
            for txt in entity_txt[i]:
                s += txt
            entities.append(f"{i}\t{s}\n")
        relations = []
        for i in range(len(relation_txt)):
            s = ""
            for txt in relation_txt[i]:
                s += txt
            relations.append(f"{i}\t{s}\n")
        with open(path + "entity_text.txt", "w", encoding="utf-8") as f:
            f.writelines(entities)
        with open(path + "relation_text.txt", "w", encoding="utf-8") as f:
            f.writelines(relations)
