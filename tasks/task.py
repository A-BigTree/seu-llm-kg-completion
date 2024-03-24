import json
import re
from queue import Empty

from common.config import *
from common.base import *
from datasets import load_from_disk
from tqdm import tqdm
import gc
import requests
import urllib.parse


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
            relation = dict()
            with open(self.path + dataset + f"/relation2id.txt", "r", encoding="utf-8") as f:
                for line in f.readlines():
                    line = line.strip()
                    if line == "":
                        continue
                    line = line.split(" ")
                    relation[line[0]] = line[1]
            wiki_croup = dict()
            reg = re.compile(r"/([^/>]+)>$")
            if dataset == DataSet.FB15K.value or dataset == DataSet.FB15K_237.value:
                with open(self.path + "entity2wikidata.json", "r", encoding="utf-8") as f:
                    wiki_croup = json.load(f)
            with open(self.path + dataset + f"/{dataset}_data.txt", "r", encoding="utf-8") as f:
                lines = f.readlines()
            result = []
            for key, _ in relation.items():
                result.append([])
            LOG_TASK.info(f"Start process {dataset} data.")
            for line in tqdm(lines):
                line = line.strip(" |.|\n")
                if line == "":
                    continue
                elems = line.split(" ")
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
                result[int(relation[elems[1]])].append(" ".join(elems) + "\n")
            for i in range(len(result)):
                with open(self.path + dataset + f"/relation/{i}_relation2entity.txt", "w", encoding="utf-8") as f:
                    f.writelines(result[i])
        with open(self.path + "error_data.txt", "w", encoding="utf-8") as f:
            f.writelines(error_data)


class GPTRequestTask(MultiThreadRequest):
    """Url request task."""

    pass


class SolrRecallTask(BaseModel):
    """Solr recall task."""

    pass
