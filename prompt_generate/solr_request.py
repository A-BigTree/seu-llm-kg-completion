import gc
import threading
import urllib.parse
from queue import Queue

import requests
from datasets import load_from_disk
from tqdm import tqdm

# Solr请求地址
SOLR_HOST = "http://localhost:8983/solr/"
DATASET_CORE_TEST = "wiki_solr"
# Solr请求核心列表
DATASET_CORE_LIST = ["wiki0", "wiki1", "wiki2", "wiki3", "wiki4"]
# Solr查询参数
SOLR_QUERY_URL = "/select?%s"
# Solr更新参数
SOLR_UPDATE_URL = "/update?commit=true"


def query_data(core: str, query: str) -> list:
    url = SOLR_HOST + core + SOLR_QUERY_URL % query
    response = requests.get(url)
    response_json = response.json()
    response_docs = response_json["response"]["docs"]
    return response_docs


def query_data_from_list(query: str) -> list:
    result = []
    for core in DATASET_CORE_LIST:
        result.extend(query_data(core, query))
    return result


def update_data_test(docs: list):
    url = SOLR_HOST + DATASET_CORE_TEST + SOLR_UPDATE_URL
    headers = {'Content-Type': 'application/json'}
    response = requests.post(url, json=docs, headers=headers)
    return response


def update_data(docs: list, index: int):
    url = SOLR_HOST + DATASET_CORE_LIST[index] + SOLR_UPDATE_URL
    headers = {'Content-Type': 'application/json'}
    response = requests.post(url, json=docs, headers=headers)
    return response


class MultiSolrReqeust(object):
    def __init__(self, queue_size: int = 5, produce_thread: int = 1, consumer_thread: int = 5):
        self.queue = Queue(queue_size)
        self.result_queue = Queue(queue_size)
        self.produce_thread = produce_thread
        self.consumer_thread = consumer_thread

    def __produce(self):
        data = load_from_disk("./data/datasets/wikipedia")
        sum_ = 1
        cache = []
        index_ = 0
        for content in tqdm(data["train"]):
            for t in content["text"].split("."):
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

    def __consume(self):
        while True:
            try:
                docs, index = self.queue.get(timeout=60)
                update_data(docs, index)
                # update_data_test(docs)
            except Exception as e:
                break

    def run_init(self):
        p_t = threading.Thread(target=self.__produce)
        p_t.start()
        c_ts = []
        for i in range(self.consumer_thread):
            c_t = threading.Thread(target=self.__consume)
            c_ts.append(c_t)
            c_t.start()
        p_t.join()
        for c_t in c_ts:
            c_t.join()

    def __consume_query(self):
        while not self.queue.empty():
            query, core = self.queue.get()
            result = query_data(core, query)
            self.result_queue.put((core, result))

    def run_query(self, query: dict, cores: list = None) -> list:
        if cores is None:
            cores = DATASET_CORE_LIST
        self.queue = Queue(len(cores))
        self.result_queue = Queue(len(cores))
        query_str = urllib.parse.urlencode(query)
        for core in cores:
            self.queue.put((query_str, core))
        c_ts = []
        for _ in range(self.consumer_thread):
            c_t = threading.Thread(target=self.__consume_query)
            c_t.start()
            c_ts.append(c_t)
        for t in tqdm(c_ts):
            t.join()
        result = []
        while not self.result_queue.empty():
            result.extend(self.result_queue.get()[1])
        return result


if __name__ == '__main__':
    pass
