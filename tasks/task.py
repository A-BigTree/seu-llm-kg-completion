import re

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
                docs, index = self.queue.get(timeout=60)
                url = SOLR_HOST + SOLR_CORES[index] + SOLR_UPDATE_URL
                headers = {'Content-Type': 'application/json'}
                requests.post(url, json=docs, headers=headers)
            except Exception:
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
