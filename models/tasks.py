from common.config import *
from common.models import *
from datasets import load_from_disk
from tqdm import tqdm
import gc
import requests


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
        for content in tqdm(data["train"]):
            for t1 in content["text"].split("\n"):
                if t1.strip() == "":
                    continue
                for t2 in t1.split("."):
                    if len(t2.strip().split(" ")) < 4:
                        continue
                    cache.append({"id": sum_, "entity_id": int(content['id']), "value": t2})
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
