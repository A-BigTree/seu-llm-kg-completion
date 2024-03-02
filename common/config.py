from enum import Enum

import yaml

with open("../config.yml", "r", encoding="utf-8") as f:
    CONFIG = yaml.load(f.read(), Loader=yaml.FullLoader)

# datasets config
DATASETS_TYPE: str = CONFIG["datasets"]["type"]
DATASETS_PATH: str = CONFIG["datasets"]["path"]
# solr config
SOLR_HOST: str = CONFIG["solr"]["host"]
SOLR_CORES: list = CONFIG["solr"]["cores"]
# Solr update
SOLR_UPDATE_DATA_DIR: str = CONFIG["solr"]["update"]["data-dir"]
SOLR_UPDATE_URL: str = CONFIG["solr"]["update"]["update-url"]
SOLR_UPDATE_QUEUE_SIZE: int = CONFIG["solr"]["update"]["queue-size"]
SOLR_UPDATE_PRODUCE_THREAD: int = CONFIG["solr"]["update"]["produce-thread"]
SOLR_UPDATE_CONSUMER_THREAD: int = CONFIG["solr"]["update"]["consume-thread"]
# Solr query
SOLR_QUERY_URL: str = CONFIG["solr"]["query"]["query-url"]
SOLR_QUERY_PARAMS: dict = CONFIG["solr"]["query"]["query-params"]

