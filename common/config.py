import yaml
import logging

with open("./config.yml", "r", encoding="utf-8") as file:
    CONFIG = yaml.load(file.read(), Loader=yaml.FullLoader)

# logging config
LOGGING_CONFIG: dict = CONFIG["logging"]

logging.basicConfig(level=LOGGING_CONFIG["level"],
                    format=LOGGING_CONFIG["format"])
LOG_TASK = logging.getLogger("Task-Logger")
LOG_UTIL = logging.getLogger("Util-Logger")
LOG_TRAIN = logging.getLogger("Train-Logger")

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
# data config
DATASETS_PATH: str = CONFIG["datasets"]["path"]
DATASETS_TYPE: list = CONFIG["datasets"]["types"]
# Train config
TRAIN_CONFIG: dict = CONFIG["training"]
