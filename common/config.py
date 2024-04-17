import yaml
import logging
import datetime

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
# OpenAI GPT config
GPT_URL: str = CONFIG["gpt"]["api-url"]
GPT_API_KEY: str = CONFIG["gpt"]["api-key"]
GPT_PROXIES: dict = CONFIG["gpt"]["proxy"]
GPT_MODEL: str = CONFIG["gpt"]["model"]
GPT_PROMPT: str = CONFIG["gpt"]["prompt"]
RELATION_EXAMPLE_NUM: int = CONFIG["gpt"]["relation-example-num"]
GPT_REQUEST_TIMEOUT: int = CONFIG["gpt"]["request-timeout"]
GPT_REQUEST_THREAD: int = CONFIG["gpt"]["request-thread"]
# Train config
TRAIN_CONFIG: dict = CONFIG["training"]
# text embedding config
TEXT_EMBEDDING_DATA_DIR: str = CONFIG["text-embedding"]["data-dir"]
TEXT_EMBEDDING_SAVE_DIR: str = CONFIG["text-embedding"]["save-dir"]
TEXT_EMBEDDING_DATASET: str = CONFIG["text-embedding"]["dataset"]
TEXT_EMBEDDING_MODEL: str = CONFIG["text-embedding"]["model"]
TEXT_EMBEDDING_TOKENIZER: str = CONFIG["text-embedding"]["tokenizer"]
TEXT_EMBEDDING_MAX_LENGTH: int = CONFIG["text-embedding"]["max-length"]

if CONFIG["tasks"]["model-train"]:
    formatter = logging.Formatter(LOGGING_CONFIG["format"])
    file_handler = logging.FileHandler(f"./data/record/{datetime.date.today()}-{TRAIN_CONFIG['model']}-{TRAIN_CONFIG['dataset']}.txt")
    file_handler.setFormatter(formatter)
    LOG_TRAIN.addHandler(file_handler)
