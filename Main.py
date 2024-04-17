import os
import sys
from random import choice

object_path = os.path.join(os.path.abspath(os.path.dirname(os.path.dirname(__file__))))
sys.path.append(object_path)


from tasks.train import *
from tasks.task import *


if __name__ == '__main__':
    # task = SolrInitTask()
    # task = GATTrainTask()
    # task = PreProcessTask()
    # task = GPTRequestTask()
    # task = PostProcessTask()
    # task = TextEmbeddingTask()
    task = MFTrainTask()
    task.run_task()
