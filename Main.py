import sys
import os

object_path = os.path.join(os.path.abspath(os.path.dirname(os.path.dirname(__file__))))
sys.path.append(object_path)

from tasks.task import *

if __name__ == '__main__':
    task = SolrInitTask()
    task.run_task()
