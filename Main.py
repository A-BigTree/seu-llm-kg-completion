import sys
import os

object_path = os.path.join(os.path.abspath(os.path.dirname(os.path.dirname(__file__))))
sys.path.append(object_path)

from models.tasks import SolrInitTask

if __name__ == '__main__':
    task = SolrInitTask()
    task.run_task()
