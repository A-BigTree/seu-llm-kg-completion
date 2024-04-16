import os
import sys

object_path = os.path.join(os.path.abspath(os.path.dirname(os.path.dirname(__file__))))
sys.path.append(object_path)

# from tasks.train import *
# from tasks.task import *


class PostProcessTask:
    def __init__(self):
        self.path = "./data/pre"
        self.dataset = "DB15K"
        self.target_path = self.path + "/" + self.dataset + "/txt/"
        self.files = []

    def exc_input(self):
        files = os.listdir(self.target_path)
        for file in files:
            if file.endswith("_all_relation.txt"):
                self.files.append(file)
        for file in self.files:
            result = []
            with open(self.target_path + file, "r", encoding="utf-8") as f:
                lines = f.readlines()
                for line in lines:
                    if line.find("<H>") != -1 and line.find("<T>") != -1:
                        result.append(line)
            result.sort(key=lambda x: len(x), reverse=False)
            with open(self.target_path + "post/" + file, "w", encoding="utf-8") as f:
                f.writelines(result)

    def run(self):
        self.exc_input()



if __name__ == '__main__':
    # task = SolrInitTask()
    # task = GATTrainTask()
    # task = PreProcessTask()
    # task = GPTRequestTask()
    # task = TextEmbeddingTask()
    # task = MFTrainTask()
    # task.run_task()
    task = PostProcessTask()
    task.run()
