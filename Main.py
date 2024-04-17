import os
import sys
from random import choice

object_path = os.path.join(os.path.abspath(os.path.dirname(os.path.dirname(__file__))))
sys.path.append(object_path)


# from tasks.train import *
# from tasks.task import *


class PostProcessTask:
    def __init__(self):
        self.path = "./data/pre/"
        self.dataset = "DB15K"
        self.target_path = self.path + self.dataset + "/txt/"
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
        path = self.path + self.dataset + "/"
        relation_id = read_relation_from_id(path)
        entity_id = read_entity_from_id(path)
        id_relation = dict()
        relation_txt = dict()
        for k, v in relation_id.items():
            id_relation[v] = k
            relation_txt[v] = []
        id_entity = dict()
        entity_txt = dict()
        for k, v in entity_id.items():
            id_entity[v] = k
            entity_txt[v] = []
        for i in range(len(self.files)):
            entities = []
            with open(path + f"relation/{i}_relation2entity.txt", "r", encoding="utf-8") as f:
                lines = f.readlines()
                for line in lines:
                    elems = line.strip().split(" ")
                    entities.append((elems[0], elems[2], int(elems[3]), int(elems[4])))
            templates = dict()
            templates["head"] = []
            templates["tail"] = []
            templates["relation"] = []
            with open(self.target_path + f"post/{i}_all_relation.txt", "r", encoding="utf-8") as f:
                lines = f.readlines()
                for line in lines:
                    if line.strip() != "":
                        templates["head"].append(line.strip())
                        templates["tail"].append(line.strip())
                        templates["relation"].append(line.strip())
            if len(templates["head"]) == 0:
                with open(self.target_path + f"{i}_all_relation.txt", "r", encoding="utf-8") as f:
                    lines = f.readlines()
                    for line in lines:
                        if line.strip() == "":
                            continue
                        if line.find("<H>") != -1:
                            templates["head"].append(line.strip())
                            templates["relation"].append(line.strip())
                        if line.find("<T>") != -1:
                            templates["tail"].append(line.strip())
                            templates["relation"].append(line.strip())
            if len(templates["head"]) == 0:
                templates["head"].append("default")
                print(f"relation: {i} head template is empty")
            if len(templates["tail"]) == 0:
                templates["tail"].append("default")
                print(f"relation: {i} tail template is empty")
            for head, tail, h_i, t_i in entities:
                head_template = choice(templates["head"])
                entity_txt[h_i].append(
                    head_template.replace("<H>", head.replace("_", " ")).replace("<T>", tail.replace("_", " ")) + ".")
                tail_template = choice(templates["tail"])
                entity_txt[t_i].append(
                    tail_template.replace("<H>", head.replace("_", " ")).replace("<T>", tail.replace("_", " ")) + ".")
                relation_template = choice(templates["head"])
                relation_txt[i].append(relation_template.replace("<H>", head.replace("_", " ")).replace("<T>",
                                                                                                        tail.replace(
                                                                                                            "_",
                                                                                                            " ")) + ".")
        entities = []
        for i in range(len(entity_txt)):
            s = ""
            for txt in entity_txt[i]:
                s += txt
            entities.append(f"{i}\t{s}\n")
        relations = []
        for i in range(len(relation_txt)):
            s = ""
            for txt in relation_txt[i]:
                s += txt
            relations.append(f"{i}\t{s}\n")
        with open(path + "entity_text.txt", "w", encoding="utf-8") as f:
            f.writelines(entities)
        with open(path + "relation_text.txt", "w", encoding="utf-8") as f:
            f.writelines(relations)


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
