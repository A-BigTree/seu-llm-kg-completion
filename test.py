from tqdm import tqdm

with open(
        "D:\ABigTree\Github\seu-llm-kg-completion\prompt_generate\data\datasets\mine_text\FB60K-NYT10\mined_text_business_company_founders.txt",
        "r", encoding="utf-8") as f:
    data = f.readlines()

head = "<COMPANY>[X]</COMPANY>"
tail = "<FOUNDER>[Y]</FOUNDER>"
pattens = ""
index = 0
for line in tqdm(data):
    index += 1
    if len(line) == 0:
        continue
    doc = line.replace("[X]", head).replace("[Y]", tail).replace("\n", "")
    pattens += doc + "\n"
    if index % 10000 == 0:
        with open("corpus.txt", "a", encoding="utf-8") as f:
            f.write(pattens)
        pattens = ""
with open("corpus.txt", "a", encoding="utf-8") as f:
    f.write(pattens)
