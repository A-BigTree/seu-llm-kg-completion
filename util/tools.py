import json
import time

from tqdm import tqdm

from common.config import DATASETS_PATH, LOG_UTIL
import requests


def deal_with_error_data():
    query_wikidata_with_freebase = '''
        PREFIX wikibase: <http://wikiba.se/ontology#>
        PREFIX wd: <http://www.wikidata.org/entity/>
        PREFIX wdt: <http://www.wikidata.org/prop/direct/>
        PREFIX rdfs: <http://www.w3.org/2000/01/rdf-schema#>
        SELECT DISTINCT ?wd ?fb ?wdLabel ?wdDescription ?alternative ?sitelink
        WHERE {
          ?wd wdt:P646 ?fb .
          OPTIONAL { ?wd schema:description ?itemdesc . }
          OPTIONAL { ?wd skos:altLabel ?alternative . 
                       FILTER (lang(?alternative) = "en").
                     }
          OPTIONAL { ?sitelink schema:about ?wd . 
                       ?sitelink schema:inLanguage "en" .
                       FILTER (SUBSTR(str(?sitelink), 1, 25) = "https://en.wikipedia.org/") .
                     } .
          VALUES ?fb { "%s" }
          SERVICE wikibase:label { bd:serviceParam wikibase:language "en". }
        }'''
    error_data = []
    with open(DATASETS_PATH + "error_data.txt", "r", encoding="utf-8") as f:
        lines = f.readlines()
        for line in lines:
            error_data.append(line.strip(" |\n"))
    cache = []
    result = {}
    while True:
        if len(cache) > 0:
            error_data = cache
            cache = []
        for entities in tqdm(zip(*(iter(error_data),) * 10)):
            query = query_wikidata_with_freebase % "\"\"".join(entities)
            url = "https://query.wikidata.org/bigdata/namespace/wdq/sparql"
            try:
                response = requests.get(url, params={"query": query, "format": "json"})
            except Exception as e:
                LOG_UTIL.error(f"Error data: {entities}")
                cache.extend(entities)
                time.sleep(5)
                continue
            if response.status_code != 200:
                LOG_UTIL.error(f"Error status code: {response.status_code}")
                LOG_UTIL.error(f"Error data: {entities}")
                cache.extend(entities)
                time.sleep(5)
                continue
            response_json = response.json()
            for item in response_json["results"]["bindings"]:
                fb = item['fb']['value']
                label = item['wdLabel']['value'] if 'wdLabel' in item else None
                desc = item['wdDescription']['value'] if 'wdDescription' in item else None
                alias = {item['alternative']['value']} if 'alternative' in item else list()

                if fb not in result:
                    result[fb] = {'label': label,
                                  'description': desc,
                                  'alternatives': alias}

        if len(cache) == 0:
            break
    LOG_UTIL.info(result)
    with open(DATASETS_PATH + "deal_error.json", "w", encoding="utf-8") as f:
        json.dump(result, f, ensure_ascii=False, indent=4)
