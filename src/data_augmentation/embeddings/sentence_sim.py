import os
import sys
import json
import numpy as np
from numpy.linalg import norm

PACKAGE_PARENT = '..'
SCRIPT_DIR = os.path.dirname(os.path.realpath(os.path.join(os.getcwd(), os.path.expanduser(__file__))))
sys.path.append(os.path.normpath(os.path.join(SCRIPT_DIR, PACKAGE_PARENT)))
PREFIX_PATH = "/".join(os.path.dirname(os.path.abspath(__file__)).split("/")[:-2]) + "/"


import configparser

def read_json(path):
    with open(path, 'r') as f:
        data = json.load(f)
    return data

def write_json(path, data):

    if not os.path.exists(os.path.dirname(path)):
        os.makedirs(os.path.dirname(path))
        
    with open(path, 'w', encoding="utf-8") as f:
        json.dump(data, f, indent=4, ensure_ascii=False)
        


def compute_similarity(test_data, train_data, train_embeddings, test_embeddings):
    similarities = []

    for test_index, _ in enumerate(test_data):
        test_emb = test_embeddings[test_index]
        train_similarities = []

        for train_index, train_line in enumerate(train_data):

            train_emb = train_embeddings[train_index]
            sim = np.dot(test_emb,train_emb)/(norm(test_emb)*norm(train_emb))
            train_sentence = " ".join(train_line['tokens'])
                
            context =  train_sentence
            train_similarities.append({"train":train_index, "simscore": sim, "sentence":context})

        train_similarities = sorted(train_similarities, key=lambda x: x["simscore"], reverse=True)
            
        similarities.append({"test":test_index, "similar_sentence":train_similarities[0]['sentence'],"train_idex":train_similarities[0]['train'], "simscore":float(train_similarities[0]['simscore'])})

        print("test index: ", test_index)

    return similarities


def semeval_compute_similarity(test_data, train_data, train_embeddings, test_embeddings):

    similarities = []

    for test_index, _ in enumerate(test_data):
        test_emb = test_embeddings[test_index]
        train_similarities = []

        for train_index, train_line in enumerate(train_data):
            train_emb = train_embeddings[train_index]
            sim = np.dot(test_emb,train_emb)/(norm(test_emb)*norm(train_emb))
            train_similarities.append({"train":train_index, "simscore":sim, "sentence":train_line})
        
        train_similarities = sorted(train_similarities, key=lambda x: x["simscore"], reverse=True)
            
        similarities.append({"test":test_index, "similar_sentence":train_similarities[0]['sentence'],"train_idex":train_similarities[0]['train'], "simscore":float(train_similarities[0]['simscore'])})

        print("test index: ", test_index)

    return similarities


def main(test_file, train_file, train_emb, test_emb, output_sim_path, dataset="semeval"):
    test_data = read_json(test_file)
    train_data = read_json(train_file)
    train_embeddings = np.load(train_emb)
    test_embeddings = np.load(test_emb)

    if dataset == "semeval":
        similarities = semeval_compute_similarity(test_data, train_data, train_embeddings, test_embeddings)
    else:
        similarities = compute_similarity(test_data, train_data, train_embeddings, test_embeddings, output_sim_path)

    write_json(similarities, output_sim_path)


if __name__ == "__main__":

    config = configparser.ConfigParser()
    config.read(PREFIX_PATH+"config.ini")

    test_file = config["SIMILARITY"]["test_file"]
    train_file = config["SIMILARITY"]["train_file"]
    train_emb = config["SIMILARITY"]["train_emb"]
    test_emb = config["SIMILARITY"]["test_emb"]
    output_sim_path = config["SIMILARITY"]["output_index"]
    main(test_file, train_file, train_emb, test_emb, output_sim_path)