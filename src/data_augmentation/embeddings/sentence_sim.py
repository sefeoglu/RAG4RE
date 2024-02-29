import os
import sys
import json
import numpy as np
from numpy.linalg import norm

sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from utils.utility import read_json, write_triplet_json
import configparser




def compute_similarity(test_data, train_data, train_embeddings, test_embeddings):


    similarities = []
    for test_index, _ in enumerate(test_data):
        test_emb = test_embeddings[test_index]
        train_similarities = []
        for train_index, train_line in enumerate(train_data):
            train_emb = train_embeddings[train_index]
            sim = np.dot(test_emb,train_emb)/(norm(test_emb)*norm(train_emb))
            train_sentence = train_line['sentence'] #" ".join(train_line['tokens'])

            head = train_line["subject"]
            tail = train_line["object"]
            relation = train_line["relation"]
                
            context = "Sentence:"+ train_sentence + "\n" + "Head: " + head + "\n" + "Tail: " + tail + "\n" + "Relation type: " + relation + "\n"
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

    write_triplet_json(similarities, output_sim_path)


if __name__ == "__main__":
    config = configparser.ConfigParser()
    config = config.read("/Users/sefika/LLM-relation-extraction/config.ini")

    test_file = config["SIMILARITY"]["test_file"]
    train_file = config["SIMILARITY"]["train_file"]
    train_emb = config["SIMILARITY"]["train_emb"]
    test_emb = config["SIMILARITY"]["test_emb"]
    output_sim_path = config["SIMILARITY"]["output_index"]
    metric = config["SIMILARITY"]["metric"]
    main(test_file, train_file, train_emb, test_emb, output_sim_path)