""" This script is used to compute the sentence embeddings for the sentences in the dataset."""
"""Created by: Sefika"""
import json
from sentence_transformers import SentenceTransformer
import numpy as np
import configparser



def read_json(path):
    with open(path, 'r') as f:
        data = json.load(f)
    return data

def compute_sentence(data):
    sent_embeddings = []
    model = SentenceTransformer('all-MiniLM-L6-v2')
    print("The embeddings will be compted for {0} sentences".format(len(data)))
    for i, line in enumerate(data):
        sent = " ".join(line['tokens'])
        clean_sent = clean_sentence(sent)
        embeddings = model.encode(clean_sent)
        sent_embeddings.append(embeddings)
        print("Processed sentence: ", i)
    print("The embeddings were completed for {0} sentences".format(len(sent_embeddings)))
    return sent_embeddings

def clean_sentence(sent):
    
    sent = sent.replace("<e1>", "")
    sent = sent.replace("</e1>", "")
    sent = sent.replace("<e2>", "")
    sent = sent.replace("</e2>", "")

    return sent
def write_embeddings(embeddings, output_file):
    np.save(output_file, embeddings)

if __name__ == "__main__":

    config = configparser.ConfigParser()
    config = config.read("/Users/sefika/LLM-relation-extraction/config.ini")
    input_file = config["EMBEDDING"]["input_embedding_path"]
    output_file = config["EMBEDDING"]["output_embedding_path"]
    data = read_json(input_file)
    embeddings = compute_sentence(data)
    write_embeddings(embeddings, output_file)