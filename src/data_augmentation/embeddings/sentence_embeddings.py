""" This script is used to compute the sentence embeddings for the sentences in the dataset."""
"""Created by: Sefika"""
import os
import sys
import json
from sentence_transformers import SentenceTransformer
import numpy as np
import configparser

PACKAGE_PARENT = '..'
SCRIPT_DIR = os.path.dirname(os.path.realpath(os.path.join(os.getcwd(), os.path.expanduser(__file__))))
sys.path.append(os.path.normpath(os.path.join(SCRIPT_DIR, PACKAGE_PARENT)))
PREFIX_PATH = "/".join(os.path.dirname(os.path.abspath(__file__)).split("/")[:-2]) + "/"

def read_json(path):
    """ Read json file"""
    with open(path, 'r') as f:
        data = json.load(f)
    return data

def write_json(path, data):
    """ Write json file"""
    
    if not os.path.exists(os.path.dirname(path)):
        os.makedirs(os.path.dirname(path))
        
    with open(path, 'w', encoding="utf-8") as f:
        json.dump(data, f, indent=4, ensure_ascii=False)
        
def compute_sentence(data):
    """Compute the sentence embeddings for the sentences in the dataset
    Args:
        data (list): list of sentences
    Returns:
        list: list of sentence embeddings
    """
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
    """Clean the sentence from the entity tags"""
    sent = sent.replace("<e1>", "")
    sent = sent.replace("</e1>", "")
    sent = sent.replace("<e2>", "")
    sent = sent.replace("</e2>", "")

    return sent

def write_embeddings(embeddings, output_file):
    np.save(output_file, embeddings)

if __name__ == "__main__":

    print("PREFIX_PATH", PREFIX_PATH)

    config = configparser.ConfigParser()
    config.read(PREFIX_PATH+"config.ini")

    input_file = config["EMBEDDING"]["input_embedding_path"]
    output_file = config["EMBEDDING"]["output_embedding_path"]
    data = read_json(input_file)
    embeddings = compute_sentence(data)
    write_embeddings(embeddings, output_file)