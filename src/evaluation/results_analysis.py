import os
import sys
from sklearn.metrics import  precision_recall_fscore_support
import configparser
import numpy as np
import pandas as pd
from utils import read_json, write_json
PACKAGE_PARENT = '.'
SCRIPT_DIR = os.path.dirname(os.path.realpath(os.path.join(os.getcwd(), os.path.expanduser(__file__))))
sys.path.append(os.path.normpath(os.path.join(SCRIPT_DIR, PACKAGE_PARENT)))
PREFIX_PATH = "/".join(os.path.dirname(os.path.abspath(__file__)).split("/")[:-1]) + "/"


def error_analysis(ground_truths, preds, labels):
    """False predicitons analysis

    Args:
        ground_truths (list): ground truth labels
        preds (list): predicted labels
        labels (list): target labels

    Returns:
        int, int: false positives and false negatives
    """
    tp, fp, fn = 0, 0, 0
    
    if type(preds[0]) == dict:
        preds = [pred.values()for pred in preds]
        preds = list(preds[0])
    # print("preds", type(preds))
    ground_truths = [ground.split(" ")[-1].split(":")[-1].strip() for ground in ground_truths]
    preds = [pred.split(":")[-1].strip() for pred in preds]
    for i, truth in enumerate(ground_truths):
        # print("truth", preds[i])

        if truth == preds[i]:
            tp += 1
        elif preds[i] in labels:
            fp += 1
        else:
            fn += 1

    return fp, fn

def get_results(preds, grounds, targets):
    """Compute precision, recall and f1 scores"""

    if type(preds[0]) == dict:
        preds = [pred.values() for pred in preds]
        preds = list(preds[0])
    # print("preds", preds)
    preds = [pred.split(":")[-1].strip() for pred in preds]
    
    grounds = [ground.split(" ")[-1].split(":")[-1].strip() for ground in grounds]
    prec, recall, f1, s = precision_recall_fscore_support(grounds, preds, labels=targets, average='micro')
       
    return prec, recall, f1

def compute_scores(predictions, ground_truths, labels):
    """Compute precision, recall and f1 scores"""
    labels = list(set(labels))
    prec, recall, f1 = get_results(predictions, ground_truths, labels)
    
    return prec, recall, f1, predictions
            
if __name__ == "__main__":
  
    config = configparser.ConfigParser()
    config.read(PREFIX_PATH+"config.ini")
    prompt_type = config["SETTINGS"]["prompt_type"]

    if prompt_type == "rag":
        prediction_path = config["OUTPUT"]["rag_test_responses_path"]
        result_path = config["OUTPUT"]["rag_test_results_path"]
        error_path = config["OUTPUT"]["rag_test_error_analysis_path"]
    else:
        prediction_path = config["OUTPUT"]["simple_prompt_responses_path"]
        result_path = config["OUTPUT"]["simple_prompt_results_path"]
        error_path = config["OUTPUT"]["simple_prompt_error_analysis_path"]
    
    ground_truths_path = config["PATH"]["test_ground_truth_path"]
    labels = config["PATH"]["relations_path"]
  
    labels = read_json(labels).keys()
    predictions = read_json(prediction_path)
    ground_truths = read_json(ground_truths_path)
    prec, recall, f1, preds = compute_scores(predictions, ground_truths, labels)

    result_metrics = {"Precision":[prec],
                      "Recall":[recall],
                      "F1":[f1]
                      }
    
    result_df = pd.DataFrame(result_metrics)
    result_df.to_json(result_path)

    fp, fn = error_analysis(ground_truths, preds, labels)
    error_analysis = {"False Positives":[fp],
                      "False Negatives":[fn]}
    error_df = pd.DataFrame(error_analysis)
    error_df.to_json(error_path)