import os
import sys
from sklearn.metrics import  precision_recall_fscore_support
import configparser
import numpy as np
import pandas as pd
PACKAGE_PARENT = '.'
SCRIPT_DIR = os.path.dirname(os.path.realpath(os.path.join(os.getcwd(), os.path.expanduser(__file__))))
sys.path.append(os.path.normpath(os.path.join(SCRIPT_DIR, PACKAGE_PARENT)))
PREFIX_PATH = "/".join(os.path.dirname(os.path.abspath(__file__)).split("/")[:-1]) + "/"
from utils import read_json, write_json
def error_analysis(ground_truths, preds, labels):
    tp =0
    fp =0
    fn =0
    for i, truth in enumerate(ground_truths):
        if truth == preds[i]:
            tp +=1
        elif preds[i] in labels:
            fp += 1
        else:
            fn += 1

    return fp, fn



def get_results(preds, grounds, targets):

    prec, recall, f1, s = precision_recall_fscore_support(grounds, preds, labels=targets, average='micro')
       
    return prec, recall, f1

def compute_scores(predictions, ground_truths, labels):
    
    preds = []
    for i, truth in enumerate(ground_truths):

        if truth in predictions[i]:
        
            pred = truth
        elif len(predictions[i])>0:
             pred=predictions[i][0].replace(".","_")
        else:
            pred=""
        if pred == "no relation type":
            pred = "no_relation"

        preds.append(pred)
#     print(preds)
    prec, recall, f1 = get_results(preds, ground_truths, labels)
    

    
    return prec, recall, f1, preds
            
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
                      "F1":[f1]}
    
    result_df = pd.DataFrame(result_metrics)
    result_df.to_json(result_path)

    fp, fn = error_analysis(ground_truths, preds, labels)
    error_analysis = {"False Positives":[fp],
                      "False Negatives":[fn]}
    error_df = pd.DataFrame(error_analysis)
    error_df.to_json(error_path)