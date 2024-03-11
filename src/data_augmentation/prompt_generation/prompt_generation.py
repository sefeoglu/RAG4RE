import os
import sys
import json

import re
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))


PACKAGE_PARENT = '.'
SCRIPT_DIR = os.path.dirname(os.path.realpath(os.path.join(os.getcwd(), os.path.expanduser(__file__))))
sys.path.append(os.path.normpath(os.path.join(SCRIPT_DIR, PACKAGE_PARENT)))
PREFIX_PATH = "/".join(os.path.dirname(os.path.abspath(__file__)).split("/")[:-1]) + "/"
from prompt_templates import get_zero_shot_template_tacred, get_zero_shot_template_tacred_rag, semeval_prompt_template_rag, semeval_prompt_template

def read_json(path):
    """Read json file"""

    with open(path, 'r') as f:
        data = json.load(f)

    return data

def tacred_format(test_data, relations, similar_sentences, type="rag"):
    """Regenerate prompt for tacred and its variants like tacrev, re-tacred

    Args:
        test_data (list): list of test data
        relations (list): list of relations (target labels)
        similar_sentences (list): list of similar sentence with corresponding test data
        type (str, optional): prompt type. Defaults to "rag".

    Returns:
        list: list of regenerated prompts
    """
    
    prompts = []
    relations = " ".join([relation for relation in relations])

    
    for index, line in enumerate(test_data):
        
        sentence = " ".join([ token for token in line['tokens']])
        head = line['subject']
        tail = line['object']

        if  type == "simple":
            prompt = get_zero_shot_template_tacred(sentence, relations, head, tail)
            data = {"prompt":prompt, "relation":line['relation']}
        else:
            context = similar_sentences[index]
            prompt = get_zero_shot_template_tacred_rag(sentence, relations, head, tail, context['similar_sentence'])
            data = {"prompt":prompt, "relation":line['relation']}
        prompts.append(data)

    print("Number Prompts:{0}".format(len(prompts)))

    return prompts

def semeval_format(test_data, relations, similar_sentences, labels, prompt_type="simple"):
    """Regenerate prompt for semeval dataset

    Args:
        test_data (list): list of test sentences along with e1 and e2
        relations (list): target relation label indexes
        similar_sentences (list): list of similar sentence with corresponding test data
        labels (list): the list  of target label names
        prompt_type (str, optional): prompt type. Defaults to "simple".

    Returns:
        list: the list of regenerated prompts
    """
    
    relation_names = list(set(relations))
    relations = ", ".join([relation for relation in relation_names])
    prompts = []

    for index, line in enumerate(test_data):

        label = labels[index]
        sentence = line
        context = similar_sentences[index]

        e1_index = sentence.find("<e1>")
        e2_index = sentence.find("<e2>")

        if e1_index < e2_index:
            head_name = re.findall("<e1>(.*?)</e1>", sentence, re.DOTALL)
            tail_name = re.findall("<e2>(.*?)</e2>", sentence, re.DOTALL)
            head = "e1"
            tail = "e2"
        else:
            # print("e2")
            head_name = re.findall("<e2>(.*?)</e2>", sentence, re.DOTALL)
            tail_name = re.findall("<e1>(.*?)</e1>", sentence, re.DOTALL)
            head = "e2"
            tail = "e1"

        head_name = " ".join(head_name)
        tail_name = " ".join(tail_name)
        
        if prompt_type == "simple":
            prompt = semeval_prompt_template(sentence, relations, head, tail, head_name, tail_name)
        
        if prompt_type == "rag":
            context = context[index]
            prompt = semeval_prompt_template_rag(sentence, relations, head, tail, head_name, tail_name, context['similar_sentence'])
            
        prompts.append({"prompt":prompt, "relation":label})

    print("Number of Prompts:{0}".format(len(prompts)))

    return prompts
  

def generate_prompts(sentences, relations, similar_sentences,  dataset="tacred", prompt_type="rag"):
    """Regenerate the user query along with similar sentence.

    Args:
        sentences (list): list of sentences or dataset
        relations (list): list of relations
        similar_sentences (list): list of similar sentences
        dataset (str, optional): dataset name. Defaults to "tacred".
        prompt_type (str, optional): approach type. Defaults to "rag".

    Returns:
        list of prompts: list of regenerated prompts
    """

    prompts = []

    if dataset == "semeval":
        
        if type == "simple":
            prompts = semeval_format(sentences, relations, relations)
        else:
            prompts = semeval_format(sentences, relations, relations, prompt_type)
    else:

        if prompt_type == "simple":
            prompts = tacred_format(sentences, relations, similar_sentences)
        else:
            prompts = tacred_format(sentences, relations, similar_sentences, prompt_type)
    
    return prompts
    


