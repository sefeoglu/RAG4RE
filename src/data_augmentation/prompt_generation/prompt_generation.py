import os
import sys
import json
from prompt_templates import get_zero_shot_template_tacred, get_zero_shot_template_tacred_rag, semeval_prompt_template_rag, semeval_prompt_template
import re

sys.path.append(os.path.join(os.path.dirname(__file__), '..'))


PACKAGE_PARENT = '.'
SCRIPT_DIR = os.path.dirname(os.path.realpath(os.path.join(os.getcwd(), os.path.expanduser(__file__))))
sys.path.append(os.path.normpath(os.path.join(SCRIPT_DIR, PACKAGE_PARENT)))
PREFIX_PATH = "/".join(os.path.dirname(os.path.abspath(__file__)).split("/")[:-1]) + "/"

def read_json(path):
    with open(path, 'r') as f:
        data = json.load(f)
    return data
def tacred_format(test_data, relations, similar_sentences, type="rag"):

    prompts = []
    relations = " ".join([relation for relation in relations])

    
    for index, line in enumerate(test_data):
        if index > -1:
            sentence = " ".join([ token for token in line['tokens']])
            head = line['subject']
            tail = line['object']

            if type == "simple":
                prompt = get_zero_shot_template_tacred(sentence, relations, head, tail)
                data = {"prompt":prompt, "relation":line['relation']}
            else:
                context = similar_sentences[index]
                train_relation = context['train_relation']
                prompt = get_zero_shot_template_tacred_rag(sentence, relations, head, tail, context['similar_sentence'], train_relation)
                data = {"prompt":prompt, "relation":line['relation']}
            prompts.append(data)

    print("Number Prompts:{0}".format(len(prompts)))

    return prompts

def semeval_format(test_data, relations, similar_sentences, labels, prompt_type="simple"):
    
    relation_names = list(set(relations))
    relations = ", ".join([relation for relation in relation_names])
    prompts = []
    for index, line in enumerate(test_data):
        label = labels[index]
        if index > -1:
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
                print("e2")
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
    prompts = []
    if dataset == "semeval":
        
        if type == "simple":
            prompts = semeval_format(sentences, relations, relations)
        else:
            prompts = semeval_format(sentences, relations, relations, "rag")
    else:
        if prompt_type == "simple":
            tacred_format(sentences, relations, similar_sentences)
        else:
            tacred_format(sentences, relations, similar_sentences, "rag")
    
    return prompts
    


