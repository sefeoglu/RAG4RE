import re

def clean_t5_response(dataset_name, test_data, test_results, relations):
    """ Refine t5 response.
    Args:
        dataset_name (str): test data name
        test_data (list): list of sentences along with their types
        test_results (list): list of sentences
        relations (list): list of sentences
    Returns:
        list: list of predictions
    """
    predictions = []
    
    relations = [relation.lower() for relation in relations]
    targets = {}
    for relation in relations:
        key = relation.split(" ")[-1].split(":")[-1].strip()
        value = relation
        if key not in targets:
            targets[key] = value

    relations = [relation.split(" ")[-1].split(":")[-1].strip() for relation in relations]

    for i in range(0,len(test_results)):
        test = test_results[i].split(" ")[-1].split(":")[-1].strip()
        predictions.append(test)
   
    preds = []
    if dataset_name != "semeval":
        for i, sentence in enumerate(test_data):
            
            subj = "org" if "ORGANIZATION" == sentence["subject_type"] else "per"

            if predictions[i] in ["alternate_names", "parents"]:
                preds.append(subj+":"+ predictions[i])
            elif predictions[i] in relations:
                preds.append(targets[predictions[i]])
            else:
                preds.append(predictions[i])
    else:
        preds = predictions
    # print("predictions", len(preds))
            
    return preds

def clean_instruction(data):
    """clean the instruction from the responses when the llm is llama or mistral

    Args:
        data (list): list of responses

    Returns:
        list: list of cleaned responses from instructions
    """
    clean_data = []
    
    for _, line in enumerate(data):

        if "Answer:" in line:
            raw_answer = line.replace("\n","").split("Answer:")[-1]
        else: 
            raw_answer = line.split("[/INST]")[1]
        clean_data.append(raw_answer)

    return clean_data

def find_relations_inanswer(dataset_name, data, responses, relations):
    """find the relations in the answer

    Args:
        dataset_name (str): test data name
        data (list): list of test data along with their types
        responses (list): list of responses
        relations (list): target labels

    Returns:
        list: list of precisted relations
    """
    clean_data = {}
    relations = [relation.lower() for relation in relations]
    targets = {}
    
    for relation in relations:
        key = relation.split(" ")[-1].split(":")[-1].strip()
        value = relation
        if key not in targets:
            targets[key] = value
    # print(targets.keys())
    relations = [relation.lower().replace("per","").replace("org","") for relation in relations]
    relations.append("no relation")
    # print(responses)
    for i, item in enumerate(responses):
        relation_types = [relation for relation in relations if relation.lower() in item.replace("\\","").lower()]
        
        if len(relation_types) == 0:
            m = re.search('\"per:(.+?)\"', item)
            if m is not None:
                relation_types.append(str(m.group(1)).replace("\\",""))
                
            m = re.search('\"org:(.+?)\"', item)
            if m is not None:
                relation_types.append(str(m.group(1)).replace("\\",""))
            if len(relation_types)!=2:
                m = re.search('per:(.+?)."', item)
                if m is not None:
                    relation_types.append(str(m.group(1)).replace("\\","").replace('"', ''))

                m = re.search('org:(.+?)."', item)
                if m is not None:
                    relation_types.append(str(m.group(1)).replace("\\","").replace('"', ''))

        print(relation_types)
        if len(relation_types) == 0:
            relation_types.append("")

        clean_data[str(i)] =  "no_relation" if relation_types[0] == "no relation" else relation_types[0]
    write_json("/Users/sefika/Documents/RAG4RE/results/llama2_7b/returned_responses/llama_7b_tacred_simple_clean.json", clean_data)
    preds = []
    if dataset_name != "semeval":
        for i, sentence in enumerate(data):
            # print(sentence)
            subj = "org" if "ORGANIZATION" == sentence["subject_type"] else "per"

            if clean_data[str(i)] in ["alternate_names", "parents"]:
                preds.append(subj+":"+clean_data[str(i)])
            elif clean_data[str(i)] in relations:
                preds.append(targets[clean_data[str(i)].replace(":","")])
            else:
                preds.append(clean_data[str(i)])
    else:
        preds =  clean_data

    return preds 

def postprocessing(dataset_name, test_data, responses, relations, model_name):
    """
    Postprocess the response from the data augmentation module.
    Args:
        test_data (list): list of test data along with their types
        responses (list): list of responses
        relations (list): list of relations
        model_name (str): model name
    Returns:
        list: list of post processed responses
    """
    # postprocess the response
    if "t5" in model_name.lower():
        responses = clean_t5_response(dataset_name, test_data, responses, relations)
    else:
        responses = clean_instruction(responses)
        responses = find_relations_inanswer(dataset_name,test_data, responses, relations)

    return responses
