import re


def clean_t5_response(test_results, relations):
    """ Refine t5 response."""
    predictions = []
    relations = [relation.lower() for relation in relations]
    relations = [relation.split(" ")[-1].split(":")[-1].strip() for relation in relations]

    for i in range(0,len(test_results)):
        test = test_results[i].split(" ")[-1].split(":")[-1].strip()
        if test.lower() in relations:
            predictions.append(test)
    return predictions

def clean_instruction(data):
    clean_data = []
    
    for i, line in enumerate(data):

        if "Answer:" in line:
            raw_answer = line.replace("\n","").split("Answer:")[1]
        else: 
            raw_answer = ""
        clean_data.append(raw_answer)

    return clean_data

def find_relations_inanswer(data, relations):
    clean_data = {}
    for i, item in enumerate(data):
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


        if len(relation_types) == 0:
            relation_types.append("")
        clean_data[str(i)] = relation_types
        
    return clean_data, 

def postprocessing(responses, relations, model_name):
    """
    Postprocess the response from the data augmentation module.
    """
    # postprocess the response
    if model_name == "t5":
        responses= clean_t5_response(responses)
    else:
        responses = clean_instruction(responses)
        responses = find_relations_inanswer(responses, relations)

    return responses