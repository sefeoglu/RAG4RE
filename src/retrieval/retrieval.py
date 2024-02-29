import os
import sys
import configparser

from data_augmentation.prompt_generation.prompt_generation import generate_prompts
from generation_module.generation import LLM
import configparser
from utils import read_json, write_json
PACKAGE_PARENT = '.'
SCRIPT_DIR = os.path.dirname(os.path.realpath(os.path.join(os.getcwd(), os.path.expanduser(__file__))))
sys.path.append(os.path.normpath(os.path.join(SCRIPT_DIR, PACKAGE_PARENT)))
PREFIX_PATH = "/".join(os.path.dirname(os.path.abspath(__file__)).split("/")[:-1]) + "/"
from refinement import postprocessing

def benchmark_data_augmentation_call(config_file_path):
    """
    This function is used to benchmark the retrieval module.
    """
    print("PREFIX_PATH", PREFIX_PATH)

    config = configparser.ConfigParser()
    config.read(PREFIX_PATH+"config.ini")
    
    test_data_path = config["PATH"]["test_data_path"]
    similar_sentences_path = config["SIMILARITY"]["output_index"]
    relations_path = config["PATH"]["relations_path"]
    
    similar_sentences = read_json(similar_sentences_path)
    relations = read_json(relations_path)
    relations = relations.keys()
    test_data = read_json(test_data_path)

    dataset = config["SETTINGS"]["dataset"]
    prompt_type = config["SETTINGS"]["prompt_type"]
    model_name = config["SETTINGS"]["model_name"]
    
    if prompt_type == "rag":
        print("RAG")
        output_prompts_path = config["OUTPUT"]["rag_test_prompts_path"]
        output_responses_path = config["OUTPUT"]["rag_test_responses_path"]
        prompts = generate_prompts(test_data, relations, similar_sentences,  dataset, prompt_type)
    else:
        output_prompts_path = config["OUTPUT"]["simple_prompt_path"]
        output_responses_path = config["OUTPUT"]["simple_prompt_responses_path"]
        prompts = generate_prompts(test_data, relations, similar_sentences,  dataset, prompt_type)

    llm_instance = LLM(model_name)
    
    responses = []

    for prompt in prompts:
        prompt = prompt["prompt"]
        response = llm_instance.get_prediction(prompt)
        responses.append(response)

    responses = postprocessing(responses, relations, model_name)
    
    write_json(output_prompts_path, prompts)
    write_json(output_responses_path, responses)
    

# if __name__ == "__main__":
#     config_file_path = "config.ini"
#     benchmark_data_augmentation_call(config_file_path)

