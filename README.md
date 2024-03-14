# RAG4RE
[![Python  3.10.9](https://img.shields.io/badge/python-3.10.9-blue.svg)](https://www.python.org/downloads/release/python-3109/)


The repository consists of the source codes of "Retrieval-Augmented Generation-based Relation Extraction" journal paper which has been submitted to Semantic Web Journal (SWJ).

Note that TACRED is licensed by the Linguistic Data Consortium (LDC), so we cannot directly publish the prompts or the raw results from the experiments conducted with Llama and Mistral, since the responses of these models consists of the prompts in their instruction parts. However, we have published the returned results when Llama and Mistral were integrated. Upon an official request, the data can be accessed on LDC, and the experiments can be easily replicated by following the instructions provided.

## Project Folder Hierarchy

````
.
├── LICENSE
├── README.md
├── data                            ---> dataset, such as tacred, tacrev, re-tacred and semeval
├── results                         ---> results will be saved here.
└── src
    ├── config.ini                  ---> configuration for dataset, approach and llm and results.
    ├── data_preparation
    ├── main.py                     ---> the pipeline is started with this
    ├── retrieval                   ---> retrieval module
    │   ├── refinement.py
    │   └── retriever.py
    ├── data_augmentation           ---> regenerated the user query
    │   ├── embeddings
    │   └── prompt_generation
    ├── generation_module           ---> llm prompting.
    │   └── generation.py
    ├── evaluation                  ---> evaluate and visualize results. 
    │   ├── results_analysis.py
    │   └── vizualization.py
    └── utils.py                    
````
## How to run
Change the paths and configs under `config.ini` for your experiment.
* 1.) Datasets
  
   Put the following dataset under `data` folder.
  
   * TACRED dataset is lincensed by Linguistic Data Consortium (LDC), so please download it from [here](https://catalog.ldc.upenn.edu/LDC2018T24)
     
   * TACREV dataset is constructed from TACRED via the tacrev [codes](https://github.com/DFKI-NLP/tacrev)
     
   * Re-TACRED dataset is derived from TACRED via this [repository](https://github.com/gstoica27/Re-TACRED)

   * SemEval is available at the [hugging face](https://huggingface.co/datasets/sem_eval_2010_task_8) and under `data` folder.

* 2.) First install requirements
  
````bash
    pip install -r requirements.txt
````
* 3.) Compute embeddings and similarities for benchmark datasets in advance
````bash
    cd src/data_augmentation/embeddings
    python sentence_embeddings.py
    python sentence_sim.py
````
* 4.) Run Project
  
````bash
$ python src/main.py

````

