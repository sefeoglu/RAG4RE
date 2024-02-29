# RAG4RE
Retrieval-Augmented System-based Relation Extraction


## Project Folder Hierarchy

````bash
.
├── LICENSE
├── README.md
├── data
├── results
└── src
    ├── config.ini
    ├── data_augmentation
    │   ├── embeddings
    │   └── prompt_generation
    ├── data_preparation
    ├── evaluation
    │   ├── results_analysis.py
    │   └── vizualization.py
    ├── generation_module
    │   └── generation.py
    ├── main.py
    ├── retrieval
    │   ├── refinement.py
    │   └── retrieval.py
    └── utils.py
````
## How to run
Change the paths and configs under `config.ini`
* 1.) Datasets
   Put the following dataset under `data` folder.
  
   * TACRED dataset is lincensed by Linguistic Data Consortium (LDC), so please download it from [here](https://catalog.ldc.upenn.edu/LDC2018T24)
     
   * TACREV dataset is constructed from TACRED via the tacrev [codes](https://github.com/DFKI-NLP/tacrev)
     
   * Re-TACRED dataset is derived from TACRED via this [repository](https://github.com/gstoica27/Re-TACRED)

   * SemEval is available at the [hugging face](https://huggingface.co/datasets/sem_eval_2010_task_8) and under `data` folder.

* 2.) First instrall requirements
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

