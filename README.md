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
* 1.) First instrall requirements
````bash
    pip install -r requirements.txt
  
````
* 2.) Compute embeddings and similarities for benchmark datasets in advance
````bash
    cd src/data_augmentation/embeddings
    python sentence_embeddings.py
    python sentence_sim.py
````
* 3.) Run Project
  
````bash
$ python src/main.py

````

