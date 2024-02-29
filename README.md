# RAG4RE
Retrieval-Augmented System-based Relation Extraction
## Folder Hierarchy

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
Then:
````bash
$ python src/main.py
```
