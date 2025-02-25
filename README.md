# Reproduction of GS-VQA with LLM Question Parsing

## Project Structure
```
├── logs/ 
│   └── log_<X>.csv 
├── poetry.lock 
├── pyproject.toml 
└── src/ 
    ├── constants.py 
    ├── eval_full_pipeline.py 
    ├── asp_encoding/ (1)
    │   ├── asp_utils.py 
    │   ├── perfect_information_encoding.py 
    │   ├── question_encoding.py 
    │   ├── encode_scene.py 
    │   └── run_clingo.py 
    ├── evaluation/ (2)
    │   └── csvlogger.py 
    ├── gs_vqa/ (3)
    │   ├── model
    │   │   ├── base_model.py
    │   │   └── clip_model.py
    │   ├── object_detection/
    │   │   ├── object_detector.py
    │   │   └── owl_vit_object_detector.py
    │   ├── pipeline/
    │   │   ├── bounding_box_optimization.py
    │   │   └──  concept_extraction.py
    │   └── gs_vqa_utils.py
    ├── llms/ (4)
    │   ├── genericLLM.py 
    │   ├── localmodels.py 
    │   └── openaiLLM.py 
    ├── prompt_tools/ (5)
    │   ├── prompt_creator.py 
    │   └── prompt_templates.py 
    ├── semantic_interpreter/ (6)
    │   ├── atomic_operations.py 
    │   ├── auxil_operations.py 
    │   ├── gqa_object.py 
    │   ├── known_exception.py 
    │   └── run_func_program.py 
    └── utils/ (7)
        ├── blind_concept_extractor.py 
        ├── prepare_data.py 
        ├── transform_representation.py 
        └── utils.py 
```

This project contains code for a neurosymbolic visual question answering pipeline originally developed on the GQA dataset.

Questions are translated into an answer set programming (ASP) representation, and a conditional scene graph is generated and also translated to ASP. Jointly this encoding can be used by a solver (clingo) to answer the question. 
Correctness of an answer can be determined with variable degree of generosity, i.e. accepting synonyms or the top-k scene interpretations.

The relevant elements of the project are:
- (1) utilities for encoding the scene graph and question into ASP. Perfect information refers to the ground truth scene graph. Clingo is the solver.
- (2) utilities for evaluating the pipeline or parts thereof. The csvlogger is a convenience wrapper around the logfiles, for evaluating and comparing runs.
- (3) the scene processing pipeline itself. Contains models for object detection (owl) and attribute/relation classification (clip). Additionally contains code to process bounding boxes and extract relevant concept from the question and perform minor input sanitization. For details see Hadl (2023)
- (4) wrapper around llms used for question parsing into asp. Includes output sanitization and error handling.
- (5) prompt templates and assembly (various in context selection and prompting strategies).
- (6) algorithmic evaluation engine for ground truth scene graphs. Given a scene graph and semantic question representation tries to parse question as graph operations to obtain answer. Includes error classification, as both engine and gt data from GQA are imperfect.
- (7) utilities for concept extraction without ground truth question encoding (blind), data preparation (collection of in context examples / aggregation from json), translation between different representations (flat/nested/asp/codelike), convenience question iterator and evaluation of correctness.
