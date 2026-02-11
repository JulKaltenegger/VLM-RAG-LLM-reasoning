# Documenting existing building materials
Documenting information about materials in existing buildings is essential for circular construction and the reuse of elements. While multimodal Vision-Language Models (VLMs) can classify materials from images, they often lack the contextual semantic depth required for technical documentation. This article introduces a VLM-augmented Retrieval-Augmented Generation (RAG) framework that utilises  ntology-constrainedLLMreasoning to bridge this gap. By integrating domain-specific text retrieval and formal ontologies, the method enables the identification of building element layers and material types. The method is validated using a reference dataset of Dutch residential
buildings and demonstrated by enriching semantic graphs with predicted material data.

# VLM-RAG-LLM-reasoning
This code implementation proposes a VLM-augmented RAG framework for ontologyconstrained LLM reasoning to enable the recognition, identification and inference of material information in existing buildings across archetypes.

* (1) A VLM-based prediction using on-site images
of buildings and structuring the corresponding predicted
residential building types and material categories in a semantic
graph;
* (2) A VLM-augmented RAG process that
incorporates domain-specific knowledge, including construction
regulations and archetype definitions;
* (3) An ontology-constrained LLM-based reasoning that incorporates
the RAG insights and parses the predicted information
as semantic graph statements.


##### Installation guid
* To reproduce the environment use pip install -r requirements.txt.
* Python version: 3.7-3.10
* cuDNN: 8.1 [link](https://developer.nvidia.com/rdp/cudnn-archive)
* CUDA: 11.2 [link](https://developer.nvidia.com/cuda-11.2.0-download-archive?target_os=Windows&target_arch=x86_64)
In order to process the code using GPU with CUDA and cuDNN add the CUDA installations \bin, \libnvvp and \extras\CUPTI\lib64 to the environmental variables (C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v11.8).


##### Project content
In this project you will find the CLIP-RAG_Ont.py file and the validation dataset (Arch_img, Mat_img, and JSON).

##### CLIP-RAG_Ont.py outline
> 1. Configuration

> 2. Load Models 
>> CLIP "openai/clip-vit-base-patch32" \
>> embedder_model "all-MiniLM-L6-v2"

> 3. Utilities
>> Normalisation

> 4. CLIP inference
>> Predict Building Type \
>> Predict Material Category

> 5. RAG retrieval
>> Archetype Corpus
>> Retrieval Query \
>> COSINE matching 

> 6. LLM reasoning
>> Context information as input
>> LLM prompt template

> 7. LLM output to TTL parsing
>> Map LLM results to ontology syntax

> 8. Validation
>> CLIP validation \ 
>> JSON validation dataset \
>> COSINE similarity between LLM reasoning and ground truth dataset.



