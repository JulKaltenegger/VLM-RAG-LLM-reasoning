import os
import json
import torch
import pandas as pd
from PIL import Image
import numpy as np
import requests
from typing import List, Dict, Optional

from sentence_transformers import SentenceTransformer, util
from transformers import CLIPProcessor, CLIPModel
import ollama
import re
import matplotlib.pyplot as plt
from sklearn.metrics import classification_report, confusion_matrix, ConfusionMatrixDisplay, accuracy_score

#################################################
#  CONFIGURATION
#################################################
# Paths
IMAGE_FOLDER = "../Arch_images"
MATERIAL_ROOT = "../Mat_images"

JSON_PATH = "../JSON/image_filenames_with_labels-260202.json"
ARCHETYPE_CSV = "..ArchetypeDescription/TextDescription.csv"


MATERIAL_FOLDERS = {
    "Brick": "Brick",
    "Concrete": "Concret"
}

# Device & dtype
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
DTYPE = torch.float16 if DEVICE == "cuda" else torch.float32

# LLM model
LLM_MODEL = "llama3:8b"

# Building types and prompts
BUILDING_TYPES = ["Single Family House", "Terrace House", "Apartment House"]
BUILDING_TYPE_PROMPTS = {
    "Terrace House": [
        "A long row of identical brick houses sharing side walls, two to three storeys tall, narrow facades, typical Dutch street",
        "A Dutch terrace house in a continuous row of houses",
        "Row houses sharing walls on both sides",
        "Residential buildings forming a long row"
    ],
    "Single Family House": [
        "A detached Dutch single family house",
        "A semi-detached house with one shared wall",
        "A single residential building not attached on both sides"
    ],
    "Apartment House": [
        "A medium to high-rise residential building",
        "A multi-storey apartment building",
        "An apartment block with stacked dwellings"
    ]
}

MATERIAL_CATEGORIES = ["Brick", "Concrete"]
MATERIAL_PROMPTS = {
    "Brick": [
        "Close-up photo of a brick wall with visible mortar joints",
        "Close-up image of red or brown fired clay bricks",
        "Detailed texture of a brick masonry wall"
    ],
    "Concrete": [
        "Close-up photo of a concrete wall surface",
        "Detailed texture of cast concrete with pores",
        "Close-up image of smooth or rough concrete material"
    ]
}
MASTER_ONTOLOGY = {
    "wall_layer_options": {
        "bmp:SingleLayer": "A solid, monolithic wall construction without separate cavities or insulation layers.",
        "bmp:MultiLayer": "A wall assembly with several distinct layers, often combining structure, insulation, and cladding (e.g., sandwich panels).",
        "bmp:CavityWall": "Two masonry leaves separated by an air space (cavity), sometimes filled with insulation to prevent moisture and heat loss.",
        "bmp:Frame": "A skeletal or frame structure of wood or steel."
    },
    "material_categories": {
        "bmp:Brick": {
            "bmp:BrickMasonryUnit": "General masonry units made of fired clay or other materials used in standard bonding.",
            "bmp:SandLimeBrick": "A white or light-grey masonry unit made of lime and sand, cured under steam pressure.",
            "bmp:ClayUnit": "Traditional red, white, or brown fired clay bricks made from river clay, used primarily for the outer facade leaf."
        },
        "bmp:Concrete": {
            "bmp:HeavyWeightAggregate": "Concrete using high-density aggregates for specialized structural or shielding purposes.",
            "bmp:HeavyWeightConcrete": "Cast-in-situ or precast concrete with high density, used for sound insulation and structural stability.",
            "bmp:LightWeightAggregate": "Concrete made with lightweight materials like expanded clay or shale for better thermal insulation.",
            "bmp:LightWeightConcrete": "Porous concrete designed for thermal performance and reduced dead load in construction.",
            "bmp:CementStucco": "A decorative and protective cement-based plaster coating applied to the exterior of walls."
        },
        "bmp:Polyester": {
            "bmp:ResinPolyester": "Polymer-based facade panels or glass-fiber reinforced resin components used as lightweight cladding."
        }
    }
 }
#################################################
#  MODELS
#################################################
# Load CLIP model
clip_model = CLIPModel.from_pretrained(
    "openai/clip-vit-base-patch32",
    torch_dtype=DTYPE
).to(DEVICE)
clip_processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")

# Load sentence transformer for corpus embeddings / semantic similarity
embedder_model = SentenceTransformer("all-MiniLM-L6-v2")

#################################################
#  UTILITIES
#################################################
def safe_lower(val: Optional[str]) -> str:
    if val is None:
        return ""
    return str(val).strip().lower()

def normalize_building_type(bt: str) -> str:
    bt = safe_lower(bt)
    bt = bt.replace("-", " ").replace("_", " ")
    mapping = {
        "apartment house": "apartment_house",
        "terrace house": "terrace_house",
        "single family house": "single_family_house",
        "semi detached house": "semi_detached_house",
        "detached house": "detached_house",
        "multi family house": "multi_family_house",
    }
    return mapping.get(bt, bt.replace(" ", "_"))

def normalize_material(mat: str) -> str:
    mat = safe_lower(mat)
    return mat.replace("-", " ").replace("_", " ")

#################################################
#  CLIP INFERENCE
#################################################
def predict_building_type(image: Image.Image, prompt_dict: Dict[str, List[str]], k: int = 2) -> List[Dict]:
    prompts, labels = [], []
    for btype, plist in prompt_dict.items():
        for p in plist:
            prompts.append(p)
            labels.append(btype)
    inputs = clip_processor(text=prompts, images=image, return_tensors="pt", padding=True)
    for key in inputs:
        inputs[key] = inputs[key].to(DEVICE)
    with torch.no_grad():
        outputs = clip_model(**inputs)
    logits = outputs.logits_per_image[0]
    class_logits = {}
    for logit, label in zip(logits, labels):
        class_logits.setdefault(label, []).append(logit.item())
    class_logits = {k: sum(v)/len(v) for k, v in class_logits.items()}
    probs = torch.softmax(torch.tensor(list(class_logits.values())), dim=0)
    ranked = sorted(
        [{"BuildingType": lbl, "confidence": float(p)} for lbl, p in zip(class_logits.keys(), probs)],
        key=lambda x: x["confidence"], reverse=True
    )
    return ranked[:k]

def predict_material(image: Image.Image, material_category: List[str], k: int = 1) -> List[Dict]:
    prompts, labels = [], []
    for mat in material_category:
        for p in MATERIAL_PROMPTS[mat]:
            prompts.append(p)
            labels.append(mat)
    inputs = clip_processor(text=prompts, images=image, return_tensors="pt", padding=True)
    for key in inputs:
        inputs[key] = inputs[key].to(DEVICE)
    with torch.no_grad():
        outputs = clip_model(**inputs)
    logits = outputs.logits_per_image[0]
    class_logits = {}
    for logit, label in zip(logits, labels):
        class_logits.setdefault(label, []).append(logit.item())
    class_logits = {k: sum(v)/len(v) for k, v in class_logits.items()}
    probs = torch.softmax(torch.tensor(list(class_logits.values())), dim=0)
    ranked = sorted(
        [{"material": mat, "confidence": float(p)} for mat, p in zip(class_logits.keys(), probs)],
        key=lambda x: x["confidence"], reverse=True
    )
    return ranked[:k]

#################################################
#  RAG RETRIEVAL
#################################################
# Load archetype corpus CSV
archetype_df = pd.read_csv(ARCHETYPE_CSV)
archetype_df["Building Type"] = archetype_df["Building Type"].astype(str).str.lower().str.replace(" ", "_")

def row_to_text(row):
    return (
        f"Building Type: {row.get('Building Type', '')} | "
        f"Year range: {row.get('yearFrom', '')}-{row.get('yearTo', '')} | "
        f"Wall Description: {row.get('WallDescription', '')} | "
        f"Archetype Description: {row.get('ArchetypeDescription', '')}"
    )

archetype_df["text"] = archetype_df.apply(row_to_text, axis=1)
corpus_embeddings = embedder_model.encode(archetype_df["text"].tolist(), convert_to_tensor=True)

def build_retrieval_query(bt_preds, mat_pred, year, region) -> str:
    # Construct query string for embedding-based retrieval
    return (
        f"Dutch residential {bt_preds[0]['BuildingType']}, "
        f"Construction period around {year} in {region}, "
        f"Facade material {mat_pred[0]['material']}."
    )

def retrieval_archetypedescription(query, top_k=5):
    """
    Standard cosine similarity retrieval of top_k corpus entries
    based purely on query embeddings.
    """
    query_emb = embedder_model.encode(query, convert_to_tensor=True)
    scores = util.cos_sim(query_emb, corpus_embeddings)[0]
    top = torch.topk(scores, k=top_k)
    retrieved = []
    for idx, score in zip(top.indices, top.values):
        row = archetype_df.iloc[int(idx)]
        retrieved.append({
            "similarity": float(score),
            "BuildingType": row["Building Type"],
            "yearFrom": row.get("yearFrom", ""),
            "yearTo": row.get("yearTo", ""),
            "WallDescription": row.get("WallDescription", ""),
            "Description": row.get("ArchetypeDescription", "")
        })
    return retrieved

def retrieval_archetypedescription_for_llm(bt_preds, mat_preds, year, region, top_k=5):
    # 1. Narrow corpus by building type AND year first
    year = int(year)
    bt_norm = normalize_building_type(bt_preds[0]['BuildingType'])

    eligible_rows = archetype_df[
        (archetype_df['Building Type'] == bt_norm) &
        (archetype_df['yearFrom'].astype(int) <= year) &
        (archetype_df['yearTo'].astype(int) >= year)
        ]

    if eligible_rows.empty:
        # fallback to all rows of that building type
        eligible_rows = archetype_df[archetype_df['Building Type'] == bt_norm]

    # 2. Compute cosine similarity against filtered subset
    corpus_embeds_filtered = embedder_model.encode(eligible_rows['text'].tolist(), convert_to_tensor=True)
    query = build_retrieval_query(bt_preds, mat_preds, year, region)
    query_emb = embedder_model.encode(query, convert_to_tensor=True)
    scores = util.cos_sim(query_emb, corpus_embeds_filtered)[0]
    top = torch.topk(scores, k=min(top_k, len(scores)))

    retrieved = []
    for idx, score in zip(top.indices, top.values):
        row = eligible_rows.iloc[int(idx)]
        retrieved.append({
            "similarity": float(score),
            "BuildingType": row["Building Type"],
            "yearFrom": row.get("yearFrom", ""),
            "yearTo": row.get("yearTo", ""),
            "WallDescription": row.get("WallDescription", ""),
            "Description": row.get("ArchetypeDescription", "")
        })
    return retrieved

def filter_retrieved_by_year_and_type_soft(retrieved, year, building_type):
    """
    Return the best matching archetype for the given year and building type.
    Uses soft year matching: closest year to the midpoint of yearFrom-yearTo.
    """
    year = int(year)
    bt_norm = building_type.strip().lower()
    candidates = []

    for item in retrieved:
        try:
            y_from = int(item["yearFrom"])
            y_to = int(item["yearTo"])
            item_bt = str(item["BuildingType"]).strip().lower()
            if bt_norm in item_bt:
                midpoint = (y_from + y_to) / 2
                distance = abs(year - midpoint)
                candidates.append((distance, item))
        except Exception:
            continue

    if not candidates:
        # fallback: use first retrieved row
        fallback = retrieved[0].copy() if retrieved else {}
        fallback["__fallback__"] = True
        return [fallback]

    # Return the row with minimal year distance
    candidates.sort(key=lambda x: x[0])
    return [candidates[0][1]]

#################################################
#  LLM REASONING
#################################################
def llm_answer(context: Dict, question: str) -> str:
    """
    Use structured prompt with CLIP + metadata + retrieval to reason and output TTL triples.
    """
    region = context.get("metadata", {}).get("region", "the Netherlands")
    year = context.get("metadata", {}).get("year", "unknown")
    visual_preds = context.get("visual_predictions", {})
    building_types = visual_preds.get("building_types", [])
    materials = visual_preds.get("materials", [])

    # 1. Extract specific values for strict mapping
    bt_pred = building_types[0]["BuildingType"] if building_types else "Unknown"
    bt_conf = building_types[0]["confidence"] if building_types else 0.0

    # This is the "Category" (e.g., Brick) which must stay distinct from Subtypes
    mat_cat_pred = materials[0]["material"] if materials else "Unknown"
    mat_cat_conf = materials[0]["confidence"] if materials else 0.0

    # 2. Build the Technical Dictionary Guidance
    ontology_guidance = "TECHNICAL DICTIONARY:\n"
    for iri, desc in MASTER_ONTOLOGY["wall_layer_options"].items():
        ontology_guidance += f"- {iri}: {desc}\n"
    for category, subtypes in MASTER_ONTOLOGY["material_categories"].items():
        ontology_guidance += f"\nCategory {category}:\n"
        for iri, desc in subtypes.items():
            ontology_guidance += f"  * {iri}: {desc}\n"

    # 3. Dynamic Retrieval Formatting
    retrieved = context.get("retrieve_archdescription", [])
    retrieved_corpus_text = ""
    if not retrieved or (isinstance(retrieved[0], dict) and retrieved[0].get("__fallback__")):
        retrieved_corpus_text = "No specific RVO record found. Use general Dutch construction knowledge for this era."
    else:
        for i, item in enumerate(retrieved, start=1):
            retrieved_corpus_text += (
                f"   [SOURCE {i}]: Building Type: {item.get('BuildingType')}, "
                f"Year: {item.get('yearFrom')}-{item.get('yearTo')}, "
                f"Wall: {item.get('WallDescription')}, "
                f"Full Description: {item.get('Description')}\n"
            )

    # 4. Structured Prompt with corrected Example Hierarchy
    prompt = f"""
    You are an expert in Dutch residential building construction. Generate valid TTL.

    INPUT EVIDENCE:
    - Metadata Year: {year}
    - Metadata Region: {region}
    - Visual Prediction (Type): {bt_pred} (conf: {bt_conf:.2f})
    - Visual Prediction (Material Category): {mat_cat_pred} (conf: {mat_cat_conf:.2f})

    STRICT SELECTION RULES:
    1. 'bmp:hasLayerSet': Select the IRI from the Dictionary that best matches the RVO evidence.
    2. 'bmp:hasMaterialCategory': Use 'bmp:{mat_cat_pred}'. Do NOT use subtypes here.
    3. 'bmp:hasMaterialType': Select a subtype IRI ONLY from the Dictionary list under Category 'bmp:{mat_cat_pred}'.
    4. 'bmp:WallArchetypeDescription': REQUIRED. Synthesize a technical summary of the physical layers (materials, thickness, air gaps) from the RVO evidence.

    OUTPUT TASK:
    Generate TTL. Ensure the 'WallArchetypeDescription' is part of 'inst:wall01'.
    Produce TTL. Match the 'bmp:hasLayerSet' IRI to the dictionary based ONLY on the RVO Evidence above.
    RVO CORPUS EVIDENCE:
    {retrieved_corpus_text}

    DICTIONARY
    {ontology_guidance}

    EXAMPLE OUTPUT FORMAT:
    @prefix at: <https://w3id.org/at#> .
    @prefix beo: <https://w3id.org/beo#> .
    @prefix bmp: <https://w3id.org/bmp#> .
    @prefix inst: <https://netherlandsterracehouse.org/inst#> .
    @prefix xsd: <https://www.w3.org/2001/XMLSchema#>.

    inst:building01
      at:BuildingType [at:hasValue at:{bt_pred.replace(" ", "")}; at:confidence "{bt_conf:.2f}"];
      at:ConstructedIn "{year}"^^xsd:gYear ;
      at:Region "{region}"^^xsd:string .

    inst:wall01
      rdf:type beo:Wall ;
      bmp:hasLayerSet <IRI_FROM_DICTIONARY> ;
      bmp:hasOuterLayer inst:Layer01 ;
      bmp:WallArchetypeDescription "Technical summary of layers from RVO evidence..."^^xsd:string .

    inst:Layer01
      rdf:type bmp:Layer ;
      bmp:hasMaterialCategory [bmp:hasValue bmp:{mat_cat_pred}; at:confidence "{mat_cat_conf:.2f}"] ;
      bmp:hasMaterialType <SUBTYPE_IRI_FROM_DICTIONARY> .
    """

    response = ollama.chat(
        model=LLM_MODEL,
        messages=[{"role": "user", "content": prompt}],
        options={"temperature": 0.1, "top_p": 0.9}
    )
    return response["message"]["content"]


def answer_question(building_image, material_image, question, year, region):
    """
    Full pipeline for a single building instance:
    1. Visual inference with CLIP (building type + material)
    2. RAG retrieval using CLIP + metadata
    3. Soft filtering by year and building type
    4. LLM reasoning constrained by ontology
    """
    # 1. CLIP VISUAL PREDICTIONS
    building_type_preds = predict_building_type(building_image, BUILDING_TYPE_PROMPTS,k=2 )
    material_preds = predict_material(material_image, MATERIAL_CATEGORIES, k=2 )

    # 2. RAG RETRIEVAL (SEMANTIC SEARCH)
    # Retrieval uses CLIP predictions + metadata to query the corpus
    retrieved_candidates = retrieval_archetypedescription_for_llm(building_type_preds, material_preds, year, region, top_k=5)
    # 3. SOFT FILTERING BY YEAR & BUILDING TYPE
    filtered_retrieval = filter_retrieved_by_year_and_type_soft(retrieved_candidates, year, building_type_preds[0]["BuildingType"])

    # 4. CONTEXT FOR LLM REASONING
    context = {
        "visual_predictions": {
            "building_types": building_type_preds,
            "materials": material_preds
        },
        "metadata": {
            "year": year,
            "region": region
        },
        "retrieve_archdescription": filtered_retrieval,
        "ontology": MASTER_ONTOLOGY
    }

    # 5. LLM reasoning → TTL
    ttl_text = llm_answer(context, question)
    print("RAW LLM TTL OUTPUT")
    print(ttl_text)

    # 6. Parse TTL + map free-text descriptions to ontology IRIs
    parsed_result = parse_llm_ttl_output(ttl_text, embedder_model, MASTER_ONTOLOGY)

    return parsed_result

############# derive ontology master descriptions from nested structure
def get_ontology_desc(iri: str) -> str:
    """Recursively finds a definition in MASTER_ONTOLOGY."""
    # Check Layer Sets
    if iri in MASTER_ONTOLOGY["wall_layer_options"]:
        return MASTER_ONTOLOGY["wall_layer_options"][iri]

    # Check Material Categories and their Subtypes
    for cat_iri, subtypes in MASTER_ONTOLOGY["material_categories"].items():
        if iri == cat_iri:
            return f"General material category for {iri}"
        if iri in subtypes:
            return subtypes[iri]
    return iri  # Fallback

# map to text
def map_text_to_ontology(text: str, ontology_options: list, embedder_model) -> dict:
    if not text or not ontology_options:
        return {"pred": None, "confidence": 0.0}

    # Use the helper to get rich descriptions for each option
    search_targets = [get_ontology_desc(opt) for opt in ontology_options]

    option_embeddings = embedder_model.encode(search_targets, convert_to_tensor=True)
    text_emb = embedder_model.encode(text, convert_to_tensor=True)

    scores = util.cos_sim(text_emb, option_embeddings)[0]
    best_idx = torch.argmax(scores).item()

    return {"pred": ontology_options[best_idx], "confidence": float(scores[best_idx])}

#################################################
#  TTL PARSING
#################################################
def parse_llm_ttl_output(ttl_text: str, embedder_model, ontology: dict) -> dict:
    result = {}
    # Normalize text for easier pattern matching
    text = " ".join(ttl_text.split())

    #  1. Building Type
    m_bt = re.search(r"at:BuildingType\s*\[\s*at:hasValue\s+(?:at:)?\"?([\w\s]+)\"?", text, re.I)
    m_conf_bt = re.search(r"at:BuildingType.*?at:confidence\s*\"?([0-9.]+)\"?", text, re.I)
    if m_bt:
        bt_val = m_bt.group(1).strip().replace(" ", "")
        result["BuildingType"] = f"at:{bt_val}"
        result["BuildingTypeConfidence"] = float(m_conf_bt.group(1)) if m_conf_bt else 0.0
    else:
        result["BuildingType"], result["BuildingTypeConfidence"] = None, 0.0

    #  2. Year
    m_yr_block = re.search(r'at:ConstructedIn[^;.]+', text)
    if m_yr_block:
        m_yr = re.search(r'(\d{4})', m_yr_block.group(0))
        result["Year"] = int(m_yr.group(1)) if m_yr else None
    else:
        result["Year"] = None

    #  3. Region
    m_reg = re.search(r'at:Region\s+(?:at:|:)?["\']?([^"\'\^;\.\s]+)', text)
    result["Region"] = m_reg.group(1).strip() if m_reg else None

    #  4. WallArchetypeDescription (The 'Reasoning' source)
    m_desc = re.search(r'bmp:WallArchetypeDescription\s+"([^"]+)"', text, re.I)
    wall_desc = m_desc.group(1) if m_desc else ""
    result["WallArchetypeDescription"] = wall_desc

    #  5. LayerSet (Dynamic Mapping)
    m_ls = re.search(r"bmp:hasLayerSet\s+((?:bmp:\w+)|(?:\"[^\"]+\"))", text, re.I)
    ls_raw = m_ls.group(1) if m_ls else None

    # Always perform a semantic check against allowed LayerSet options
    ls_options = list(MASTER_ONTOLOGY["wall_layer_options"].keys())
    # We pass the rich definitions to the mapping function
    res_ls = map_text_to_ontology(wall_desc if wall_desc else "Construction", ls_options, embedder_model)

    if ls_raw and ls_raw.startswith("bmp:"):
        result["LayerSet"] = ls_raw
    else:
        result["LayerSet"] = res_ls["pred"]  # Fallback to best semantic match
    result["LayerSetConfidence"] = res_ls["confidence"]

    #  6. Material Category
    m_mat_cat = re.search(r"bmp:hasMaterialCategory\s*\[\s*bmp:hasValue\s+(?:at:|bmp:)?\"?(\w+)\"?", text, re.I)
    m_conf_mat = re.search(r"bmp:hasMaterialCategory.*?at:confidence\s*\"?([0-9.]+)\"?", text, re.I)
    mat_cat_label = m_mat_cat.group(1).strip() if m_mat_cat else ""
    result["MaterialCategory"] = f"bmp:{mat_cat_label}" if mat_cat_label else None
    result["MaterialCategoryConfidence"] = float(m_conf_mat.group(1)) if m_conf_mat else 0.0

    #  7. Material Type (Categorical Subtype Search)
    m_mt = re.search(r"bmp:hasMaterialType\s+((?:bmp:\w+)|(?:\"[^\"]+\"))", text, re.I)
    mt_raw = m_mt.group(1) if m_mt else None

    # DYNAMIC LOGIC: Only search subtypes belonging to the detected category
    cat_iri = f"bmp:{mat_cat_label}"
    allowed_subtypes = list(MASTER_ONTOLOGY["material_categories"].get(cat_iri, {}).keys())

    if not allowed_subtypes:
        # If CLIP category is unknown, search ALL subtypes in the ontology
        for sub_dict in MASTER_ONTOLOGY["material_categories"].values():
            allowed_subtypes.extend(list(sub_dict.keys()))

    # Calculate confidence based on the valid subtype list
    res_mt = map_text_to_ontology(wall_desc if wall_desc else "Material", allowed_subtypes, embedder_model)

    if mt_raw and mt_raw.startswith("bmp:"):
        result["MaterialType"] = mt_raw
    else:
        result["MaterialType"] = mt_raw.replace('"', '') if mt_raw else res_mt["pred"]

    result["MaterialTypeConfidence"] = res_mt["confidence"]

    return result

#################################################
#  VALIDATION
#################################################
def validate_clip_building(dataset, image_folder):
    """Validate CLIP building type predictions over dataset"""
    y_true, y_pred = [], []
    for entry in dataset:
        img_path = os.path.join(image_folder, entry["buildingImage"])
        if not os.path.exists(img_path):
            continue
        image = Image.open(img_path).convert("RGB")
        pred = predict_building_type(image, BUILDING_TYPE_PROMPTS, k=1)[0]
        y_true.append(normalize_building_type(entry["BuildingType"]))
        y_pred.append(normalize_building_type(pred["BuildingType"]))
    print("\n=== CLIP Building Type Validation ===")
    print(classification_report(y_true, y_pred, zero_division=0))
    print("Accuracy:", accuracy_score(y_true, y_pred))
    cm = confusion_matrix(y_true, y_pred, labels=sorted(set(y_true)))
    ConfusionMatrixDisplay(cm, display_labels=sorted(set(y_true))).plot(xticks_rotation=45, cmap="OrRd")
    plt.show()

def validate_clip_material(material_root, material_folders):
    """Validate CLIP material predictions over close-up dataset"""
    y_true, y_pred = [], []
    for mat, folder in material_folders.items():
        folder_path = os.path.join(material_root, folder)
        for fname in os.listdir(folder_path):
            if not fname.lower().endswith((".png", ".jpg", ".jpeg")):
                continue
            img_path = os.path.join(folder_path, fname)
            image = Image.open(img_path).convert("RGB")
            pred = predict_material(image, MATERIAL_CATEGORIES, k=1)[0]["material"]
            y_true.append(mat.lower())
            y_pred.append(pred.lower())
    print("\n=== CLIP Material Validation ===")
    print(classification_report(y_true, y_pred))
    ConfusionMatrixDisplay.from_predictions(y_true, y_pred, labels=["brick","concrete"], normalize="true", cmap="OrRd")
    plt.show()

#################################################
#  SINGLE IMAGE DEMO
#################################################
def run_single_image_demo(building_url, material_url, year, region):
    building_image = Image.open(requests.get(building_url, stream=True).raw).convert("RGB")
    material_image = Image.open(requests.get(material_url, stream=True).raw).convert("RGB")
    print("CLIP Predictions:")

    bt_preds = predict_building_type(building_image, BUILDING_TYPE_PROMPTS, k=2)
    mat_preds = predict_material(material_image, MATERIAL_CATEGORIES, k=2)
    print("Building type:", bt_preds)
    print("Material:", mat_preds)
    print("\nLLM TTL Reasoning:")

    ttl = answer_question(building_image, material_image, "Most likely wall construction?", year, region)
    #print(ttl)

    print("\nParsed TTL:")
    for k, v in ttl.items():
        print(f"{k}: {v}")

#################################################
#  VALIDATION data set
#################################################
def run_pipeline_benchmark(dataset: List[Dict], image_folder: str):
    """
    Full Pipeline Validation:
    Iterates through JSON, runs CLIP + RAG + LLM, and benchmarks against labels.
    """
    results = []

    print(f"\nStarting Full Pipeline Benchmark ({len(dataset)} images) ===")

    for entry in dataset:
        img_filename = entry["buildingImage"]
        img_path = os.path.join(image_folder, img_filename)

        if not os.path.exists(img_path):
            print(f"Skipping: {img_filename} (Not found)")
            continue

        # Load image once for both CLIP tasks
        image = Image.open(img_path).convert("RGB")

        # 1. Run the existing Pipeline
        # We pass the same image for both building and material focus
        try:
            parsed_output = answer_question(
                building_image=image,
                material_image=image,
                question="Determine the most likely wall construction based on visuals and age.",
                year=int(entry["yearTo"]),
                region=entry["Region"]
            )

            # 2. Benchmark Semantic Similarity (Reasoning vs. FacadeDescription)
            llm_desc = parsed_output.get("WallArchetypeDescription", "")
            gt_desc = entry.get("FacadeDescription", "")

            semantic_score = 0.0
            if llm_desc and gt_desc:
                emb1 = embedder_model.encode(llm_desc, convert_to_tensor=True)
                emb2 = embedder_model.encode(gt_desc, convert_to_tensor=True)
                semantic_score = float(util.cos_sim(emb1, emb2)[0][0])

            # 3. Benchmark Categorical Accuracy (LayerSet)
            # We normalize the ground truth string to match the bmp:IRI format
            gt_ls_iri = f"bmp:{entry['LayerSet'].replace(' ', '')}"
            pred_ls_iri = parsed_output.get("LayerSet", "")
            ls_match = 1 if safe_lower(gt_ls_iri) == safe_lower(pred_ls_iri) else 0

            results.append({
                "buildingImage": img_filename,
                "GT_LayerSet": entry["LayerSet"],
                "Pred_LayerSet": pred_ls_iri,
                "LayerSet_Match": ls_match,
                "GT_Material": entry["materialType"],
                "Pred_Material": parsed_output.get("MaterialType"),
                "Semantic_Similarity": semantic_score,
                "Confidence": parsed_output.get("LayerSetConfidence", 0.0)
            })

        except Exception as e:
            print(f"Error processing {img_filename}: {e}")
            continue

    # Create Benchmark Report
    df_results = pd.DataFrame(results)

    print("\n" + "=" * 50)
    print("PIPELINE BENCHMARK REPORT")
    print("=" * 50)
    print(f"Average Semantic Similarity: {df_results['Semantic_Similarity'].mean():.4f}")
    print(f"LayerSet Accuracy: {df_results['LayerSet_Match'].mean() * 100:.2f}%")
    print("=" * 50)

    # Optional: Display failures
    failures = df_results[df_results['LayerSet_Match'] == 0]
    if not failures.empty:
        print("\nTop LayerSet Mismatches:")
        print(failures[['buildingImage', 'GT_LayerSet', 'Pred_LayerSet']].head())

    return df_results
#################################################
#  MAIN EXECUTION
#################################################

if __name__ == "__main__":
    # Load validation JSON
    with open(JSON_PATH, "r") as f:
        dataset = json.load(f)

    # Single image demo
    run_single_image_demo(
    #     "https://images.squarespace-cdn.com/content/v1/58aa4fc51e5b6ce697e5bf40/1562930170757-AAI2VSWQ6K2YVEKA5ERH/House+-+dutch+gardens.jpg?format=2500w",
    #     "https://images.squarespace-cdn.com/content/v1/58aa4fc51e5b6ce697e5bf40/1562930170757-AAI2VSWQ6K2YVEKA5ERH/House+-+dutch+gardens.jpg?format=2500w",
    #    "https://expatsamsterdam.com/wp-content/uploads/2020/03/terraced-house-netherlands.jpg",
    #    "https://expatsamsterdam.com/wp-content/uploads/2020/03/terraced-house-netherlands.jpg",
        "https://i.pinimg.com/736x/67/90/a2/6790a2288d59b608f7f0f739e2f9cd58.jpg",
        "https://i.pinimg.com/1200x/f5/8f/5a/f58f5ab2652364a12c4ed7d5c897ccb9.jpg",
    #    "https://rotterdamwoont.nl/app/uploads/2018/01/69f94814c921049a2fbc79a58b5232a5-3.jpg",
    #    "https://as2.ftcdn.net/jpg/02/36/01/87/1000_F_236018798_o3c0Qwe73gFpSlTyEatwliBrEc6l7rrl.webp",
        year=1965,
        region="Netherlands" )

    # Stage 1: CLIP validations
    validate_clip_building(dataset, IMAGE_FOLDER)
    validate_clip_material(MATERIAL_ROOT, MATERIAL_FOLDERS)

    #Stage 2: LLM semantic validation over complete :
    benchmark_results = run_pipeline_benchmark(dataset, IMAGE_FOLDER)
    benchmark_results.to_csv("pipeline_benchmark_results.csv", index=False)
    print("\nResults saved to pipeline_benchmark_results3.csv")