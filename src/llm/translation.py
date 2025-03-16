import wikipedia
import re
import os
import torch
import faiss
from sentence_transformers import SentenceTransformer
from transformers import pipeline

# Ensure sentencepiece is installed
try:
    import sentencepiece
except ImportError:
    raise ImportError("Please install sentencepiece: pip install sentencepiece")

# Device setup
device = torch.device("cpu")
print(f"Using device: {device}")

# Language code mapping for Wikipedia
LANGUAGE_TO_WIKI_CODE = {
    "English": "en",
    "Spanish": "es",
    "French": "fr",
    "German": "de",
    "Chinese": "zh",
    "Japanese": "ja",
    "Italian": "it",
    "Russian": "ru",
    "Portuguese": "pt",
    "Korean": "ko",
    "Arabic": "ar",
    "Hindi": "hi",
    "Turkish": "tr",
    "Dutch": "nl",
    "Swedish": "sv",
    "Polish": "pl",
    "Indonesian": "id",
    "Ukrainian": "uk"
}

# Set Wikipedia language based on source language
def set_wikipedia_language(source_lang):
    wiki_code = LANGUAGE_TO_WIKI_CODE.get(source_lang, "en")
    wikipedia.set_lang(wiki_code)
    print(f"Set Wikipedia language to: {wiki_code} (based on detected language: {source_lang})")

# Fetch content from Wikipedia
def fetch_wikipedia_content(topics, num_sentences=10):
    corpus = []
    for topic in topics:
        try:
            page = wikipedia.page(topic)
            # Split into sentences and take the first num_sentences
            sentences = re.split(r'(?<=[.!?])\s+', page.content)[:num_sentences]
            corpus.extend(sentences)
        except wikipedia.exceptions.DisambiguationError as e:
            print(f"Disambiguation error for {topic}: {e.options}")
        except wikipedia.exceptions.PageError:
            print(f"Page not found for {topic}")
    return corpus

# Build RAG vector store with CPU support
embedding_model = SentenceTransformer('paraphrase-multilingual-MiniLM-L12-v2', device=device)

def build_rag_database(corpus):
    print("Encoding embeddings...")
    embeddings = embedding_model.encode(corpus, convert_to_tensor=True, device=device)
    print("Moving embeddings to CPU for FAISS...")
    embeddings = embeddings.cpu().numpy()  # Ensure CPU compatibility
    dimension = embeddings.shape[1]
    print(f"Creating FAISS index with dimension {dimension}...")
    index = faiss.IndexFlatL2(dimension)
    print("Adding embeddings to index...")
    index.add(embeddings)
    return index, corpus

# Translate subtitle with RAG
def translate_with_rag(subtitle, source_lang, target_lang, k=3):
    # Retrieve similar examples using RAG
    print("Encoding query embedding...")
    query_embedding = embedding_model.encode([subtitle], convert_to_tensor=True, device=device)
    query_embedding = query_embedding.cpu().numpy()
    print("Searching FAISS index...")
    distances, indices = index.search(query_embedding, k)
    retrieved_examples = [corpus[i] for i in indices[0]]

    # Map source language to model code
    SOURCE_TO_MODEL_CODE = {
        "English": "en",
        "Spanish": "es",
        "French": "fr",
        "German": "de",
        "Chinese": "zh",
        "Japanese": "ja",
        "Italian": "it",
        "Russian": "ru",
        "Portuguese": "pt",
        "Korean": "ko",
        "Arabic": "ar",
        "Hindi": "hi",
        "Turkish": "tr",
        "Dutch": "nl",
        "Swedish": "sv",
        "Polish": "pl",
        "Indonesian": "id",
        "Ukrainian": "uk"
    }

    source_code = SOURCE_TO_MODEL_CODE.get(source_lang, "en")
    language_models = {
        "en": f"Helsinki-NLP/opus-mt-{source_code}-en",
        "es": f"Helsinki-NLP/opus-mt-{source_code}-es",
        "fr": f"Helsinki-NLP/opus-mt-{source_code}-fr"
    }

    if target_lang not in language_models:
        print(f"Unsupported target language: {target_lang}. Supported languages: {list(language_models.keys())}")
        return subtitle

    # Load the translation model
    try:
        translator = pipeline("translation", model=language_models[target_lang], device=-1)  # -1 forces CPU
    except Exception as e:
        print(f"Error loading translation model for {source_lang} to {target_lang}: {str(e)}")
        return subtitle

    # Build prompt with retrieved examples (for future use with more advanced models)
    prompt = f"""
    Translate the following {source_lang} subtitle into {target_lang.upper()}. Maintain the educational meaning and context.
    Here are some similar examples for reference (in {source_lang}):
    1. "{retrieved_examples[0]}"
    2. "{retrieved_examples[1]}"
    3. "{retrieved_examples[2]}"
    
    Subtitle to translate: "{subtitle}"
    """

    # Translate the subtitle
    try:
        translation = translator(subtitle)[0]['translation_text']
    except Exception as e:
        print(f"Translation error for subtitle '{subtitle}': {str(e)}")
        return subtitle

    return translation

# Process SRT file with translation
index = None
corpus = None

def process_srt(input_srt, output_srt, source_lang, target_lang):
    set_wikipedia_language(source_lang)
    topics = ["Vector (mathematics)", "Linear algebra", "Mathematics"] if source_lang == "English" else ["Vektor (Mathematik)", "Lineare Algebra", "Grundlagen der Mathematik"]
    print("Fetching Wikipedia content...")
    global corpus
    corpus = fetch_wikipedia_content(topics)
    print(f"Retrieved {len(corpus)} sentences for RAG database.")

    print("Building RAG vector store...")
    global index
    index, corpus = build_rag_database(corpus)

    with open(input_srt, "r", encoding="utf-8") as f:
        lines = f.readlines()

    with open(output_srt, "w", encoding="utf-8") as f_out:
        i = 0
        while i < len(lines):
            if lines[i].strip().isdigit():
                f_out.write(lines[i])  # Write index
                f_out.write(lines[i+1])  # Write timestamp
                subtitle_text = lines[i+2].strip()
                translated_text = translate_with_rag(subtitle_text, source_lang, target_lang)
                f_out.write(f"{translated_text}\n\n")
                i += 4
            else:
                i += 1