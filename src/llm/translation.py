import wikipedia
import re
import os
import torch
import faiss
from sentence_transformers import SentenceTransformer
from transformers import pipeline
from keybert import KeyBERT

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

# Extract keywords from subtitles
def extract_topics_from_subtitles(input_srt, source_lang, top_n=3): # top_n=3 means we want to extract 3 keywords from the subtitles
    # Read the subtitles from the SRT file
    subtitle_text = ""
    with open(input_srt, "r", encoding="utf-8") as f:
        lines = f.readlines()
        for i in range(len(lines)):
            if lines[i].strip().isdigit() and i + 2 < len(lines):
                subtitle_text += lines[i + 2].strip() + " "

    if not subtitle_text:
        print("No subtitle text found for keyword extraction.")
        # Fallback to generic topics
        return ["Mathematics", "Education"] if source_lang == "English" else ["Mathematik", "Bildung"] # fallback topics set to Mathematics and Education (can be changed or removed)

    # Initialize KeyBERT with a multilingual model
    kw_model = KeyBERT(model='paraphrase-multilingual-MiniLM-L12-v2')

    # Extract keywords
    keywords = kw_model.extract_keywords(
        subtitle_text,
        keyphrase_ngram_range=(1, 2),  # Extract 1-2 word phrases (can be changed)
        stop_words=None,  # Let KeyBERT handle stop words for the language
        top_n=top_n,
        diversity=0.5  # Ensure diverse keywords
    )

    # Extract just the keywords (not their scores)
    topics = [keyword[0] for keyword in keywords]
    print(f"Extracted topics from subtitles: {topics}")

    # If no keywords are extracted, fall back to generic topics (again, can be changed or removed)
    if not topics:
        print("No keywords extracted. Using fallback topics.")
        topics = ["Mathematics", "Education"] if source_lang == "English" else ["Mathematik", "Bildung"]

    return topics

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
    embeddings = embeddings.cpu().numpy()
    dimension = embeddings.shape[1]
    print(f"Creating FAISS index with dimension {dimension}...")
    index = faiss.IndexFlatL2(dimension)
    print("Adding embeddings to index...")
    index.add(embeddings)
    return index, corpus

# Translate subtitle with RAG
def translate_with_rag(subtitle, source_lang, target_lang, k=3):
    print(f"Translating '{subtitle}' from {source_lang} to {target_lang}")
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
        "fr": f"Helsinki-NLP/opus-mt-{source_code}-fr",
        "de": f"Helsinki-NLP/opus-mt-{source_code}-de"
    }

    if target_lang not in language_models:
        print(f"Unsupported target language: {target_lang}. Supported languages: {list(language_models.keys())}")
        return subtitle

    print(f"Loading model for {target_lang} from {language_models[target_lang]}...")
    try:
        translator = pipeline("translation", model=language_models[target_lang], device=-1)
        print("Model loaded successfully.")
    except Exception as e:
        print(f"Error loading translation model for {source_lang} to {target_lang}: {str(e)}")
        try:
            from transformers import MarianTokenizer, MarianMTModel
            model_name = language_models[target_lang]
            tokenizer = MarianTokenizer.from_pretrained(model_name)
            model = MarianMTModel.from_pretrained(model_name)
            translator = lambda x: tokenizer.decode(model.generate(**tokenizer(x, return_tensors="pt"))[0], skip_special_tokens=True)
            print("Fallback model loaded successfully.")
        except Exception as e2:
            print(f"Fallback failed: {str(e2)}. Returning original text.")
            return subtitle

    prompt = f"""
    Translate the following {source_lang} subtitle into {target_lang.upper()}. Maintain the educational meaning and context.
    Here are some similar examples for reference (in {source_lang}):
    1. "{retrieved_examples[0]}"
    2. "{retrieved_examples[1]}"
    3. "{retrieved_examples[2]}"
    
    Subtitle to translate: "{subtitle}"
    """

    print("Performing translation...")
    try:
        translation = translator(subtitle)[0]['translation_text'] if callable(getattr(translator, 'predict', None)) else translator(subtitle)
        print(f"Translated to: {translation}")
        return translation
    except Exception as e:
        print(f"Translation error for subtitle '{subtitle}': {str(e)}")
        return subtitle

# Process SRT file with translation
index = None
corpus = None

def process_srt(input_srt, output_srt, source_lang, target_lang):
    set_wikipedia_language(source_lang)
    # Extract topics from subtitles
    print("Extracting topics from subtitles...")
    topics = extract_topics_from_subtitles(input_srt, source_lang, top_n=3)
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