import wikipedia
import nltk
from sentence_transformers import SentenceTransformer
import faiss
import numpy as np
from keybert import KeyBERT
import srt
import openai
from transformers import pipeline
from dotenv import load_dotenv
import os

# Load environment variables from .env file
load_dotenv()

# Initialize the OpenAI client using the environment variable
client = openai.OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

# Download necessary NLTK data
nltk.download('punkt')
nltk.download('punkt_tab')

# Load the SentenceTransformer model for embeddings
device = 'cpu'  # Use 'cuda' if you have a GPU
embedding_model = SentenceTransformer('paraphrase-multilingual-MiniLM-L12-v2', device=device)

# Initialize KeyBERT for keyword extraction
kw_model = KeyBERT(model='paraphrase-multilingual-MiniLM-L12-v2')

# Global variables for RAG
index = None
corpus = []

# Language code mapping for Wikipedia and translation models
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

# Initialize the fallback translation model (Helsinki-NLP/opus-mt-de-en for German to English)
fallback_translator = pipeline("translation", model="Helsinki-NLP/opus-mt-de-en", device=-1)

# Set Wikipedia language based on source language
def set_wikipedia_language(source_lang):
    wiki_code = LANGUAGE_TO_WIKI_CODE.get(source_lang, "en")
    wikipedia.set_lang(wiki_code)
    print(f"Set Wikipedia language to: {wiki_code} (based on detected language: {source_lang})")

def extract_topics_from_subtitles(subtitles, top_n=5):
    """Extract key topics from subtitles using KeyBERT."""
    text = " ".join(subtitle.content for subtitle in subtitles)
    if not text.strip():
        print("No subtitle text found for keyword extraction.")
        return ["Mathematik", "Bildung"] if wikipedia.lang == "de" else ["Mathematics", "Education"]
    
    keywords = kw_model.extract_keywords(text, keyphrase_ngram_range=(1, 2), stop_words=None, top_n=top_n)
    topics = [keyword[0] for keyword in keywords]
    
    if not topics:
        print("No keywords extracted. Using fallback topics.")
        topics = ["Mathematik", "Bildung"] if wikipedia.lang == "de" else ["Mathematics", "Education"]
    
    return topics

def fetch_wikipedia_content(topics):
    """Fetch relevant content from Wikipedia based on extracted topics."""
    wiki_content = []
    for topic in topics:
        try:
            search_results = wikipedia.search(topic, results=1)
            if not search_results:
                print(f"No Wikipedia search results for '{topic}'.")
                continue
            
            page_title = search_results[0]
            page = wikipedia.page(page_title)
            sentences = nltk.sent_tokenize(page.content)
            wiki_content.extend(sentences[:50])
            print(f"Fetched content from Wikipedia page: '{page_title}'")
        except wikipedia.exceptions.DisambiguationError as e:
            print(f"Disambiguation error for '{topic}': {e.options}")
        except wikipedia.exceptions.PageError:
            print(f"Wikipedia page for '{topic}' not found.")
    return wiki_content

def build_rag_database(subtitles):
    """Build a RAG database using Wikipedia content extracted from subtitle topics."""
    global index, corpus

    print("Extracting topics from subtitles...")
    topics = extract_topics_from_subtitles(subtitles)
    if not topics:
        print("No topics extracted. Cannot build RAG database.")
        return False

    print(f"Extracted topics: {topics}")
    print("Fetching Wikipedia content...")
    corpus = fetch_wikipedia_content(topics)
    if not corpus:
        print("No Wikipedia content fetched. Cannot build RAG database.")
        return False

    print(f"Retrieved {len(corpus)} sentences from Wikipedia.")
    print("Building FAISS index...")
    embeddings = embedding_model.encode(corpus, convert_to_tensor=True, device=device)
    embeddings = embeddings.cpu().numpy()

    dimension = embeddings.shape[1]
    index = faiss.IndexFlatL2(dimension)
    index.add(embeddings)
    print("FAISS index built successfully.")
    return True

def translate_with_rag(subtitles_batch, source_lang, target_lang, k=3):
    """Translate a batch of subtitles using RAG to provide context, leveraging GPT-3.5-turbo with Helsinki-NLP fallback."""
    print(f"Translating batch of {len(subtitles_batch)} subtitles from {source_lang} to {target_lang}")
    
    if index is None or not corpus:
        print("RAG database not available. Performing direct translation with fallback.")
        return [fallback_translator(subtitle)[0]['translation_text'] for subtitle in subtitles_batch]
    
    print("Encoding query embedding for first subtitle in batch...")
    query_embedding = embedding_model.encode([subtitles_batch[0]], convert_to_tensor=True, device=device)
    query_embedding = query_embedding.cpu().numpy()
    print("Searching FAISS index...")
    distances, indices = index.search(query_embedding, k)
    retrieved_examples = [corpus[i] for i in indices[0]]

    prompt = f"""
    Translate the following {source_lang} subtitles into {target_lang.upper()}. Always translate every part of the text into {target_lang.upper()}, even if the context is unclear, technical, conversational, or mixed in style. Do not assume any part of the text is already in {target_lang.upper()} or does not need translation. Maintain the original meaning and adapt to the natural tone of the content, whether educational, narrative, or technical, across any field, using clear and fluent phrasing. Use the examples below as a loose reference for style and context, but prioritize complete and accurate translation over strict adherence to the examples.
    Here are some similar examples for reference (in {source_lang}):
        1. "{retrieved_examples[0]}"
        2. "{retrieved_examples[1]}"
        3. "{retrieved_examples[2]}"

    Subtitles to translate:
    """
    for i, subtitle in enumerate(subtitles_batch, 1):
        prompt += f"{i}. \"{subtitle}\"\n"

    try:
        response = client.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=[
                {"role": "system", "content": "You are a helpful assistant specializing in translation for educational content. Return translations as a numbered list (e.g., 1. [translation] 2. [translation]...). Do not include quotation marks in the translations."},
                {"role": "user", "content": prompt}
            ],
            max_tokens=500,
            temperature=0.3
        )
        translation_text = response.choices[0].message.content.strip()
        print(f"Raw API response: {translation_text}")
        translations = []
        for idx, line in enumerate(translation_text.split('\n')):
            if line.strip() and line[0].isdigit() and '.' in line:
                translation = line[line.index('.') + 1:].strip().strip('"')
                original_subtitle = subtitles_batch[idx].strip()
                translated_clean = translation.strip()
                if translated_clean == original_subtitle:
                    print(f"Warning: Translation for subtitle {idx + 1} ('{translation}') matches original ('{original_subtitle}'). Forcing fallback.")
                    fallback_translations = [fallback_translator(sub)[0]['translation_text'] for sub in subtitles_batch]
                    print(f"Fallback translations (Helsinki-NLP): {fallback_translations}")
                    return [t.strip('"') for t in fallback_translations]
                translations.append(translation)
        if not translations and translation_text.strip():
            print("No numbered list detected. Using single-line response as translation.")
            translations = [translation_text.strip().strip('"')]
        if not translations:
            print("Warning: No translations extracted from GPT-3.5-turbo response. Falling back to Helsinki-NLP.")
            fallback_translations = [fallback_translator(subtitle)[0]['translation_text'] for subtitle in subtitles_batch]
            print(f"Fallback translations (Helsinki-NLP): {fallback_translations}")
            return [t.strip('"') for t in fallback_translations]
        print(f"Translated batch to: {translations}")
        return translations
    except Exception as e:
        print(f"Translation error with GPT-3.5-turbo: {str(e)}. Falling back to Helsinki-NLP.")
        fallback_translations = [fallback_translator(subtitle)[0]['translation_text'] for subtitle in subtitles_batch]
        print(f"Fallback translations (Helsinki-NLP): {fallback_translations}")
        return [t.strip('"') for t in fallback_translations]

def process_srt(input_srt_path, output_srt_path, source_lang, target_lang, batch_size=5):
    """Process an SRT file, translating subtitles in batches using RAG."""
    print(f"Processing SRT file: {input_srt_path}")
    print(f"Source language: {source_lang}, Target language: {target_lang}")

    set_wikipedia_language(source_lang)

    with open(input_srt_path, 'r', encoding='utf-8') as f:
        subtitles = list(srt.parse(f))

    build_rag_database(subtitles)

    translated_subtitles = []
    batch = []
    batch_indices = []  # Track indices of subtitles in the batch

    for idx, subtitle in enumerate(subtitles):
        if subtitle.content.strip():
            batch.append(subtitle.content)
            batch_indices.append(idx)  # Store the original index
            if len(batch) >= batch_size:
                translated_texts = translate_with_rag(batch, source_lang, target_lang)
                for i, translated_text in enumerate(translated_texts):
                    original_idx = batch_indices[i]
                    new_subtitle = srt.Subtitle(
                        index=original_idx + 1,  # Preserve original index (1-based)
                        start=subtitles[original_idx].start,
                        end=subtitles[original_idx].end,
                        content=translated_text
                    )
                    # Ensure translated_subtitles is long enough
                    while len(translated_subtitles) <= original_idx:
                        translated_subtitles.append(None)
                    translated_subtitles[original_idx] = new_subtitle
                batch = []
                batch_indices = []
        else:
            new_subtitle = srt.Subtitle(
                index=idx + 1,
                start=subtitle.start,
                end=subtitle.end,
                content=subtitle.content
            )
            while len(translated_subtitles) <= idx:
                translated_subtitles.append(None)
            translated_subtitles[idx] = new_subtitle

    # Handle any remaining subtitles in the batch
    if batch:
        translated_texts = translate_with_rag(batch, source_lang, target_lang)
        for i, translated_text in enumerate(translated_texts):
            original_idx = batch_indices[i]
            new_subtitle = srt.Subtitle(
                index=original_idx + 1,
                start=subtitles[original_idx].start,
                end=subtitles[original_idx].end,
                content=translated_text
            )
            while len(translated_subtitles) <= original_idx:
                translated_subtitles.append(None)
            translated_subtitles[original_idx] = new_subtitle

    # Filter out any None values (just in case) and reindex
    translated_subtitles = [sub for sub in translated_subtitles if sub is not None]
    for i, sub in enumerate(translated_subtitles):
        sub.index = i + 1  # Reindex to ensure sequential indices

    with open(output_srt_path, 'w', encoding='utf-8') as f:
        f.write(srt.compose(translated_subtitles))
    print(f"Translated SRT file saved to: {output_srt_path}")
    return True