import argparse
import PyPDF2
import spacy
from sklearn.feature_extraction.text import TfidfVectorizer
import numpy as np

def remove_titles_and_speakers(raw_text):
    """
    Remove lines containing PDF title, speaker names, or any unwanted patterns.
    For example, lines with 'MITOCW', 'watch?v', 'YEN-JIE LEE', 'AUDIENCE'.
    Adjust keywords as necessary.
    """
    lines = raw_text.splitlines()
    cleaned_lines = []
    skip_keywords = ["MITOCW", "watch?v", "YEN-JIE LEE", "AUDIENCE"]
    for line in lines:
        # Check if line contains any unwanted keyword
        if any(keyword in line for keyword in skip_keywords):
            continue
        cleaned_lines.append(line)
    return "\n".join(cleaned_lines)

def remove_one_char_and_common_words(tokens, custom_stopwords=None):
    """
    Remove single-character tokens (e.g. 'a', 'F') and
    any additional custom stopwords (e.g. 'the', 'this', 'that').
    """
    if custom_stopwords is None:
        custom_stopwords = set()
    # spaCy's default stopwords can also be used if desired
    custom_stopwords.update(spacy.load("en_core_web_sm").Defaults.stop_words)
    
    filtered = []
    for tok in tokens:
        # Remove single-character tokens and those in custom stopwords
        if len(tok) <= 1:
            continue
        if tok.lower() in custom_stopwords:
            continue
        filtered.append(tok)
    return filtered

def pdf_to_text(pdf_path):
    """
    Extract text from all pages of a PDF file.
    """
    text = ""
    with open(pdf_path, "rb") as f:
        reader = PyPDF2.PdfReader(f)
        for page in reader.pages:
            page_text = page.extract_text()
            if page_text:
                text += page_text + "\n"
    return text

def clean_text(raw_text):
    """
    Apply cleaning steps: remove titles/speakers, etc.
    """
    cleaned = remove_titles_and_speakers(raw_text)
    return cleaned

def extract_technical_terms(text, nlp, custom_stopwords=None):
    """
    Extract technical terms using:
      1. Named Entity Recognition (NER)
      2. Noun chunks (POS tagging)
      3. TF-IDF to identify important words
    """

    if custom_stopwords is None:
        custom_stopwords = {"the", "this", "that", "and", "or", "for", 
                            "from", "with", "your", "about", "there"}
        
    excluded_labels = {
        "CARDINAL",
        "ORDINAL",
        "QUANTITY",
        "MONEY",
        "PERCENT",
        "DATE",
        "TIME",
    }

    # -- (A) NER --
    doc = nlp(text)
    named_entities = set()
    for ent in doc.ents:
        if ent.label_ in excluded_labels:
            print(f"Excluding {ent.text} {ent.label_}")
            continue

        # Filter single-character or unwanted stopwords
        tokens = remove_one_char_and_common_words(ent.text.split(),
                                                  custom_stopwords)
        # Rebuild text after filtering
        filtered_text = " ".join(tokens)
        if len(filtered_text.split()) >= 1:
            named_entities.add(filtered_text)
    print(f"Named Entities: {named_entities}\n")

    # -- (B) Noun chunks (compound nouns) --
    noun_chunks = set()
    for chunk in doc.noun_chunks:
        skip_chunk = False
        for token in chunk:
            if token.ent_type_ in excluded_labels:
                skip_chunk = True
                break
        if skip_chunk:
            continue

        tokens = remove_one_char_and_common_words(chunk.text.split(),
                                                  custom_stopwords)
        filtered_text = " ".join(tokens)
        if len(filtered_text.split()) >= 2:  # keep only multi-word
            noun_chunks.add(filtered_text)
    print(f"Noun Chunks: {noun_chunks}\n")

    # -- (C) TF-IDF on paragraphs --
    paragraphs = [p.strip() for p in text.split("\n") if p.strip()]
    if not paragraphs:
        paragraphs = [text]

    vectorizer = TfidfVectorizer(stop_words="english")
    tfidf_matrix = vectorizer.fit_transform(paragraphs)
    feature_names = vectorizer.get_feature_names_out()
    avg_scores = tfidf_matrix.mean(axis=0).A1
    tfidf_scores = dict(zip(feature_names, avg_scores))

    # Median threshold to select "important" words
    threshold = np.median(list(tfidf_scores.values()))
    important_terms = set()
    for term, score in tfidf_scores.items():
        if score > threshold:
            tokens = remove_one_char_and_common_words(term.split(),
                                                      custom_stopwords)
            filtered_text = " ".join(tokens)
            
            if len(filtered_text.split()) >= 1:
                important_terms.add(filtered_text)
    print(f"Important Terms (TF-IDF): {important_terms}\n")

    # Combine
    technical_terms = named_entities.union(noun_chunks).union(important_terms)
    return technical_terms

def main(pdf_path):
    print(f"Extracting technical terms from: {pdf_path}")
    # Load spaCy model
    # spacy.require_gpu()
    nlp = spacy.load("en_core_web_sm")
    print(f"Loaded spaCy model '{nlp.meta['name']}'")

    # Extract text from the PDF
    raw_text = pdf_to_text(pdf_path)
    if not raw_text.strip():
        print("No text could be extracted from the PDF.")
        return
    print(f"Extracted {len(raw_text)} characters from the PDF.")

    # Clean the text (remove bold titles, speaker names, etc.)
    cleaned = clean_text(raw_text)
    print(f"Cleaned text to {len(cleaned)} characters.")

    # Extract technical terms
    technical_terms = extract_technical_terms(cleaned, nlp)

    print("Extracted Technical Terms (multi-word, cleaned)")

    with open("D:/AAI3008 - Large Language Models/dataset/bias list/english/technical_terms.txt", "w", encoding="utf-8") as file:
        for term in sorted(technical_terms):
            file.write(term + "\n")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Extract technical terms from a PDF transcript using NER, TF-IDF, and POS tagging."
    )
    parser.add_argument("pdf_path", type=str, help="Path to the PDF file containing the transcript")
    args = parser.parse_args()
    main(args.pdf_path)
