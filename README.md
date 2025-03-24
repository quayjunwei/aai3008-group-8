# aai3008-group-8

## Contributions
- MUHAMMAD FIRDAUZ BIN KAMARULZAMAN
- SHERWYN CHAN YIN KIT
- MUHAMMAD AKID NUFAIRI BIN NASHILY
- DOMINICK SEAH ZI YU
- QUAY JUN WEI

## Project Description
A seamless system that allows users to upload educational videos in one language and receive the same content with high-quality translated and synchronised subtitles in their preferred language

1. **Automatic Speech Recognition (ASR)**: Converts spoken language in videos to text using OpenAI's Whisper model, which has been fine-tuned for several European languages in technical contexts.

2. **Transcript Processing**: Handles the alignment, cleaning, and segmentation of transcribed text. The system processes audio with silence detection using PyDub, performs timestamp adjustment, and formats transcriptions into SRT subtitle files. FFmpeg is used for embedding subtitles into videos.

Subtitle embedding is handled using FFmpeg:

```python
ffmpeg -i [video_path] -vf subtitles=[subtitle_path] -c:a copy [output_path]
```

This creates a video file with hard-coded subtitles while preserving the original audio quality.



3. **Language Model Translation (LLM)**: Translates processed text using a hybrid approach:
   - Primary translation using OpenAI's GPT-3.5-turbo API
   - Context enhancement through a Retrieval-Augmented Generation (RAG) system that:
     - Extracts key topics from subtitles using KeyBERT
     - Retrieves relevant Wikipedia content in the source language
     - Uses FAISS vector database for efficient similarity search
   - Fallback to Helsinki-NLP models when API fails or for specific language pairs

4. **Text-to-Speech (TTS)**: Converts translated text back to speech using Google's gTTS (Google Text-to-Speech) API with support for multiple languages.

The TTS module uses Google's gTTS API with support for multiple languages:

```python
from gtts import gTTS
tts = gTTS(text=text, lang=lang_code, slow=False)
```

5. **Streamlit UI**: Provides a user-friendly interface for uploading videos, displaying them, and overlaying translated text.

The system follows a comprehensive dataflow: Speech Input → ASR → Transcript Processing → RAG-enhanced LLM Translation → Output Generation (SRT files) → Video with Embedded Subtitles → (optional) TTS.

## Getting started

### We'll be using venv as our virtual environment
```
pip install virtualenv
```

#### 1. **Create virtual environment**

Mac
```
python3 -m venv myenv
```

Windows
```
python -m venv myenv
```

#### 2. Activate virtual environment

Mac
```
source myenv/bin/activate
```

Windows
```
source myenv/Scripts/activate
```


## Dependencies

This project requires the following key libraries:
- whisper (OpenAI's ASR model)
- OpenAI API (for GPT-3.5-turbo translation)
- sentence-transformers
- FAISS (Facebook AI Similarity Search)
- KeyBERT (for keyword extraction)
- NLTK (Natural Language Toolkit)
- Wikipedia API
- srt (subtitle processing)
- gTTS (Google Text-to-Speech)
- Transformers (Hugging Face)
- torch (PyTorch)
- FFmpeg (video processing)
- PyDub (audio processing and silence detection)
- Streamlit (web interface)

Make sure all dependencies are installed using the provided `requirements.txt` file.

## Environment Setup

Create a `.env` file in the project root with your OpenAI API key:
```
OPENAI_API_KEY=your_api_key_here
```

#### 4. Make sure you have OPENAI API Key located in
```
├── src/
│   ├── llm/.env               # API KEY LOCATION

OPENAI_API_KEY=INSERT YOUR API KEY HERE!
```

## Project Organisation
```
├── src/                       # Source code
│   ├── asr/                   # Automatic Speech Recognition module
|   ├── input/                 # Input data
│   ├── llm/                   # Language Learning Model module
|   ├── output/                # Output data
│   ├── transcript/            # Transcript processing module
|   ├── tts/                   # Text-to-Speech module
│   └── main.py                # Streamlit app
├── .gitignore                 # Git ignore file
├── README.md                  # Readme file
└── requirements.txt           # Project dependencies
```

## Running the Streamlit App

To run the Streamlit application, use the following command from the project root directory:

```
streamlit run src/main.py
```

## How to Use the App

1. **Upload Video**: Use the file upload feature to select and upload your video file for translation.

##### Sample video

A sample educational video in German can be downloaded [here](https://drive.google.com/file/d/1YGQOHM4f5TCwZB2N2JGsjmGGoqNRiktd/view?usp=sharing) 

2. **Select Languages**: Choose the source language of the video and the target language for translation. The system supports multiple languages including:
   - English, Spanish, French, German, Chinese, Japanese
   - Italian, Russian, Portuguese, Korean, Arabic, Hindi
   - Turkish, Dutch, Swedish, Polish, Indonesian, Ukrainian

3. **Process Video**: Click the "Process" button to start the translation pipeline:
   - The system transcribes the audio using the OpenAI Whisper ASR model
   - The transcript is cleaned and segmented with NLTK
   - Key topics are extracted from the content using KeyBERT
   - Relevant Wikipedia content in the source language is retrieved for context
   - The text is translated using GPT-3.5-turbo with RAG enhancement
   - SRT subtitle files are generated
   - FFmpeg embeds the subtitles into the video

4. **View Results**: The processed video will be displayed with translated text overlays synchronised with the original audio.

5. **Text-to-Speech**: You can optionally convert the translated subtitles to speech in the target language using the TTS feature.

6. **Download Options**:
   - Download the translated video with embedded subtitles
   - Download the SRT subtitle file separately

## Current Limitations and Future Work

- Improving RAG system with more diverse knowledge sources beyond Wikipedia
- Implementing a more robust fallback system for language pairs not well-supported by Helsinki-NLP
- Developing a comprehensive evaluation framework using BLEU, METEOR, chrF scores
- Enhancing the UI with real-time processing feedback and advanced customisation options
- Implementing an offline mode with downloadable models for environments without internet access