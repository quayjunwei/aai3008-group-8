from gtts import gTTS
import io


def generate_speech(text, lang="en"):
    """
    Generate speech from text and return audio bytes directly

    Parameters:
    text (str): Text to convert to speech
    lang (str): Language code (e.g., 'en', 'es', 'fr', 'de')

    Returns:
    bytes: Audio data as bytes
    """
    try:
        # Create a bytes buffer
        mp3_fp = io.BytesIO()

        # Generate speech directly to the buffer
        tts = gTTS(text=text, lang=lang, slow=False)
        tts.write_to_fp(mp3_fp)

        # Get the bytes from the buffer
        mp3_fp.seek(0)
        audio_bytes = mp3_fp.read()

        return audio_bytes

    except Exception as e:
        print(f"Text-to-speech error: {str(e)}")
        return None


def get_language_code(language_name):
    """
    Convert language name to ISO language code for gTTS

    Parameters:
    language_name (str): Full language name

    Returns:
    str: ISO language code
    """
    language_map = {
        "English": "en",
        "Spanish": "es",
        "French": "fr",
        "German": "de",
        "Chinese": "zh-CN",
        "Japanese": "ja",
    }

    return language_map.get(language_name, "en")  # Default to English if not found
