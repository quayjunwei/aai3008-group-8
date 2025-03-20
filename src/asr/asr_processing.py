import os
import whisper
import csv
import ffmpeg
from pydub import AudioSegment
from pydub.silence import detect_nonsilent
import torch

# Pair the language codes to their respective language
LANGUAGE_MAP = {
    'en': 'English',
    'es': 'Spanish',
    'fr': 'French',
    'de': 'German',
    'zh': 'Chinese',
    'ja': 'Japanese',
    'it': 'Italian',
    'ru': 'Russian',
    'pt': 'Portuguese',
    'ko': 'Korean',
    'ar': 'Arabic',
    'hi': 'Hindi',
    'tr': 'Turkish',
    'nl': 'Dutch',
    'sv': 'Swedish',
    'pl': 'Polish',
    'id': 'Indonesian',
    'uk': 'Ukrainian'
}

# USe CUDA if available 
def get_device():
    if torch.cuda.is_available():
        device = torch.device("cuda")
        print(f"Using GPU: {torch.cuda.get_device_name(0)}")
    else:
        device = torch.device("cpu")
        print("Using CPU")
    return device

# Using FFmpeg to extract the audio from the video
def extract_audio(video_path, audio_path):
    try:
        ffmpeg.input(video_path).output(audio_path, acodec="mp3").run(overwrite_output=True)
        print(f"Audio extracted and saved at: {audio_path}")
        return True, None
    except Exception as e:
        print(f"Extract audio error: {str(e)}")
        return False, str(e)

# Set the subtitle timing to a proper HH:MM:SS,MSS (hour:min:second,milisecond) format which is the standard for .srt
def seconds_to_srt_time(seconds):
    hours = int(seconds // 3600)
    minutes = int((seconds % 3600) // 60)
    secs = int(seconds % 60)
    milliseconds = int((seconds - int(seconds)) * 1000)
    return f"{hours:02}:{minutes:02}:{secs:02},{milliseconds:03}"

def transcribe_audio(audio_path, output_srt, output_csv):
    try:
        device = get_device()
        model = whisper.load_model("small", device=device)

        # Optimize GPU performance
        torch.backends.cudnn.benchmark = True if device.type == "cuda" else False

        # Load audio and detect non-silent ranges
        audio = AudioSegment.from_file(audio_path)
        # Detect silence with adjusted parameters (modify values as needed)
        nonsilent_ranges = detect_nonsilent(
            audio,
            min_silence_len=300,    # Reduced from 500ms for better detection
            silence_thresh=-40      # dBFS threshold
        )

        # Trim audio to exclude leading/trailing silence
        if nonsilent_ranges:
            start_trim = nonsilent_ranges[0][0]  # Start of first non-silent in ms
            end_trim = nonsilent_ranges[-1][1]   # End of last non-silent in ms
            trimmed_audio = audio[start_trim:end_trim]
        else:
            # Handle completely silent audio
            trimmed_audio = audio[:0]  # Create empty audio segment

        # Export trimmed audio to temporary file
        temp_audio_path = f"trimmed_{os.path.basename(audio_path)}"
        trimmed_audio.export(temp_audio_path, format="mp3")

        # Transcribe trimmed audio
        result = model.transcribe(temp_audio_path)

        # Calculate offset to adjust timestamps (convert ms to seconds)
        offset = start_trim / 1000 if nonsilent_ranges else 0

        # Process segments and adjust timestamps
        adjusted_segments = []
        for segment in result["segments"]:
            # Add offset to map back to original timeline
            adj_start = segment["start"] + offset
            adj_end = segment["end"] + offset

            # Filter short segments (minimum 0.5 seconds duration)
            if (adj_end - adj_start) >= 0.5:
                adjusted_segments.append({
                    "start": adj_start,
                    "end": adj_end,
                    "text": segment["text"].strip()
                })

        # Generate SRT file with corrected timestamps
        with open(output_srt, "w", encoding="utf-8") as f:
            for i, segment in enumerate(adjusted_segments, start=1):
                start_time = seconds_to_srt_time(segment["start"])
                end_time = seconds_to_srt_time(segment["end"])
                f.write(f"{i}\n{start_time} --> {end_time}\n{segment['text']}\n\n")

        # Generate CSV report
        with open(output_csv, "w", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=["start", "end", "text"], quoting=csv.QUOTE_ALL)
            writer.writeheader()
            writer.writerows(adjusted_segments)

        # Language detection
        lang_code = result.get("language", "Unknown").lower()
        detected_lang = LANGUAGE_MAP.get(lang_code, f"Unknown ({lang_code})")

        # Cleanup temporary audio file
        os.remove(temp_audio_path)

        return True, None, detected_lang
    
    except Exception as e:
        print(f"Transcribe error: {str(e)}")
        return False, str(e), None