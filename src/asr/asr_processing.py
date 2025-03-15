import os
import whisper
import csv
import subprocess
import ffmpeg
import shutil
from pathlib import Path

def ensure_dir(directory):
    if not os.path.exists(directory):
        os.makedirs(directory)

def extract_audio(video_path, audio_path):
    try:
        ffmpeg.input(video_path).output(audio_path, acodec="mp3").run(overwrite_output=True)
        print(f"Audio extracted and saved at: {audio_path}")
        return True, None
    except Exception as e:
        print(f"Extract audio error: {str(e)}")
        return False, str(e)

def seconds_to_srt_time(seconds):
    hours = int(seconds // 3600)
    minutes = int((seconds % 3600) // 60)
    secs = int(seconds % 60)
    milliseconds = int((seconds - int(seconds)) * 1000)
    return f"{hours:02}:{minutes:02}:{secs:02},{milliseconds:03}"

def transcribe_audio(audio_path, output_srt, output_csv):
    try:
        model = whisper.load_model("small")
        result = model.transcribe(audio_path)
        
        # Add language detection
        detected_lang = result.get("language", "Unknown").capitalize()

        with open(output_srt, "w", encoding="utf-8") as f:
            for i, segment in enumerate(result["segments"], start=1):
                start_time = seconds_to_srt_time(segment["start"])
                end_time = seconds_to_srt_time(segment["end"])
                f.write(f"{i}\n{start_time} --> {end_time}\n{segment['text']}\n\n")
        
        with open(output_csv, "w", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=["start", "end", "text"], quoting=csv.QUOTE_ALL)
            writer.writeheader()
            for subtitle in result["segments"]:
                writer.writerow({k: subtitle[k] for k in ["start", "end", "text"]})
        
        return True, None, detected_lang
    except Exception as e:
        print(f"Transcribe error: {str(e)}")
        return False, str(e), None

def embed_subtitles(video_path, subtitle_path, output_path):
    try:
        video_dir = os.path.dirname(video_path)
        temp_srt = os.path.join(video_dir, "temp.srt")
        shutil.copy(subtitle_path, temp_srt)
        
        cmd = [
            'ffmpeg',
            '-i', video_path,
            '-vf', f'subtitles={os.path.basename(temp_srt)}',
            '-c:a', 'copy',
            output_path,
            '-y'
        ]
        
        original_dir = os.getcwd()
        os.chdir(video_dir)
        process = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        os.chdir(original_dir)
        
        if process.returncode != 0:
            raise Exception(f"FFmpeg error: {process.stderr.decode()}")
            
        return True, None
    except Exception as e:
        print(f"Embed subtitles error: {str(e)}")
        return False, str(e)
    finally:
        if os.path.exists(temp_srt):
            os.remove(temp_srt)