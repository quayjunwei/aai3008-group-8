import ffmpeg
import whisper
import jiwer
import csv
import os
import subprocess


#######################################################
## This is our initial ASR python, can delete? - Fir ##
#######################################################

def extract_audio(video_path, audio_path):
    # Extract audio from video
    ffmpeg.input(video_path).output(audio_path, acodec="mp3").run(overwrite_output=True)
    print(f"Audio extracted and saved at: {audio_path}")

def jiwer_transform(transcription):
    # To calculate WER and CER (not used to save as subtitles)
    transformation = jiwer.Compose([
    jiwer.ToLowerCase(),
    jiwer.RemovePunctuation(),
    jiwer.RemoveWhiteSpace(replace_by_space=True)])
    return transformation(transcription)

def seconds_to_srt_time(seconds):
    """Convert seconds to SRT time format: HH:MM:SS,mmm."""
    hours = int(seconds // 3600)
    minutes = int((seconds % 3600) // 60)
    secs = int(seconds % 60)
    milliseconds = int((seconds - int(seconds)) * 1000)
    return f"{hours:02}:{minutes:02}:{secs:02},{milliseconds:03}"

def transcribe_audio(audio_path, output_srt, output_csv):
    # Transcribe audio to text
    model = whisper.load_model("small") # whichever model you choose (or custom one)
    result = model.transcribe(audio_path)
    # transcription = result["text"].strip()

    # Saving to SRT file
    with open(output_srt, "w", encoding="utf-8") as f:
       for i, segment in enumerate(result["segments"], start=1):
            start_time = seconds_to_srt_time(segment["start"])
            end_time = seconds_to_srt_time(segment["end"])
            subtitle = segment["text"]
            f.write(f"{i}\n")
            f.write(f"{start_time} --> {end_time}\n")
            f.write(f"{subtitle}\n\n")
    print(f"Subtitles saved at: {output_srt}")

    # Saving to CSV file
    with open(output_csv, "w", encoding="utf-8") as f:
        fieldnames = ["start", "end", "text"]
        writer = csv.DictWriter(f, fieldnames=fieldnames, quoting=csv.QUOTE_ALL)
        writer.writeheader()
        for subtitle in result["segments"]:
            filtered_subtitle = { key: subtitle[key] for key in fieldnames if key in subtitle }
            writer.writerow(filtered_subtitle)
    print(f"Subtitles saved at: {output_csv}")

srt_file = "subtitles.srt"

if __name__ == "__main__":
    video_file = "input/german.mp4"
    audio_file = "extracted_audio.mp3"
    subtitle_file = "output/subtitles.srt"
    csv_file = "output/subtitles.csv"

    extract_audio(video_file, audio_file)
    transcribe_audio(audio_file, subtitle_file, csv_file)
    output_video = "output/german_with_sub.mp4"

    (
    ffmpeg
    .input(video_file)
    .output(output_video, vf=f"subtitles={subtitle_file}")
    .run()
    )

    os.remove(audio_file)