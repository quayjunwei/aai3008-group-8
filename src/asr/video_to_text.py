import ffmpeg
import whisper
import jiwer
import csv
import os

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

def transcribe_audio(audio_path, output_srt, output_csv):
    # Transcribe audio to text
    model = whisper.load_model("tiny") # whichever model you choose (or custom one)
    result = model.transcribe(audio_path)
    # transcription = result["text"].strip()

    # Saving to SRT file
    with open(output_srt, "w", encoding="utf-8") as f:
        for segment in result["segments"]:
            start = segment["start"]
            end = segment["end"]
            subtitle = segment["text"]
            f.write(f"{start:.2f} --> {end:.2f}\n{subtitle}\n\n")
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

if __name__ == "__main__":
    video_file = "index.mp4"
    audio_file = "extracted_audio.mp3"
    subtitle_file = "subtitles.srt"
    csv_file = "subtitles.csv"

    extract_audio(video_file, audio_file)
    transcribe_audio(audio_file, subtitle_file, csv_file)

    os.remove(audio_file)