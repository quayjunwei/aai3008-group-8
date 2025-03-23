import os
import streamlit as st
import csv
from asr.asr_processing import extract_audio, transcribe_audio
from transcript.transcript_processing import embed_subtitles
from llm.translation import process_srt
from tts.text_to_speech import generate_speech, get_language_code

st.set_page_config(page_title="Educational Video Translation System", layout="wide")

# CSS to improve audio player positioning and spacing
st.markdown(
    """
<style>
/* Add more space between transcript segments */
.element-container:has(.stAudio) {
    margin-top: 35px;
    margin-bottom: 15px;
}

/* Adjust divider spacing */
.element-container:has(hr) {
    margin-top: 25px;
}

/* Add space before timestamp */
.element-container:has(strong) {
    margin-top: 20px;
}
</style>
""",
    unsafe_allow_html=True,
)


def ensure_dir(directory):
    if not os.path.exists(directory):
        os.makedirs(directory)


# Create directories
script_dir = os.path.dirname(os.path.abspath(__file__))
input_dir = os.path.join(script_dir, "input")
output_dir = os.path.join(script_dir, "output")
ensure_dir(input_dir)
ensure_dir(output_dir)

# Streamlit UI
st.title("Educational Video Translation System")

# Initialize session state with default values
if "processed" not in st.session_state:
    st.session_state.processed = False
    st.session_state.uploaded_file = None
    st.session_state.selected_lang = "auto-detect"
    st.session_state.detected_lang = None
    st.session_state.output_video_path = None
    st.session_state.srt_path = None
    st.session_state.csv_path = None
    st.session_state.translated_video_path = None
    st.session_state.input_video_path = None

col1, col2 = st.columns(2)

with col1:
    if not st.session_state.processed:
        # Upload section
        st.subheader("Upload Video")
        uploaded_file = st.file_uploader(
            "Choose a video file",
            type=["mp4", "mov", "avi", "mkv"],
            key="file_uploader",
        )

        if uploaded_file:
            st.session_state.uploaded_file = uploaded_file
            input_video_path = os.path.join(input_dir, "input_video.mp4")
            st.session_state.input_video_path = input_video_path
            with open(input_video_path, "wb") as f:
                f.write(uploaded_file.getbuffer())
            st.video(input_video_path)

            # Language selection
            st.session_state.selected_lang = st.selectbox(
                "Select video language",
                [
                    "auto-detect",
                    "English",
                    "Spanish",
                    "French",
                    "German",
                    "Chinese",
                    "Japanese",
                ],
            )

            # Process button
            if st.button("Generate Video with Subtitles"):
                with st.spinner("Processing video..."):
                    # Processing pipeline
                    audio_path = os.path.join(output_dir, "extracted_audio.mp3")
                    srt_path = os.path.join(output_dir, "subtitles.srt")
                    csv_path = os.path.join(output_dir, "subtitles.csv")
                    output_video_path = os.path.join(
                        output_dir, "video_with_subtitles.mp4"
                    )

                    success, error = extract_audio(input_video_path, audio_path)
                    if not success:
                        st.error(f"Audio extraction failed: {error}")
                        st.stop()

                    success, error, detected_lang = transcribe_audio(
                        audio_path, srt_path, csv_path
                    )
                    if not success:
                        st.error(f"Transcription failed: {error}")
                        st.stop()

                    success, error = embed_subtitles(
                        input_video_path, srt_path, output_video_path
                    )
                    if not success:
                        st.error(f"Subtitle embedding failed: {error}")
                        st.stop()

                    # Update session state
                    st.session_state.processed = True
                    st.session_state.detected_lang = detected_lang
                    st.session_state.output_video_path = output_video_path
                    st.session_state.srt_path = srt_path
                    st.session_state.csv_path = csv_path
                    st.session_state.uploaded_file = None
                    st.rerun()

    if st.session_state.processed:
        # Results display
        st.subheader("Processed Video")
        if st.session_state.output_video_path and os.path.exists(
            st.session_state.output_video_path
        ):
            st.video(st.session_state.output_video_path)
        else:
            st.warning("Video not processed yet.")

        # Language display
        lang_source = (
            "Detected"
            if st.session_state.selected_lang == "auto-detect"
            else "Selected"
        )
        display_lang = (
            st.session_state.detected_lang
            if st.session_state.selected_lang == "auto-detect"
            else st.session_state.selected_lang
        )
        st.markdown(f"**{lang_source} Language:** {display_lang or 'Unknown'}")

        download_col1, download_col2, download_col3 = st.columns(3)

        if st.session_state.output_video_path and os.path.exists(
            st.session_state.output_video_path
        ):
            with open(st.session_state.output_video_path, "rb") as f:
                with download_col1:
                    st.download_button(
                        "Download Subtitled Video", f, "video_with_subtitles.mp4"
                    )

        if st.session_state.srt_path and os.path.exists(st.session_state.srt_path):
            with open(st.session_state.srt_path, "rb") as f:
                with download_col2:
                    st.download_button("Download SRT Subtitles", f, "subtitles.srt")

        if st.session_state.csv_path and os.path.exists(st.session_state.csv_path):
            with open(st.session_state.csv_path, "rb") as f:
                with download_col3:
                    st.download_button("Download Transcript CSV", f, "subtitles.csv")

        # Full transcript with TTS feature
        st.subheader("Full Transcript")
        if st.session_state.csv_path and os.path.exists(st.session_state.csv_path):
            try:
                with open(st.session_state.csv_path, "r", encoding="utf-8") as f:
                    reader = csv.DictReader(f)
                    for i, row in enumerate(reader):
                        # Create a container for this segment
                        segment_container = st.container()

                        with segment_container:
                            # Display timestamp and text
                            st.write(
                                f"**{float(row['start']):.1f}s - {float(row['end']):.1f}s**"
                            )

                            # Create columns for text and audio button (more space for text)
                            text_col, audio_col = st.columns([18, 1])

                            with text_col:
                                st.write(row["text"])

                            with audio_col:
                                # Add audio button
                                if st.button("🔊", key=f"btn_orig_{i}"):
                                    # Generate speech on demand when the button is clicked
                                    lang_code = (
                                        get_language_code(display_lang)
                                        if display_lang
                                        else "en"
                                    )
                                    audio_data = generate_speech(
                                        row["text"], lang=lang_code
                                    )

                                    # Store the audio in session state
                                    st.session_state[f"audio_orig_{i}"] = audio_data

                        # If audio exists in session state for this segment, play it
                        audio_key = f"audio_orig_{i}"
                        if (
                            audio_key in st.session_state
                            and st.session_state[audio_key] is not None
                        ):
                            # Add extra spacing for the audio player
                            st.audio(st.session_state[audio_key], format="audio/mp3")

                        # Add a divider with extra space
                        st.divider()
            except Exception as e:
                st.error(f"Error loading transcript: {str(e)}")
        else:
            st.warning("Transcript not available. Please process the video first.")

with col2:
    # Initialize translation state
    if "translated" not in st.session_state:
        st.session_state.translated = False

    if st.session_state.processed:
        if not st.session_state.translated:
            # Translation input section
            st.subheader("Translate Subtitles")
            target_lang = st.selectbox(
                "Translate subtitles to:",
                ["en", "es", "fr"],
                index=0,
                format_func=lambda x: {
                    "en": "English",
                    "es": "Spanish",
                    "fr": "French",
                }[x],
            )

            if st.button("Translate and Generate Video"):
                with st.spinner("Translating and embedding subtitles..."):
                    if st.session_state.srt_path and os.path.exists(st.session_state.srt_path):
                        # Store the target language code in session state for later use
                        st.session_state.target_lang_code = target_lang

                        translated_srt = os.path.join(
                            output_dir, f"subtitles_{target_lang}.srt"
                        )
                        translated_csv = os.path.join(
                            output_dir, f"subtitles_{target_lang}.csv"
                        )  # New CSV file
                        source_lang = st.session_state.detected_lang

                        # Process both SRT and CSV
                        process_srt(
                            st.session_state.srt_path,
                            translated_srt,
                            source_lang,
                            target_lang,
                        )

                        # Convert translated SRT to CSV
                        with open(
                            translated_srt, "r", encoding="utf-8"
                        ) as srt_file, open(
                            translated_csv, "w", newline="", encoding="utf-8"
                        ) as csv_file:
                            writer = csv.writer(csv_file)
                            writer.writerow(["start", "end", "text"])

                            while True:
                                index = srt_file.readline()
                                if not index:
                                    break
                                times = srt_file.readline().strip()
                                text = srt_file.readline().strip()
                                srt_file.readline()  # empty line

                                # Parse times
                                start_end = times.split(" --> ")
                                start = start_end[0].replace(",", ".")
                                end = start_end[1].replace(",", ".")

                                writer.writerow([start, end, text])

                        translated_video = os.path.join(
                            output_dir, f"video_{target_lang}.mp4"
                        )
                        success, error = embed_subtitles(
                            st.session_state.input_video_path,
                            translated_srt,
                            translated_video,
                        )

                        if success:
                            # New code to add TTS audio
                            try:
                                from pydub import AudioSegment
                                import subprocess
                                from io import BytesIO

                                # Read translated CSV
                                segments = []
                                with open(translated_csv, 'r', encoding='utf-8') as f:
                                    reader = csv.DictReader(f)
                                    for row in reader:
                                        segments.append({
                                            'start': row['start'],
                                            'text': row['text']
                                        })

                                # Get video duration using ffprobe
                                def get_video_duration(video_path):
                                    cmd = [
                                        'ffprobe', '-v', 'error',
                                        '-show_entries', 'format=duration',
                                        '-of', 'default=noprint_wrappers=1:nokey=1', video_path
                                    ]
                                    result = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
                                    return float(result.stdout.decode().strip())

                                duration = get_video_duration(st.session_state.input_video_path)
                                silent_audio = AudioSegment.silent(int(duration * 1000))  # in milliseconds

                                # Overlay TTS segments
                                for seg in segments:
                                    start_str = seg['start'].replace(",", ".")
                                    parts = start_str.split(':')
                                    start_sec = sum(float(p) * 60**i for i, p in enumerate(reversed(parts)))
                                    start_ms = int(start_sec * 1000)

                                    audio_bytes = generate_speech(seg['text'], lang=target_lang)
                                    if not audio_bytes:
                                        continue  # Skip failed segments
                                    audio = AudioSegment.from_file(BytesIO(audio_bytes), format="mp3")
                                    silent_audio = silent_audio.overlay(audio, position=start_ms)

                                # Save combined TTS audio
                                tts_audio_path = os.path.join(output_dir, f"tts_{target_lang}.mp3")
                                silent_audio.export(tts_audio_path, format="mp3")

                                # Merge TTS audio with translated video
                                final_video_path = os.path.join(output_dir, f"video_{target_lang}_tts.mp4")
                                cmd = [
                                    'ffmpeg', '-i', translated_video,
                                    '-i', tts_audio_path,
                                    '-c:v', 'copy', '-c:a', 'aac',
                                    '-map', '0:v', '-map', '1:a',
                                    '-shortest', '-y', final_video_path
                                ]
                                subprocess.run(cmd, check=True)

                                # Update session state with new video
                                st.session_state.translated_video_path = final_video_path
                            
                            except Exception as e:
                                st.error(f"TTS audio merge failed: {str(e)}")
                                st.stop()

                            # st.session_state.translated_video_path = translated_video
                            st.session_state.translated_srt_path = translated_srt
                            st.session_state.translated_csv_path = translated_csv
                            st.session_state.translated = True
                            st.session_state.target_lang = {
                                 "en": "English",
                                 "es": "Spanish",
                                 "fr": "French",
                            }[target_lang]
                            st.rerun()
                        else:
                            st.error(f"Translation embedding failed: {error}")
                    else:
                        st.error("Subtitles file not available")

        if st.session_state.translated:
            # Translated results display
            st.subheader("Translated Video")

            if st.session_state.translated_video_path and os.path.exists(
                st.session_state.translated_video_path
            ):
                st.video(st.session_state.translated_video_path)

                # Display chosen translated language
                st.markdown(f"**Translated Language:** {st.session_state.target_lang}")

                translate_col1, translate_col2 = st.columns(2)
                # Download translated video button
                with open(st.session_state.translated_video_path, "rb") as f:
                    with translate_col1:
                        st.download_button(
                            "Download Translated Video",
                            f,
                            file_name=f"video_translated_{st.session_state.target_lang_code}.mp4",
                            key="translated_video_download",
                        )

                # Retranslate option button
                with translate_col2:
                    if st.button("Translate to Another Language"):
                        st.session_state.translated = False
                        st.session_state.translated_video_path = None
                        st.session_state.target_lang = None
                        st.session_state.target_lang_code = None
                        st.rerun()

                # Translated transcript section with TTS feature
                st.subheader("Translated Transcript")
                if st.session_state.translated_csv_path and os.path.exists(
                    st.session_state.translated_csv_path
                ):
                    try:
                        with open(
                            st.session_state.translated_csv_path, "r", encoding="utf-8"
                        ) as f:
                            reader = csv.DictReader(f)
                            for i, row in enumerate(reader):
                                # Create a container for this segment
                                segment_container = st.container()

                                with segment_container:
                                    # Convert SRT time format to seconds
                                    start_sec = sum(
                                        x * float(t)
                                        for x, t in zip(
                                            [3600, 60, 1],
                                            row["start"].replace(",", ".").split(":"),
                                        )
                                    )
                                    end_sec = sum(
                                        x * float(t)
                                        for x, t in zip(
                                            [3600, 60, 1],
                                            row["end"].replace(",", ".").split(":"),
                                        )
                                    )

                                    # Display timestamp and text
                                    st.write(f"**{start_sec:.1f}s - {end_sec:.1f}s**")

                                    # Create columns for text and audio button (more space for text)
                                    text_col, audio_col = st.columns([18, 1])

                                    with text_col:
                                        st.write(row["text"])

                                    with audio_col:
                                        # Add audio button
                                        if st.button("🔊", key=f"btn_trans_{i}"):
                                            # Generate speech on demand
                                            audio_data = generate_speech(
                                                row["text"],
                                                lang=st.session_state.target_lang_code,
                                            )

                                            # Store the audio in session state
                                            st.session_state[f"audio_trans_{i}"] = (
                                                audio_data
                                            )

                                # If audio exists in session state for this segment, play it
                                audio_key = f"audio_trans_{i}"
                                if (
                                    audio_key in st.session_state
                                    and st.session_state[audio_key] is not None
                                ):
                                    # Add extra spacing for the audio player
                                    st.audio(
                                        st.session_state[audio_key], format="audio/mp3"
                                    )

                                # Add a divider with extra space
                                st.divider()
                    except Exception as e:
                        st.error(f"Error loading translated transcript: {str(e)}")
                else:
                    st.warning("Translated transcript not available")

            else:
                st.warning("Translated video not found")

    else:
        st.subheader("Translated Video")
        st.info("Please process a video first to enable translation")

# Cleanup
if os.path.exists(os.path.join(output_dir, "extracted_audio.mp3")):
    try:
        os.remove(os.path.join(output_dir, "extracted_audio.mp3"))
    except:
        pass
