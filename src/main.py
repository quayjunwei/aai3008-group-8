import os
import streamlit as st
import csv
from asr.asr_processing import extract_audio, transcribe_audio
from transcript.transcript_processing import embed_subtitles
from llm.translation import process_srt

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
st.set_page_config(page_title="Educational Video Translation System", layout="wide")
st.title("Educational Video Translation System")

# Initialize session state with default values
if 'processed' not in st.session_state:
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
        uploaded_file = st.file_uploader("Choose a video file", 
                                       type=["mp4", "mov", "avi", "mkv"],
                                       key="file_uploader")
        
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
                ["auto-detect", "English", "Spanish", "French", "German", "Chinese", "Japanese"]
            )
            
            # Process button
            if st.button("Generate Video with Subtitles"):
                with st.spinner("Processing video..."):
                    # Processing pipeline
                    audio_path = os.path.join(output_dir, "extracted_audio.mp3")
                    srt_path = os.path.join(output_dir, "subtitles.srt")
                    csv_path = os.path.join(output_dir, "subtitles.csv")
                    output_video_path = os.path.join(output_dir, "video_with_subtitles.mp4")
                    
                    success, error = extract_audio(input_video_path, audio_path)
                    if not success:
                        st.error(f"Audio extraction failed: {error}")
                        st.stop()
                    
                    success, error, detected_lang = transcribe_audio(audio_path, srt_path, csv_path)
                    if not success:
                        st.error(f"Transcription failed: {error}")
                        st.stop()
                    
                    success, error = embed_subtitles(input_video_path, srt_path, output_video_path)
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
        if st.session_state.output_video_path and os.path.exists(st.session_state.output_video_path):
            st.video(st.session_state.output_video_path)
        else:
            st.warning("Video not processed yet.")
        
        # Language display
        lang_source = "Detected" if st.session_state.selected_lang == "auto-detect" else "Selected"
        display_lang = st.session_state.detected_lang if st.session_state.selected_lang == "auto-detect" else st.session_state.selected_lang
        st.markdown(f"**{lang_source} Language:** {display_lang or 'Unknown'}")
        
        # Download buttons
        if st.session_state.output_video_path and os.path.exists(st.session_state.output_video_path):
            with open(st.session_state.output_video_path, "rb") as f:
                st.download_button("Download Subtitled Video", f, "video_with_subtitles.mp4")
        if st.session_state.srt_path and os.path.exists(st.session_state.srt_path):
            with open(st.session_state.srt_path, "rb") as f:
                st.download_button("Download SRT Subtitles", f, "subtitles.srt")
        if st.session_state.csv_path and os.path.exists(st.session_state.csv_path):
            with open(st.session_state.csv_path, "rb") as f:
                st.download_button("Download Transcript CSV", f, "subtitles.csv")
        
        # Full transcript
        st.subheader("Full Transcript")
        if st.session_state.csv_path and os.path.exists(st.session_state.csv_path):
            try:
                with open(st.session_state.csv_path, "r", encoding="utf-8") as f:
                    reader = csv.DictReader(f)
                    for row in reader:
                        st.write(f"**{float(row['start']):.1f}s - {float(row['end']):.1f}s**")
                        st.write(row['text'])
                        st.divider()
            except Exception as e:
                st.error(f"Error loading transcript: {str(e)}")
        else:
            st.warning("Transcript not available. Please process the video first.")

with col2:
    if st.session_state.processed:
        # Translation section
        st.subheader("Translate Subtitles")
        target_lang = st.selectbox("Translate subtitles to:", ["en", "es", "fr"], index=0, format_func=lambda x: {"en": "English", "es": "Spanish", "fr": "French"}[x])
        if st.button("Translate and Generate Video"):
            with st.spinner("Translating and embedding subtitles..."):
                if st.session_state.srt_path and os.path.exists(st.session_state.srt_path) and st.session_state.detected_lang and st.session_state.input_video_path:
                    translated_srt = os.path.join(output_dir, f"subtitles_{target_lang}.srt")
                    source_lang = st.session_state.detected_lang
                    print(f"Translating from {source_lang} to {target_lang}")
                    process_srt(st.session_state.srt_path, translated_srt, source_lang, target_lang)
                    
                    translated_video = os.path.join(output_dir, f"video_{target_lang}.mp4")
                    success, error = embed_subtitles(st.session_state.input_video_path, translated_srt, translated_video)
                    if success:
                        st.session_state.translated_video_path = translated_video
                        st.rerun()  # Refresh the UI to update the video display
                    else:
                        st.error(f"Translation embedding failed: {error}")
                else:
                    st.error("Subtitles, detected language, or input video path not available. Please process the video first.")

        # Display translated video if available
        if st.session_state.translated_video_path and os.path.exists(st.session_state.translated_video_path):
            st.subheader("Translated Video")
            video_placeholder = st.empty()  # Use a placeholder to manage the video display
            video_placeholder.video(st.session_state.translated_video_path)
            with open(st.session_state.translated_video_path, "rb") as f:
                st.download_button(f"Download Video", f, f"video_{target_lang}.mp4")
        else:
            st.info("No translated video available yet.")

# Cleanup
if os.path.exists(os.path.join(output_dir, "extracted_audio.mp3")):
    try:
        os.remove(os.path.join(output_dir, "extracted_audio.mp3"))
    except:
        pass