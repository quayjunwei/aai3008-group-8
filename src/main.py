import streamlit as st
import os
import shutil
import csv
from asr.asr_processing import ensure_dir, extract_audio, transcribe_audio, embed_subtitles

# Create directories
script_dir = os.path.dirname(os.path.abspath(__file__))
input_dir = os.path.join(script_dir, "input")
output_dir = os.path.join(script_dir, "output")
ensure_dir(input_dir)
ensure_dir(output_dir)

# Streamlit UI
st.set_page_config(page_title="Educational Video Translation System", layout="wide")
st.title("Educational Video Translation System")

# Initialize session state
if 'processed' not in st.session_state:
    st.session_state.processed = False
    st.session_state.uploaded_file = None
    st.session_state.selected_lang = "auto-detect"
    st.session_state.detected_lang = None

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
                    # Clear temporary files from memory
                    st.session_state.uploaded_file = None
                    st.rerun()

    if st.session_state.processed:
        # Results display
        st.subheader("Processed Video")
        st.video(st.session_state.output_video_path)
        
        # Language display
        lang_source = "Detected" if st.session_state.selected_lang == "auto-detect" else "Selected"
        display_lang = st.session_state.detected_lang if st.session_state.selected_lang == "auto-detect" else st.session_state.selected_lang
        st.markdown(f"**{lang_source} Language:** {display_lang}")
        
        # Download buttons
        with open(st.session_state.output_video_path, "rb") as f:
            st.download_button("Download Subtitled Video", f, "video_with_subtitles.mp4")
        
        with open(st.session_state.srt_path, "rb") as f:
            st.download_button("Download SRT Subtitles", f, "subtitles.srt")
        
        with open(st.session_state.csv_path, "rb") as f:
            st.download_button("Download Transcript CSV", f, "subtitles.csv")
        
        # Full transcript
        st.subheader("Full Transcript")
        try:
            with open(st.session_state.csv_path, "r", encoding="utf-8") as f:
                reader = csv.DictReader(f)
                for row in reader:
                    st.write(f"**{float(row['start']):.1f}s - {float(row['end']):.1f}s**")
                    st.write(row['text'])
                    st.divider()
        except Exception as e:
            st.error(f"Error loading transcript: {str(e)}")

with col2:
    st.subheader("Translated Video")
    # Placeholder for translation features
    st.info("Translation features coming soon!")

# Cleanup
if os.path.exists(os.path.join(output_dir, "extracted_audio.mp3")):
    try:
        os.remove(os.path.join(output_dir, "extracted_audio.mp3"))
    except:
        pass