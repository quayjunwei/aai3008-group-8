import os
import subprocess
import shutil

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