import os
import subprocess

def has_audio(input_path):
    """Check if a video has an audio stream using ffprobe."""
    command = [
        "ffprobe", "-i", input_path,
        "-show_streams", "-select_streams", "a",
        "-loglevel", "error"
    ]
    result = subprocess.run(command, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    return bool(result.stdout.strip())

def video_to_wav_ffmpeg(input_path, output_path, sample_rate=16000):
    """Convert a video file to wav format using ffmpeg."""
    if not has_audio(input_path):
        print(f"Skipping {input_path} (no audio track found)")
        return
    
    command = [
        "ffmpeg", "-i", input_path,
        "-vn",                     # no video
        "-acodec", "pcm_s16le",    # WAV format
        "-ar", str(sample_rate),   # sample rate
        "-ac", "1",                # mono channel
        output_path, "-y"          # overwrite existing output
    ]
    try:
        subprocess.run(command, check=True)
        print(f"Converted {input_path} -> {output_path}")
    except subprocess.CalledProcessError as e:
        print(f"Error converting {input_path}: {e}")

def convert_videos_in_folders(root_dir, output_root="Audio", sample_rate=16000):
    """Convert all videos in emotion folders to wav files."""
    os.makedirs(output_root, exist_ok=True)

    for emotion_folder in os.listdir(root_dir):
        emotion_path = os.path.join(root_dir, emotion_folder)
        
        if os.path.isdir(emotion_path):
            output_emotion_path = os.path.join(output_root, emotion_folder)
            os.makedirs(output_emotion_path, exist_ok=True)

            for file in os.listdir(emotion_path):
                if file.lower().endswith((".mp4", ".avi", ".mov", ".mkv")):
                    input_file = os.path.join(emotion_path, file)
                    output_file = os.path.join(
                        output_emotion_path, os.path.splitext(file)[0] + ".wav"
                    )
                    print(f"Processing {input_file}")
                    video_to_wav_ffmpeg(input_file, output_file, sample_rate)

if __name__ == "__main__":
    root_directory = "./data"
    convert_videos_in_folders(root_directory, output_root="Audio", sample_rate=16000)
