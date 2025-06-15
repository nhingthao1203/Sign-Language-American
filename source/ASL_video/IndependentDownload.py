import os
import shutil
import json
import yt_dlp
from moviepy.editor import *
import moviepy.editor as mpy

# where to save
SAVE_PATH = "MS-ASL100_VAL"
temp_path = SAVE_PATH + "/untrimmed_videos"

if not os.path.exists(SAVE_PATH):
    os.makedirs(SAVE_PATH)
if not os.path.exists(temp_path):
    os.makedirs(temp_path)

try:
    # Các file JSON có thể dùng: MSASL_test.json, MSASL_TEST25.json, MSASL_VAL25.json, MSASL_TRAIN25.json
    with open('E:\MSASL-valid-dataset-downloader\MSASL_VAL25.json') as train_json:
        videos = json.load(train_json)
except Exception as e:
    print(f"Connection Error: {e}")
    exit()

# loop through the videos in the dataset
total = len(videos)
for i in range(total):
    try:
        url = videos[i]['url']
        start_time = videos[i]['start_time']
        end_time = videos[i]['end_time']
        label = videos[i]['label']
        pretitle = videos[i]['clean_text']
        video_title = pretitle + str(i)
        output_title = video_title + ".mp4"
        temp_file_path = f"{temp_path}/{video_title}.mp4"

        folder_path = f"{SAVE_PATH}/{pretitle}"
        output_path = f"{folder_path}/{output_title}"

        if not os.path.exists(folder_path):
            os.makedirs(folder_path)

        print(f"\n===== [{i}/{total}] {video_title} =====")
        print("Setup Completed!")

        # --- Dùng yt-dlp để tải video ---
        ydl_opts = {
            'outtmpl': temp_file_path,
            'quiet': True,
            'format': 'bestvideo+bestaudio/best',
            'merge_output_format': 'mp4'
        }

        with yt_dlp.YoutubeDL(ydl_opts) as ydl:
            ydl.download([url])

        print("Download Completed!")

        # --- Dùng moviepy để cắt video ---
        clip = VideoFileClip(temp_file_path).subclip(start_time, end_time)
        clip.write_videofile(output_path)
        print("Task Completed!")

    except Exception as e:
        print(f"Some Error at index {i} ({video_title}): {e}")
        #with open("error_log.txt", "a", encoding="utf-8") as log_file:
            #log_file.write(f"{i}: {video_title}, URL: {url}, Error: {e}\n")
