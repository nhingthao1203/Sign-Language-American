import os
import cv2
import numpy as np
from tqdm import tqdm
import random
import torch
from torch.utils.data import Dataset, DataLoader
import subprocess
import tempfile
import shutil


class ASLVideoDataset:
    def __init__(self, root_dir, transform=None, num_frames=16, frame_selection='uniform', convert_problematic=True):
        """
        Process ASL videos organized in a directory structure where each category is a folder.

        Args:
            root_dir: Root directory containing category folders
            transform: Optional transforms to apply to frames
            num_frames: Number of frames to extract from each video
            frame_selection: Method to select frames ('uniform', 'random')
            convert_problematic: Whether to convert problematic videos to MJPEG format
        """
        self.root_dir = root_dir
        self.transform = transform
        self.num_frames = num_frames
        self.frame_selection = frame_selection
        self.convert_problematic = convert_problematic
        self.problematic_videos = []
        self.temp_dir = tempfile.mkdtemp() if convert_problematic else None

        # Get all categories (folders)
        self.categories = [d for d in os.listdir(root_dir)
                           if os.path.isdir(os.path.join(root_dir, d))]
        self.categories.sort()

        # Create category-to-index mapping
        self.category_to_idx = {cat: idx for idx, cat in enumerate(self.categories)}

        # Get all video paths and their labels
        self.samples = []
        for category in self.categories:
            category_dir = os.path.join(root_dir, category)
            for filename in os.listdir(category_dir):
                if filename.endswith(('.mp4', '.avi', '.mov')):
                    video_path = os.path.join(category_dir, filename)
                    self.samples.append((video_path, self.category_to_idx[category]))

    def __del__(self):
        # Clean up temp directory when object is destroyed
        if self.temp_dir and os.path.exists(self.temp_dir):
            shutil.rmtree(self.temp_dir)

    def convert_video_format(self, input_path):
        """Convert problematic video to MJPEG format"""
        if not os.path.exists(self.temp_dir):
            os.makedirs(self.temp_dir, exist_ok=True)

        output_filename = os.path.basename(input_path)
        output_path = os.path.join(self.temp_dir, f"converted_{output_filename}")

        try:
            # Check if ffmpeg is available
            subprocess.run(['ffmpeg', '-version'], stdout=subprocess.PIPE, stderr=subprocess.PIPE, check=True)

            # Convert video using ffmpeg
            command = [
                'ffmpeg', '-i', input_path,
                '-c:v', 'mjpeg', '-q:v', '3',
                '-an', output_path, '-y'
            ]

            # Run the command with a timeout
            subprocess.run(command, stdout=subprocess.PIPE, stderr=subprocess.PIPE, timeout=30)

            if os.path.exists(output_path) and os.path.getsize(output_path) > 0:
                print(f"Successfully converted {input_path} to MJPEG format")
                return output_path
            else:
                print(f"Failed to convert {input_path}: Output file is empty or missing")
                return input_path

        except (subprocess.SubprocessError, FileNotFoundError) as e:
            print(f"Error converting video {input_path}: {str(e)}")
            return input_path

    def extract_frames_ffmpeg(self, video_path):
        """Extract frames using FFmpeg instead of OpenCV"""
        frames = []
        temp_frame_dir = os.path.join(self.temp_dir, f"frames_{os.path.basename(video_path)}")
        os.makedirs(temp_frame_dir, exist_ok=True)

        try:
            # Get frame count
            cmd = [
                'ffprobe', '-v', 'error', '-select_streams', 'v:0',
                '-count_packets', '-show_entries', 'stream=nb_read_packets',
                '-of', 'csv=p=0', video_path
            ]
            frame_count = int(subprocess.check_output(cmd).decode('utf-8').strip())

            # Calculate frame indices
            if frame_count <= self.num_frames:
                indices = range(frame_count)
            else:
                step = frame_count // self.num_frames
                indices = range(0, frame_count, step)[:self.num_frames]

            # Extract frames
            for i, idx in enumerate(indices):
                output_file = os.path.join(temp_frame_dir, f"frame_{i:03d}.jpg")
                cmd = [
                    'ffmpeg', '-i', video_path, '-vf', f'select=eq(n\\,{idx})',
                    '-vframes', '1', output_file, '-y'
                ]
                subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE)

                if os.path.exists(output_file):
                    frame = cv2.imread(output_file)
                    if frame is not None:
                        frame = cv2.resize(frame, (224, 224))
                        frames.append(frame)
                    else:
                        frames.append(np.zeros((224, 224, 3), dtype=np.uint8))
                else:
                    frames.append(np.zeros((224, 224, 3), dtype=np.uint8))

            # Fill with blank frames if we don't have enough
            while len(frames) < self.num_frames:
                frames.append(np.zeros((224, 224, 3), dtype=np.uint8))

            # Clean up temp frames
            shutil.rmtree(temp_frame_dir)

            return np.array(frames)

        except Exception as e:
            print(f"Error extracting frames with FFmpeg for {video_path}: {str(e)}")
            # Return blank frames on error
            return np.zeros((self.num_frames, 224, 224, 3), dtype=np.uint8)

    def extract_frames(self, video_path):
        """Extract frames from a video file with error handling"""
        try:
            frames = []
            cap = cv2.VideoCapture(video_path)

            if not cap.isOpened():
                print(f"Error opening video file: {video_path}")
                if self.convert_problematic:
                    print(f"Attempting to convert {video_path}")
                    converted_path = self.convert_video_format(video_path)
                    if converted_path != video_path:
                        # Try again with converted video
                        cap = cv2.VideoCapture(converted_path)
                        if not cap.isOpened():
                            print(f"Still can't open converted video: {converted_path}")
                            self.problematic_videos.append(video_path)
                            # Try FFmpeg as last resort
                            return self.extract_frames_ffmpeg(video_path)
                    else:
                        self.problematic_videos.append(video_path)
                        # Try FFmpeg as last resort
                        return self.extract_frames_ffmpeg(video_path)
                else:
                    self.problematic_videos.append(video_path)
                    return np.zeros((self.num_frames, 224, 224, 3), dtype=np.uint8)

            frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            error_frame_count = 0

            if self.frame_selection == 'uniform':
                # Select frames at uniform intervals
                if frame_count <= self.num_frames:
                    indices = np.linspace(0, frame_count - 1, self.num_frames, dtype=int)
                else:
                    indices = np.linspace(0, frame_count - 1, self.num_frames, dtype=int)

                for idx in indices:
                    cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
                    ret, frame = cap.read()
                    if ret:
                        # Resize frame to standard size
                        frame = cv2.resize(frame, (224, 224))
                        frames.append(frame)
                    else:
                        error_frame_count += 1
                        # If reading fails, append a blank frame
                        frames.append(np.zeros((224, 224, 3), dtype=np.uint8))

            elif self.frame_selection == 'random':
                # Select random frames
                if frame_count <= self.num_frames:
                    indices = list(range(frame_count))
                    while len(indices) < self.num_frames:
                        indices.append(random.randint(0, frame_count - 1))
                else:
                    indices = sorted(random.sample(range(frame_count), self.num_frames))

                for idx in indices:
                    cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
                    ret, frame = cap.read()
                    if ret:
                        frame = cv2.resize(frame, (224, 224))
                        frames.append(frame)
                    else:
                        error_frame_count += 1
                        frames.append(np.zeros((224, 224, 3), dtype=np.uint8))

            cap.release()

            # If too many frames had errors, try converting the video
            if error_frame_count > self.num_frames // 3 and self.convert_problematic:
                print(f"Too many frame errors in {video_path}: {error_frame_count}/{self.num_frames}")
                self.problematic_videos.append(video_path)
                converted_path = self.convert_video_format(video_path)
                if converted_path != video_path:
                    # Try again with FFmpeg extraction
                    return self.extract_frames_ffmpeg(converted_path)
                else:
                    return self.extract_frames_ffmpeg(video_path)

            # Fill in missing frames if needed
            while len(frames) < self.num_frames:
                frames.append(np.zeros((224, 224, 3), dtype=np.uint8))

            return np.array(frames)

        except Exception as e:
            print(f"Error processing {video_path}: {str(e)}")
            self.problematic_videos.append(video_path)

            if self.convert_problematic:
                try:
                    # Try using FFmpeg as a last resort
                    return self.extract_frames_ffmpeg(video_path)
                except:
                    return np.zeros((self.num_frames, 224, 224, 3), dtype=np.uint8)
            else:
                return np.zeros((self.num_frames, 224, 224, 3), dtype=np.uint8)

    def process_all_videos(self, output_dir=None, save_frames=False):
        """
        Process all videos in the dataset

        Args:
            output_dir: Directory to save processed frames (if save_frames is True)
            save_frames: Whether to save extracted frames to disk
        """
        if save_frames and output_dir is not None:
            os.makedirs(output_dir, exist_ok=True)

        all_features = []
        all_labels = []

        for video_path, label in tqdm(self.samples, desc="Processing videos"):
            # Extract frames
            frames = self.extract_frames(video_path)

            # Apply transforms if any
            if self.transform:
                transformed_frames = []
                for frame in frames:
                    transformed_frames.append(self.transform(frame))
                frames = np.array(transformed_frames)

            # Save frames if requested
            if save_frames and output_dir is not None:
                video_name = os.path.splitext(os.path.basename(video_path))[0]
                category_name = os.path.basename(os.path.dirname(video_path))
                video_output_dir = os.path.join(output_dir, category_name, video_name)
                os.makedirs(video_output_dir, exist_ok=True)

                for i, frame in enumerate(frames):
                    frame_path = os.path.join(video_output_dir, f"frame_{i:03d}.jpg")
                    cv2.imwrite(frame_path, frame)

            # Add to dataset
            all_features.append(frames)
            all_labels.append(label)

        # Save list of problematic videos
        if output_dir is not None and self.problematic_videos:
            with open(os.path.join(output_dir, 'problematic_videos.txt'), 'w') as f:
                for video_path in self.problematic_videos:
                    f.write(f"{video_path}\n")

            print(f"Found {len(self.problematic_videos)} problematic videos out of {len(self.samples)}")

        return np.array(all_features), np.array(all_labels)

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        video_path, label = self.samples[idx]
        frames = self.extract_frames(video_path)

        # Apply transforms if any
        if self.transform:
            transformed_frames = []
            for frame in frames:
                transformed_frames.append(self.transform(frame))
            frames = np.array(transformed_frames)

        # Convert to pytorch tensor and change to format [C, T, H, W]
        frames = torch.from_numpy(frames).float()
        frames = frames.permute(3, 0, 1, 2)  # [T, H, W, C] -> [C, T, H, W]

        return frames, label


def preprocess_videos(dataset_path, output_path=None, save_frames=False, convert_problematic=True):
    """
    Preprocess videos in the dataset

    Args:
        dataset_path: Path to the dataset root directory
        output_path: Path to save processed data (optional)
        save_frames: Whether to save extracted frames
        convert_problematic: Whether to convert problematic videos
    """
    # Create dataset
    dataset = ASLVideoDataset(
        root_dir=dataset_path,
        num_frames=16,
        frame_selection='uniform',
        convert_problematic=convert_problematic
    )

    print(f"Found {len(dataset.categories)} categories:")
    for i, category in enumerate(dataset.categories):
        print(f"  {i}: {category}")

    print(f"Total number of video samples: {len(dataset.samples)}")

    # Process all videos
    features, labels = dataset.process_all_videos(output_path, save_frames)

    print(f"Processed features shape: {features.shape}")
    print(f"Processed labels shape: {labels.shape}")

    # Save processed data if output path is provided
    if output_path is not None:
        os.makedirs(output_path, exist_ok=True)
        np.save(os.path.join(output_path, 'features.npy'), features)
        np.save(os.path.join(output_path, 'labels.npy'), labels)

        # Save category mapping
        with open(os.path.join(output_path, 'categories.txt'), 'w') as f:
            for category in dataset.categories:
                f.write(f"{category}\n")

    return features, labels, dataset.categories


def check_ffmpeg_installed():
    """Check if FFmpeg is installed on the system"""
    try:
        subprocess.run(['ffmpeg', '-version'], stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        return True
    except (subprocess.SubprocessError, FileNotFoundError):
        return False


if __name__ == "__main__":
    # Replace with your dataset path
    dataset_path = r"E:\ASL_Citizen\pythonProject\val"
    output_path = r"E:\ASL_Citizen\pythonProject\Process\val_processed"

    # Check if FFmpeg is installed
    if not check_ffmpeg_installed():
        print("Warning: FFmpeg is not installed or not found in PATH.")
        print("For better video handling, please install FFmpeg:")
        print("  - Windows: Download from https://ffmpeg.org/download.html")
        print("  - Linux: sudo apt install ffmpeg")
        print("  - macOS: brew install ffmpeg")
        convert_problematic = False
    else:
        convert_problematic = True

    features, labels, categories = preprocess_videos(
        dataset_path=dataset_path,
        output_path=output_path,
        save_frames=True,  # Set to True if you want to save individual frames
        convert_problematic=convert_problematic
    )

    print("Video processing complete!")