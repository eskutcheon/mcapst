import os
from typing import Union, Dict, List
from pprint import pprint
import torch
from torchvision.io import VideoReader, write_video



VIDEO_EXTENSIONS = (".mp4", ".webm", ".avi", ".mov", ".mkv", ".flv", ".wmv", ".m4v")


class VideoProcessor:
    def __init__(self, video_src: Union[str, bytearray, torch.ByteTensor], target_fps: int = 10, backend: str = "torchvision"):
        """ initialize a reusable video processor utility class object
            Args:
                video_src (str): Path to the video file, an in-memory file obj supported by FFMPEG, or a ByteTensor
        """
        self.video_src = video_src
        self.target_fps = target_fps
        self.backend = backend
        self.reader = self._initialize_video_reader(video_src)
        self.metadata = self._get_metadata()

    def _initialize_video_reader(self, video_src: Union[str, bytearray, torch.ByteTensor]):
        """
            Returns (torchvision.io.VideoReader) - object from fine-grained video reading API that allows frame extraction
                https://pytorch.org/vision/main/generated/torchvision.io.VideoReader.html
        """
        # placeholder implementation with dependency injection - will replace this internal logic with torchcodec later
        if self.backend == "torchvision":
            return VideoReader(video_src)
        elif self.backend == "torchcodec":
            raise NotImplementedError("torchcodec backend not yet implemented")
        else:
            raise ValueError(f"Unsupported backend: {self.backend}")

    def _get_metadata(self):
        # normalize metadata to provide consistent format (preparing for torchcodec conversion later)
        if self.backend == "torchvision":
            metadata = self.reader.get_metadata()
            print(metadata)
            return {
                "duration": metadata["video"]["duration"][0],
                "fps": metadata["video"]["fps"][0],
                #"resolution": (metadata["video"]["height"][0], metadata["video"]["width"][0]),
            }
        elif self.backend == "torchcodec":
            # Placeholder for torchcodec metadata extraction
            raise NotImplementedError("torchcodec backend not yet implemented")

    def _read_frame(self, timestamp: float):
        # again atomizing for the switch to torchcodec
        timestamp = round(timestamp, 3)
        if self.backend == "torchvision":
            self.reader.seek(timestamp)
            return next(self.reader)["data"]
        elif self.backend == "torchcodec":
            raise NotImplementedError("torchcodec backend not yet implemented")

    def frame_generator(self, start_pt: float = 0, end_pt: float = None, batch_size: int = 8):
        """ Generator function to yield video frames in minibatches """
        if end_pt is None:
            end_pt = self.metadata["duration"]
        total_frames = self.get_frame_count(start_pt, end_pt)
        timestamps = torch.linspace(start_pt, end_pt, steps=total_frames, dtype=torch.float16)
        for i in range(0, total_frames, batch_size):
            frames = []
            for tstamp in timestamps[i : i + batch_size]:
                frames.append(self._read_frame(tstamp.item()))
            yield torch.stack(frames, dim=0)

    ###################################### simpler helper functions ######################################

    def save_video_to_disk(self, frames: torch.Tensor, fps: int, output_path: os.PathLike = None):
        if output_path is None:
            # default output path
            output_dir = os.path.abspath(os.path.join(__file__, "..", "..", "..", "results"))
            os.makedirs(output_dir, exist_ok=True)
            num_files = len([p for p in os.listdir(output_dir) if p.endswith(VIDEO_EXTENSIONS)])
            output_path = os.path.join(output_dir, f"video_stylized{num_files + 1}.mp4")
        if self.backend == "torchvision":
            # NOTE: write_video requires (T, H, W, C) format
            frames = frames.permute(0, 2, 3, 1).mul(255).byte()
            write_video(output_path, frames, fps=int(fps), options={"crf": "18"})
        elif self.backend == "torchcodec":
            raise NotImplementedError("torchcodec backend not yet implemented")

    def print_metadata(self):
        pprint(self.metadata, indent=2)

    def get_frame_count(self, start_pt: float = 0, end_pt: float = None) -> int:
        return int((end_pt - start_pt) * self.target_fps + 0.9999)

    def __del__(self):
        del self.reader