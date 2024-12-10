import os
import logging
import shutil
import tempfile
from typing import List, Optional

import cv2
import numpy as np
import requests
from PIL import Image

import uvicorn
from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.responses import FileResponse
from fastapi.middleware.cors import CORSMiddleware

# Logging configuration
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Environment and Configuration
class Settings:
    """Application configuration settings"""
    HUGGING_FACE_API_TOKEN: str = os.getenv('HUGGING_FACE_API_TOKEN', '')
    MAX_VIDEO_SIZE: int = 50_000_000  # 50 MB
    FRAME_EXTRACTION_RATE: int = 1  # frame per second
    THUMBNAIL_SIZE: tuple = (512, 512)

# Video Processing Utilities
class VideoProcessor:
    @staticmethod
    def extract_frames(
        video_path: str, 
        output_dir: str, 
        frame_rate: int = Settings.FRAME_EXTRACTION_RATE
    ) -> List[str]:
        """
        Extract frames from a video at specified frame rate
        
        :param video_path: Path to the input video file
        :param output_dir: Directory to save extracted frames
        :param frame_rate: Number of frames to extract per second
        :return: List of extracted frame paths
        """
        os.makedirs(output_dir, exist_ok=True)
        video = cv2.VideoCapture(video_path)
        
        if not video.isOpened():
            raise HTTPException(status_code=400, detail="Could not open video file")
        
        fps = int(video.get(cv2.CAP_PROP_FPS))
        frame_interval = max(1, fps // frame_rate)
        
        frames = []
        frame_id = 0
        while True:
            success, frame = video.read()
            if not success:
                break
            
            if frame_id % frame_interval == 0:
                frame_path = os.path.join(output_dir, f"frame_{frame_id}.jpg")
                cv2.imwrite(frame_path, frame)
                frames.append(frame_path)
            
            frame_id += 1
        
        video.release()
        return frames

    @staticmethod
    def create_thumbnail(
        text: str = "Your Video Thumbnail", 
        size: tuple = Settings.THUMBNAIL_SIZE
    ) -> str:
        """
        Create a basic thumbnail with text
        
        :param text: Text to display on thumbnail
        :param size: Thumbnail dimensions
        :return: Path to created thumbnail
        """
        thumbnail = np.zeros((*size, 3), dtype=np.uint8)
        
        # Add text to thumbnail
        cv2.putText(
            thumbnail, 
            text[:30],  # Limit text length
            (10, size[1] // 2), 
            cv2.FONT_HERSHEY_SIMPLEX, 
            1, 
            (255, 255, 255), 
            2, 
            cv2.LINE_AA
        )
        
        thumbnail_path = os.path.join(tempfile.gettempdir(), "thumbnail.jpg")
        cv2.imwrite(thumbnail_path, thumbnail)
        return thumbnail_path

# External API Interaction Utilities
class ExternalAPIClient:
    @staticmethod
    def generate_hooks(
        summary: str, 
        api_token: str = Settings.HUGGING_FACE_API_TOKEN
    ) -> List[str]:
        """
        Generate marketing hooks using Hugging Face API
        
        :param summary: Summary text to generate hooks from
        :param api_token: Hugging Face API token
        :return: List of generated marketing hooks
        """
        if not api_token:
            logger.warning("Hugging Face API token not provided")
            return ["Learn more about this video!"]
        
        API_URL = "https://api-inference.huggingface.co/models/gpt2"
        headers = {"Authorization": f"Bearer {api_token}"}
        
        prompt = f"Generate 3 engaging marketing hooks for a video summary: '{summary}'"
        payload = {
            "inputs": prompt, 
            "parameters": {
                "max_length": 50, 
                "num_return_sequences": 3
            }
        }
        
        try:
            response = requests.post(API_URL, headers=headers, json=payload, timeout=10)
            response.raise_for_status()
            
            hooks = [hook["generated_text"].strip() for hook in response.json()]
            return hooks
        except requests.exceptions.RequestException as e:
            logger.error(f"Hook generation error: {e}")
            return ["Discover something amazing!"]

# FastAPI Application Setup
app = FastAPI(
    title="Video Thumbnail Generator",
    description="Generate engaging thumbnails from uploaded videos",
    version="1.0.0"
)

# CORS Middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allows all origins
    allow_credentials=True,
    allow_methods=["*"],  # Allows all methods
    allow_headers=["*"],  # Allows all headers
)

@app.post("/generate-thumbnail/")
async def generate_thumbnail_api(
    video: UploadFile = File(..., description="Video file to generate thumbnail from")
):
    """
    Generate a thumbnail from an uploaded video
    
    - Extracts frames
    - Generates a marketing hook
    - Creates a thumbnail with the hook
    """
    # Validate file type and size
    if not video.content_type.startswith('video/'):
        raise HTTPException(status_code=400, detail="Invalid file type. Must be a video.")
    
    # Temporary file and directory management
    temp_video_path = None
    frames_dir = None
    
    try:
        # Save uploaded video
        temp_video_path = tempfile.NamedTemporaryFile(delete=False, suffix=".mp4").name
        with open(temp_video_path, "wb") as f:
            f.write(await video.read())
        
        # Create temporary directory for frames
        frames_dir = tempfile.mkdtemp()
        
        # Extract frames
        frames = VideoProcessor.extract_frames(temp_video_path, frames_dir)
        
        # Basic frame summary (placeholder for more advanced analysis)
        frame_summaries = [f"Frame {i}" for i in range(len(frames))]
        full_summary = " ".join(frame_summaries)
        
        # Generate hooks
        hooks = ExternalAPIClient.generate_hooks(full_summary)
        prompt = hooks[0] if hooks else "Thumbnail for Your Video"
        
        # Create thumbnail
        thumbnail_path = VideoProcessor.create_thumbnail(prompt)
        
        # Return thumbnail
        return FileResponse(
            thumbnail_path, 
            media_type="image/jpeg", 
            filename="thumbnail.jpg"
        )
    
    except Exception as e:
        logger.error(f"Thumbnail generation error: {e}")
        raise HTTPException(status_code=500, detail="Thumbnail generation failed")
    
    finally:
        # Cleanup temporary files
        if temp_video_path and os.path.exists(temp_video_path):
            os.remove(temp_video_path)
        
        if frames_dir and os.path.exists(frames_dir):
            shutil.rmtree(frames_dir)

# Health Check Endpoint
@app.get("/health")
async def health_check():
    """Basic health check endpoint"""
    return {"status": "healthy"}

# Main block for direct script execution
if __name__ == "__main__":
    uvicorn.run(
        "main:app", 
        host="0.0.0.0", 
        port=8000, 
        reload=True
    )