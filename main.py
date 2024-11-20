from fastapi import FastAPI, HTTPException, Response
from pydantic import BaseModel
import requests
import re
from io import BytesIO

# YouTube API Configuration
YOUTUBE_API_KEY = "AIzaSyAAea1lUcPM7BYQmJPC-jpkUzmXocbvBIM"  # Replace with your YouTube API Key
YOUTUBE_API_URL = "https://www.googleapis.com/youtube/v3/videos"

# Hugging Face API details
HUGGING_FACE_API_URL_SUMMARY = "https://api-inference.huggingface.co/models/facebook/bart-large-cnn"
HUGGING_FACE_API_URL_IMAGE = "https://api-inference.huggingface.co/models/stabilityai/stable-diffusion-2"
HUGGING_FACE_API_KEY = "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"  # Replace with your Hugging Face API Key

# FastAPI instance
app = FastAPI()

# Pydantic model for input
class YouTubeURL(BaseModel):
    url: str

# Helper function to fetch video metadata using YouTube Data API
def fetch_youtube_metadata(youtube_url):
    try:
        # Extract video ID from the URL
        video_id = None
        
        # Check for standard YouTube URL (https://www.youtube.com/watch?v=<video_id>)
        match = re.match(r"https?://(?:www\.)?youtube\.com/watch\?v=([a-zA-Z0-9_-]+)", youtube_url)
        if match:
            video_id = match.group(1)
        
        # Check for shortened YouTube URL (https://youtu.be/<video_id>)
        if not video_id:
            match = re.match(r"https?://(?:www\.)?youtu\.be/([a-zA-Z0-9_-]+)", youtube_url)
            if match:
                video_id = match.group(1)
        
        # If video_id was not found, raise an exception
        if not video_id:
            raise HTTPException(status_code=400, detail="Invalid YouTube URL format. Could not extract video ID.")
        
        # Make request to YouTube Data API
        response = requests.get(
            YOUTUBE_API_URL,
            params={
                "part": "snippet",
                "id": video_id,
                "key": YOUTUBE_API_KEY
            }
        )

        # Debugging: log the response status and body
        if response.status_code != 200:
            raise HTTPException(status_code=500, detail=f"Error fetching metadata. Response: {response.status_code} - {response.text}")
        
        data = response.json()

        if "items" in data and data["items"]:
            video_info = data["items"][0]["snippet"]
            return {
                "title": video_info.get("title", "No title available"),
                "description": video_info.get("description", "No description available")
            }
        else:
            raise HTTPException(status_code=404, detail="Video not found.")
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Error processing YouTube URL: {str(e)}")

# Helper function to summarize text using Hugging Face API
def summarize_text(text):
    headers = {"Authorization": f"Bearer {HUGGING_FACE_API_KEY}"}
    payload = {"inputs": text}
    response = requests.post(HUGGING_FACE_API_URL_SUMMARY, headers=headers, json=payload)
    if response.status_code == 200:
        return response.json()[0]["summary_text"]
    else:
        raise HTTPException(status_code=500, detail="Error summarizing the text using Hugging Face API.")

# Helper function to generate image using Hugging Face API
def generate_image(prompt):
    headers = {"Authorization": f"Bearer {HUGGING_FACE_API_KEY}"}
    payload = {"inputs": prompt}
    response = requests.post(HUGGING_FACE_API_URL_IMAGE, headers=headers, json=payload)
    
    if response.status_code == 200:
        image_data = response.content
        return image_data
    else:
        print("Error response from Hugging Face API:", response.json())
        raise HTTPException(status_code=500, detail="Error generating the image using Hugging Face API.")

# POST endpoint to process YouTube URL
@app.post("/process_youtube_url")
def process_youtube_url(data: YouTubeURL):
    # Step 1: Fetch metadata
    metadata = fetch_youtube_metadata(data.url)

    # Step 2: Summarize description
    summarized_content = summarize_text(metadata["description"])
    
    print(summarized_content)
    # Step 3: Generate image
    image_bytes = generate_image(summarized_content)

    # Return the image as a PNG file
    return Response(content=image_bytes, media_type="image/png")
