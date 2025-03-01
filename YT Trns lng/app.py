import streamlit as st
from langgraph.graph import Graph
from dataclasses import dataclass, field
from typing import TypedDict
import google.generativeai as genai
from youtube_transcript_api import YouTubeTranscriptApi
from urllib.parse import urlparse, parse_qs

genai.configure(api_key="AIzaSyCf2fxYoO2xTJVXAOL23_WXMjpBJfOY_b0")

@dataclass(kw_only=True)
class SummaryState(TypedDict):
    extract_transcript: str = field(default=None)
    generate_content: str = field(default=None)

@dataclass(kw_only=True)
class SummaryStateInput(TypedDict):
    extract_transcript: str = field(default=None)

@dataclass(kw_only=True)
class SummaryStateOutput(TypedDict):
    generate_content: str = field(default=None)

def extract_transcript_details(youtube_video_url):
    try:
        video_id = youtube_video_url.split("=")[1]
        transcript_text = YouTubeTranscriptApi.get_transcript(video_id)
        transcript = " ".join(i["text"] for i in transcript_text)
        return transcript
    except Exception as e:
        raise e



def generate_summary(state: SummaryStateInput):
    transcript_text = state["extract_transcript"]
    prompt = f"""You are a YouTube video summarizer. You will be taking the transcript text
and summarizing the entire video and providing the important summary in points
within 250 words. Please provide the summary of the text given here:

{transcript_text}"""
    model = genai.GenerativeModel("models/gemini-1.5-pro-latest")
    response = model.generate_content(prompt)
    return {"generate_content": response.text}

def summarize_video(transcript_text):
    workflow = Graph()
    workflow.add_node("summarizer", generate_summary)
    workflow.set_entry_point("summarizer")
    workflow.set_finish_point("summarizer")
    executor = workflow.compile()
    return executor.invoke({"extract_transcript": transcript_text})

st.title("YouTube Video Summarizer using LangGraph and Google Gemini")
youtube_link = st.text_input("Enter YouTube Video Link:")

if youtube_link:
    parsed_url = urlparse(youtube_link)
    query_params = parse_qs(parsed_url.query)
    video_ids = query_params.get("v")
    if video_ids:
        video_id = video_ids[0]
        st.image(f"http://img.youtube.com/vi/{video_id}/0.jpg", use_container_width=True)

if st.button("Get Detailed Notes"):
    transcript_text = extract_transcript_details(youtube_link)
    if transcript_text:
        summary = summarize_video(transcript_text)
        if summary and "generate_content" in summary:
            st.markdown("## Video Summary:")
            st.write(summary["generate_content"])
        else:
            st.error("Error: No summary was generated. Please check your input.")
