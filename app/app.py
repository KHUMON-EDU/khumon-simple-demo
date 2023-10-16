import os
import re
import tempfile

import openai
import streamlit as st
from dotenv import load_dotenv
from langchain.chat_models import ChatOpenAI
from langchain.document_loaders import PyPDFLoader
from moviepy.editor import VideoFileClip, vfx
from PIL import Image
from langchain.prompts import PromptTemplate

load_dotenv(verbose=True)

url_regex = (
    "(https:\/\/www\.|http:\/\/www\.|https:\/\/|http:\/\/)?[a-zA-Z0-9]{2,}(\.[a-zA-Z0-9]{2,})(\.[a-zA-Z0-9]{2,})?"
)

summarization_prompt_template = PromptTemplate.from_template(
    """
You are responsible for summarizing the lecture materials of your major in a way that is easy for students to understand. Please organize and summarize the given sentences, and ignore external links and unnecessary words. script: {docs}
"""
)

generation_prompt_template = PromptTemplate.from_template(
    """
You are responsible for creating 10 questions to help students understand the summarized lecture materials of your major.
Avoid non-essential questions.
Questions should be short answer or essay format.
Technical terms and proper nouns related to major should be used in English. Generate answers to questions concisely, focusing on keywords.
The output format is as follows.

Output format:

Q1: Content of question 1
A1: Content of answer 1

Q2: Content of question 2
A2: Content of answer 2

Q3 Content of question 2
A3: Content of answer 2

script: {summary}"""
)

translation_prompt_template = PromptTemplate.from_template(
    """
Translate the given sentences into Korean in a natural way, following the format. Keep technical terms and proper nouns in English.
{script}
"""
)


def process_pdf(source):
    with tempfile.NamedTemporaryFile(delete=False) as tmp_file:
        tmp_file.write(source.read())
    loader = PyPDFLoader(tmp_file.name, extract_images=is_ocr)
    pages = loader.load_and_split()
    os.remove(tmp_file.name)
    docs = get_docs(pages)
    return docs


def process_mp4(source):
    with tempfile.TemporaryDirectory() as td:
        tmp_video_path = os.path.join(td, "temp_video.mp4")
        tmp_audio_path = os.path.join(td, "temp_audio.mp3")

        with open(tmp_video_path, "wb") as tmp_video:
            tmp_video.write(source.read())
        video_clip = VideoFileClip(tmp_video_path)
        video_clip = video_clip.fx(vfx.speedx, 1.5)
        video_clip.audio.write_audiofile(tmp_audio_path)
        with open(tmp_audio_path, "rb") as tmp_audio:
            transcript = openai.Audio.transcribe("whisper-1", tmp_audio, api_key=os.getenv("OPENAI_API_KEY"))["text"]

    return transcript


def get_docs(pages):
    docs = ""
    for page in pages:
        content = page.page_content
        replace_content = re.sub(url_regex, "", content)
        docs += replace_content
    return docs


logo_image = Image.open("static/logo.jpg")
st.image(logo_image, width=100)
st.title("KHUMON DEMO")

with st.sidebar:
    mode = st.radio("Input Format", ["PDF", "VIDEO"])

if mode == "PDF":
    is_ocr = st.toggle("Extract images from pdf")
    source = st.file_uploader("강의 PDF를 업로드해주세요!", type="pdf", label_visibility="collapsed")

elif mode == "VIDEO":
    source = st.file_uploader("강의 비디오(MP4)를 업로드해주세요!", type="mp4", label_visibility="collapsed")


if st.button("Make! ✈️"):
    try:
        with st.spinner("분석 중 🏃"):
            if mode == "PDF":
                docs = process_pdf(source)
            elif mode == "VIDEO":
                docs = process_mp4(source)

            llm = ChatOpenAI(model_name="gpt-3.5-turbo-16k", temperature=0)
            summarization_prompt = summarization_prompt_template.format(docs=docs[:10000])
            summary = llm.predict(summarization_prompt)
            generation_prompt = generation_prompt_template.format(summary=summary)
            question = llm.predict(generation_prompt)
            translation_prompt = translation_prompt_template.format(script=question)
            result = llm.predict(translation_prompt)
        st.subheader("✒️ 요약")
        st.text(summary)
        st.subheader("❓ 질문")
        st.text(result)
    except Exception as e:
        st.error(f"An error occurred: {e}")
