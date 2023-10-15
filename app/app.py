import os
import tempfile
import re

import streamlit as st
from dotenv import load_dotenv
from langchain.document_loaders import PyPDFLoader
from langchain.chat_models import ChatOpenAI
from PIL import Image
from moviepy.editor import VideoFileClip, vfx
import openai

load_dotenv(verbose=True)

url_regex = (
    "(https:\/\/www\.|http:\/\/www\.|https:\/\/|http:\/\/)?[a-zA-Z0-9]{2,}(\.[a-zA-Z0-9]{2,})(\.[a-zA-Z0-9]{2,})?"
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
        tmp_video_path  = os.path.join(td,'temp_video.mp4')
        tmp_audio_path =os.path.join(td,'temp_audio.mp3')

        with open(tmp_video_path, 'wb') as tmp_video:
            tmp_video.write(source.read())
        video_clip = VideoFileClip(tmp_video_path)
        video_clip = video_clip.fx(vfx.speedx, 1.5)
        video_clip.audio.write_audiofile(tmp_audio_path)
        with open(tmp_audio_path, 'rb') as tmp_audio:
            transcript = openai.Audio.transcribe("whisper-1", tmp_audio, api_key=os.getenv("OPENAI_API_KEY"))['text']

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
    mode = st.radio("Input Format",["PDF", "VIDEO"])

if mode == 'PDF':
    is_ocr = st.toggle('Extract images from pdf')
    source = st.file_uploader("강의 PDF를 업로드해주세요!", type="pdf", label_visibility="collapsed")

elif mode =="VIDEO":
    source = st.file_uploader("강의 비디오(MP4)를 업로드해주세요!", type="mp4", label_visibility="collapsed")



if st.button("Make! ✈️"):
    try:
        with st.spinner("분석 중 🏃"):
            if mode == 'PDF':
                docs = process_pdf(source)
            elif mode == 'VIDEO':
                docs = process_mp4(source)

            llm = ChatOpenAI(temperature=0)
            summary = llm.predict(
                f"당신은 전공 강의 자료 요약기 입니다. 아래 주어진 대본을 키워드 위주로 적절하게 요약하세요. 링크나 의미 없는 단어, 문장 들은 무시해도 좋습니다. 대본: {docs[:10000]} "
            )
            question = llm.predict(
                f"당신은 전공 강의를 요약한 자료를 바탕으로 학생의 이해를 돕기 위한 문제를 10개를 만드세요. 문제는 단답형 또는 주관식으로 만드세요. 대본: {summary} "
            )
        st.subheader("✒️ 요약")
        st.text(summary)
        st.subheader("❓ 질문")
        st.text(question)
    except Exception as e:
        st.error(f"An error occurred: {e}")
