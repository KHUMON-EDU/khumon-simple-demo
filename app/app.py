import os
import tempfile
import re

import streamlit as st
from dotenv import load_dotenv
from langchain.document_loaders import PyPDFLoader
from langchain.chat_models import ChatOpenAI
from PIL import Image
import whisper

load_dotenv()

url_regex = (
    "(https:\/\/www\.|http:\/\/www\.|https:\/\/|http:\/\/)?[a-zA-Z0-9]{2,}(\.[a-zA-Z0-9]{2,})(\.[a-zA-Z0-9]{2,})?"
)
with st.spinner("모델 준비 중 🏃"):
    model = whisper.load_model("base")

def process_pdf(source):
    try:
        with tempfile.NamedTemporaryFile(delete=False) as tmp_file:
            tmp_file.write(source.read())
        loader = PyPDFLoader(tmp_file.name, extract_images=is_ocr)
        pages = loader.load_and_split()
        os.remove(tmp_file.name)
        docs = get_docs(pages)
        return docs

    except:
        pass

def process_mp4(source):
    try:
        with tempfile.NamedTemporaryFile(delete=False) as tmp_file:
            tmp_file.write(source.read())
        source_text = model.transcribe(tmp_file.name, verbose=True)['text']
        return source_text

    except:
        pass

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
