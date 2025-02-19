import os
import pickle
import google.generativeai as genai
import json
import numpy as np
import pandas as pd
import faiss
from flask import Blueprint, request, jsonify, render_template
from langchain.schema import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
import time


chatbot_bp = Blueprint('chatbot', __name__)

# 🔹 경로 설정
FAISS_INDEX_DIR = "cos/ingredients_faiss_index"
PICKLE_VECTORSTORE_PATH = os.path.join(FAISS_INDEX_DIR, "vectorstore.pkl")
CSV_PATH = "cos/data/paulas_choice_ingredients_all_v2.csv"


# 🔹 API 키 설정
gemini_api_key = "you api key"
genai.configure(api_key=gemini_api_key)

#🔹 HuggingFace 임베딩 모델 로드
def load_embedding():
    return HuggingFaceEmbeddings(model_name="sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2")

# 🔹 FAISS 벡터 저장소 로드
def load_vectorstore(embedding_model):
    if os.path.exists(PICKLE_VECTORSTORE_PATH):
        with open(PICKLE_VECTORSTORE_PATH, "rb") as f:
            vectorstore = pickle.load(f)
    else:
        df = pd.read_csv(CSV_PATH)
        df['ingredients'] = df['ingredients'].fillna("")

        documents = [Document(page_content=row['ingredients'], metadata={"source": row['ingredients']}) for _, row in df.iterrows()]
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
        splits = text_splitter.split_documents(documents)
        
        vectorstore = FAISS.from_documents(splits, embedding=embedding_model)

        with open(PICKLE_VECTORSTORE_PATH, "wb") as f:
            pickle.dump(vectorstore, f)

    return vectorstore

# 🔹 모델 및 벡터 저장소 초기화
embedding_model = load_embedding()
vectorstore = load_vectorstore(embedding_model)
gemini_model = genai.GenerativeModel('gemini-1.5-flash')

def generate_ai_response(prompt):
    """Gemini 모델의 성능 최적화를 위한 AI 응답 생성 함수"""
    try:
        response = gemini_model.generate_content(
            prompt,
            generation_config={
                "temperature": 0.7,  # ✅ 창의성 조절 (낮을수록 일관된 답변 제공)
                "top_p": 0.9,  # ✅ 응답 품질 향상 (최고 확률의 답변 선택)
                "frequency_penalty": 0.3,  # ✅ 동일한 단어 반복 방지
                "presence_penalty": 0.2   # ✅ 새로운 정보 생성 유도
            }
        )
        
        # ✅ 응답이 여러 개일 경우, 가장 신뢰할 수 있는 답변을 선택
        if response and response.candidates:
            for candidate in response.candidates:
                if candidate.content and candidate.content.parts:
                    return candidate.content.parts[0].text.strip()

        return "⚠️ AI 응답을 생성할 수 없습니다. 다시 시도해 주세요."

    except Exception as e:
        return f"⚠️ 오류 발생: {str(e)}"


# 🔹 RAG 기반 답변 생성
def rag_chatbot(question):

    # "챗봇 분석"에 대한 질문이 들어온 경우
    if "챗봇 분석" in question:
        return '챗봇 응답 시간 분석을 보려면 <a href="http://127.0.0.1:5000/chart" target="_blank">여기</a>를 클릭하세요!'

    retrieved_docs = vectorstore.similarity_search(question, k=5)
    context_texts = "\n".join([doc.page_content for doc in retrieved_docs])

    prompt = f"""

[참고 문서]
{context_texts}

[질문]
{question}

[응답 가이드]
- 질문에 **"화장품 추천"**이 포함되면, 모든 응답의 **첫 줄**에  
  **"❗ 특정 화장품 추천은 어렵습니다. 대신, 성분 추천이 가능합니다. 원하는 성분이나 피부 고민을 말씀해주세요!"** 를 추가하세요.
- 질문에 "화장품"이라는 단어가 포함되면, 제품 추천 대신 **성분 정보**와 **피부 타입** 관련 내용을 중심으로 답변하세요.
- 질문이 **특정 성분**(예: 히알루론산, 티트리 오일 등)에 대한 것이라면 **개조식**으로 정보를 제공하세요.  
- 질문이 **피부 고민**이나 **사용법 관련 조언**인 경우에는 **대화형 답변**을 제공하세요.
- **핵심 정보만 제공**: 불필요한 내용은 생략하고, 중요한 포인트만 강조합니다.
- **목록 형식 활용**: 성분 정보나 효과 등을 리스트 형식으로 제공하여 가독성을 높이세요.
- **간결한 문장 사용**: 문장은 최대 1~2개의 핵심 포인트만 담고, 7~10줄 이내로 유지하세요.
- **불필요한 세부사항 생략**: 과도한 배경 설명을 줄이고, 핵심 내용에 집중하세요.
- **대화형 답변**: 친절하면서도 핵심적인 정보를 빠르게 전달하세요.


---

[질문 유형별 응답 방식]

🟢 **질문에 특정 성분이 포함된 경우** (예: "히알루론산 효능이 뭐야?")  
→ 아래 형식으로 개조식으로 답변하세요.

1️⃣ **성분명 (영문명)**  
🔹 **효과**: 주요 피부 효능 요약  
🔹 **특징**: 성분의 핵심적인 역할  
🔹 **활용**: 어떤 제품에 포함되는지  
🔹 **주의사항**: 사용 시 주의할 점  

**[예시]**  
1️⃣ **살리실산 (Salicylic Acid)**  
🔹 **효과**: 각질 제거, 모공 청결, 피지 조절  
🔹 **특징**: 지용성 BHA로 모공 속 깊이 침투하여 피지 분해  
🔹 **활용**: 여드름 전용 클렌저, 토너, 스팟 트리트먼트  
🔹 **주의사항**: 건조함 유발 가능, 저농도부터 사용 권장  

---

🟡 **질문이 일반적인 피부 고민이나 조언 요청인 경우**  
→ 자연스러운 대화형 답변을 생성하세요.  
→ 너무 기계적인 답변이 아니라 친절하게 설명하세요.  

**[예시]**  
🙋‍♂️ "자외선 차단제(선크림)을 고를 때 어떤 점을 주의해야 하나요?"  
🤖 "좋은 질문이에요! 선크림을 고를 때는 SPF와 PA 등급을 확인하는 것이 중요해요.  
SPF 30~50 정도가 일상생활에 적합하고, PA+++ 이상이면 자외선 차단 효과가 좋아요.  
또한, 지성 피부라면 산뜻한 젤 타입을, 건성 피부라면 보습력이 좋은 크림 타입을 선택하는 게 좋아요!"

🙋‍♂️ "여드름 피부인데 어떤 성분을 피해야 하나요?"  
🤖 "여드름 피부라면 **코코넛 오일, 미네랄 오일, 라놀린** 같은 성분은 피하는 게 좋아요.  
이 성분들은 모공을 막아 여드름을 악화시킬 수 있거든요! 대신 **살리실산, 나이아신아마이드** 같은 성분이 도움이 될 수 있어요."

---

⚠️ **추가 주의사항**  
- 만약 질문이 애매한 경우, "질문을 조금 더 구체적으로 해주시면 더 정확한 정보를 제공해 드릴 수 있어요!" 라고 답변하세요.  
- 대화형 답변에서도 핵심 내용을 빠르게 전달할 수 있도록 명확한 정보를 포함하세요.  
- **화장품 브랜드나 제품명을 직접 추천하지 말고**, 성분 정보만 제공하세요.  
- 모든 성분은 개인의 피부 타입에 따라 반응이 다를 수 있습니다.
- 새로운 성분을 사용할 때는 반드시 패치 테스트 후 사용하세요.
- **모든 응답에서 "화장품 추천"이 포함된 경우 첫 줄에 메시지를 추가해야 합니다.**
- **"화장품 추천" 요청이 들어오면 제품명이 아닌 성분 정보와 피부 타입 중심으로 답변하세요.**

---
"❗ 특정 화장품 추천은 어렵습니다. 대신, 성분 추천이 가능합니다. 원하는 성분이나 피부 고민을 말씀해주세요!"
---
"""
   
    print(context_texts)
    
    # response = gemini_model.generate_content(prompt)
    # ✅ 개선된 Gemini API 호출
    answer = generate_ai_response(prompt)


    return answer



# 🔹 응답 시간 로그 저장 함수
LOG_FILE_PATH = "cos/response_time_log.json"

def log_response_time(response_time):
    """응답 시간을 JSON 파일에 저장하는 함수"""
    log_entry = {
        "timestamp": time.time(),
        "response_time": response_time
    }

    try:
        # 기존 로그 불러오기 (파일이 없으면 빈 리스트)
        logs = []
        if os.path.exists(LOG_FILE_PATH):
            with open(LOG_FILE_PATH, "r") as f:
                logs = json.load(f)

        # 새 로그 추가
        logs.append(log_entry)

        # 100개 이상의 데이터가 쌓이면 오래된 데이터 제거
        if len(logs) > 100:
            logs.pop(0)

        # 로그 파일 업데이트
        with open(LOG_FILE_PATH, "w") as f:
            json.dump(logs, f, indent=4)

    except Exception as e:
        print(f"로그 저장 중 오류 발생: {str(e)}")  # 서버 로그 출력


@chatbot_bp.route("/chatbot")
def chatbot_home():
    return render_template("input.html")  # 🔹 챗봇 UI 페이지 (input.html 사용)

# @chatbot_bp.route("/chatbot")
# def chatbot_home():
#     return render_template("result.html")  # 🔹 챗봇 UI 페이지 (input.html 사용)

@chatbot_bp.route("/ask", methods=["POST"])
def ask():
    data = request.json
    user_query = data.get("question", "").strip()

    if not user_query:
        return jsonify({"answer": "질문을 입력해주세요!"})

    try:
        start_time = time.time()  # 시작 시간 기록
        response = rag_chatbot(user_query)  # ✅ Gemini API 호출
        end_time = time.time()  # 종료 시간 기록

        response_time = end_time - start_time  # 응답 시간 계산
        log_response_time(response_time)  # ✅ 응답 시간 로그 저장
        
        return jsonify({"answer": response, "response_time": response_time})
    except Exception as e:
        return jsonify({"answer": f"오류 발생: {str(e)}"})

@chatbot_bp.route("/chart")
def show_chart():
    return render_template("charts.html")

@chatbot_bp.route("/response_time_chart_json")
def response_time_chart_json():
    """응답 시간 데이터를 JSON으로 반환하는 엔드포인트"""
    log_file = "cos/response_time_log.json"

    try:
        with open(log_file, "r") as f:
            logs = json.load(f)

        timestamps = [log["timestamp"] for log in logs]
        response_times = [log["response_time"] for log in logs]

        # 오래된 순으로 정렬
        sorted_data = sorted(zip(timestamps, response_times))
        timestamps, response_times = zip(*sorted_data)

        return jsonify({"timestamps": timestamps, "response_times": response_times})

    except Exception as e:
        return jsonify({"error": f"Error loading response times: {str(e)}"})

