

# 맞춤형 화장품 추천 알고리즘 개발 및 웹 서비스 구현

 사용자의 화장품 성분 데이터를 기반으로 맞춤형 추천 알고리즘을 개발하여 웹 서비스를 제공합니다.<br>
 벡터 유사도 검색 기법을 활용해 사용자 경험을 개선하고 데이터 기반 의사 결정을 지원합니다.<br>
<table>
  <thead>
    <tr>
      <th>항목</th>
      <th>내용</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td>개발 기간</td>
      <td>2025년 1월 31일 ~ 2025년 2월 17일</td>
    </tr>
    <tr>
      <td>팀 구성</td>
      <td>김미경, 하연우, 이한세</td>
    </tr>
    <tr>
      <td>프로젝트 목표</td>
      <td>
        <ul>        
          <li>K 뷰티의 관심도 증가로 인한 트랜드 파악 및 성분 별 맞춤 화장품 추천 서비스 개발</li>  
          <li>개발된 알고리즘을 활용한 웹 서비스 구현</li>
          <li>사용자 편의성을 고려한 직관적인 UI/UX 디자인</li>
          <li>화장품 성분 정보 및 Q&A 챗봇 기능 제공</li>
        </ul>
      </td>
    </tr>
  </tbody>
</table>
<br>
< 스킨케어 트랜드 파악 워드 클라우드 자료 >
<div>
  <img src="static/images/keyword_1.webp" style="width:300px; height:200px; border-radius: 5px;">
  <img src="static/images/keyword_2.webp" style="width:300px; height:200px; border-radius: 5px;">
</div>

# 목차
  1. 소개 및 일정
  2. 사용 기술 스택 (Stacks)
  3. 화면 구성 (UI Components)



# 프로젝트 진행 관리
<div>
  <img src="static/images/Untitled_diagram_1.webp" style="width:300px; height:200px; border-radius: 5px;">
  <img src="static/images/Untitled_diagram_2.webp" style="width:300px; height:200px; border-radius: 5px;">
</div>



# Stacks
<div style="text-align: left;">
<h3>Environment</h3>
<br>
<div style="display: flex; flex-wrap: wrap; gap: 10px; align-items: center;">
<img src="https://img.shields.io/badge/visual studio code-008FC7?style=for-the-badge&logo=visual studio code&logoColor=white">
<img src="https://img.shields.io/badge/jupyter-F37626?style=for-the-badge&logo=jupyter&logoColor=white">
<img src="https://img.shields.io/badge/github-181717?style=for-the-badge&logo=github&logoColor=white">
<img src="https://img.shields.io/badge/git-F05032?style=for-the-badge&logo=git&logoColor=white">
</div>

<h3>Frontend</h3>
<div style="display: flex; flex-wrap: wrap; gap: 10px; align-items: center;">
<img src="https://img.shields.io/badge/html5-E34F26?style=for-the-badge&logo=html5&logoColor=white">
<img src="https://img.shields.io/badge/css-1572B6?style=for-the-badge&logo=css3&logoColor=white">
<img src="https://img.shields.io/badge/javascript-F7DF1E?style=for-the-badge&logo=javascript&logoColor=black">
<img src="https://img.shields.io/badge/bootstrap 5-7952B3?style=for-the-badge&logo=bootstrap&logoColor=white">
<img src="https://img.shields.io/badge/chart.js-FF6384?style=for-the-badge&logo=chartdotjs&logoColor=white">
<img src="https://img.shields.io/badge/plotly.js-3b4cc0?style=for-the-badge&logo=plotly&logoColor=white">
<img src="https://img.shields.io/badge/ajax (jQuery)-0769AD?style=for-the-badge&logo=jquery&logoColor=white">
</div>

<h3>Backend</h3>
<div style="display: flex; flex-wrap: wrap; gap: 10px; align-items: center;">
<img src="https://img.shields.io/badge/python-3776AB?style=for-the-badge&logo=python&logoColor=white">
<img src="https://img.shields.io/badge/flask-000000?style=for-the-badge&logo=flask&logoColor=white">
<img src="https://img.shields.io/badge/langchain-00A6FF?style=for-the-badge&logo=langchain&logoColor=white">
</div>

<h3>Database & Storage</h3>
<div style="display: flex; flex-wrap: wrap; gap: 10px; align-items: center;">
<img src="https://img.shields.io/badge/faiss vector DB-008000?style=for-the-badge&logo=faiss&logoColor=white">
</div>

<h3>Machine Learning & AI</h3>
<div style="display: flex; flex-wrap: wrap; gap: 10px; align-items: center;">
<img src="https://img.shields.io/badge/numpy-013243?style=for-the-badge&logo=numpy&logoColor=white">
<img src="https://img.shields.io/badge/pandas-150458?style=for-the-badge&logo=pandas&logoColor=white">
<img src="https://img.shields.io/badge/matplotlib-11557c?style=for-the-badge&logo=matplotlib&logoColor=white">
<img src="https://img.shields.io/badge/scikit learn-F7931E?style=for-the-badge&logo=scikit-learn&logoColor=white">
<img src="https://img.shields.io/badge/word2vec-FFCC00?style=for-the-badge&logo=google&logoColor=black">
<img src="https://img.shields.io/badge/faiss-008000?style=for-the-badge&logo=facebook&logoColor=white">
<img src="https://img.shields.io/badge/HuggingFaceEmbeddings-FF9900?style=for-the-badge&logo=huggingface&logoColor=black">
<img src="https://img.shields.io/badge/cosine similarity-663399?style=for-the-badge&logo=mathworks&logoColor=white">
<img src="https://img.shields.io/badge/transformers-DC143C?style=for-the-badge&logo=ai&logoColor=white">
<img src="https://img.shields.io/badge/genai-FF66CC?style=for-the-badge&logo=google&logoColor=white">
<img src="https://img.shields.io/badge/youtube api-FF0000?style=for-the-badge&logo=youtube&logoColor=white">
</div>
<br>
<br>

# 화면 구성

<div style="display: flex; flex-direction: column; gap: 20px;">
  <h4>메인 화면 페이지</h4>
  <div style="display: flex; align-items: center; gap: 20px;">
    <img src="static/images/main_1.png" alt="메인 화면 페이지" style="width:300px; height:200px; border-radius: 5px;">
    <img src="static/images/main_2.png" alt="메인 화면 페이지" style="width:300px; height:200px; border-radius: 5px;">
    <p>메인 화면에서는 사용자가 성분을 입력할 수 있는 입력 폼과 검색 기능을 제공합니다.</p>
  </div>

  <h4>성분 추천 결과 페이지</h4>
  <div style="display: flex; align-items: center; gap: 20px;">
    <img src="static/images/results.png" alt="성분 추천 결과 페이지" style="width:300px; height:200px; border-radius: 5px;">
    <img src="static/images/visualization_1.png" alt="추천 결과 시각화 페이지" style="width:300px; height:200px; border-radius: 5px;">
    <p>성분 추천 결과 페이지에서는 입력된 성분을 바탕으로 AI 모델이 추천한 유사 성분 리스트와 시각화된 데이터를 확인할 수 있습니다.</p>
  </div>

  <h4>추천 결과 시각화 페이지</h4>
  <div style="display: flex; align-items: center; gap: 20px;">
    <img src="static/images/visualization_2.png" alt="추천 결과 시각화 페이지" style="width:300px; height:200px; border-radius: 5px;">
    <p>추천 결과 시각화 페이지에서는 Plotly.js와 Chart.js를 사용하여 성분 추천 결과를 다양한 차트와 그래프로 시각화하여 제공합니다.</p>
  </div>

  <h4>Chatbot</h4>
  <div style="display: flex; align-items: center; gap: 20px;">
    <img src="static/images/chatbot_after_1.png" alt="chatbot" style="width:200px; height:300px; border-radius: 5px;">
    <img src="static/images/chatbot_after_2.png" alt="chatbot" style="width:200px; height:300px; border-radius: 5px;">
    <p>Chatbot은 사용자의 질문에 대해 LLM 기반 Langchain을 활용하여 성분 추천 및 제품 정보 검색 기능을 제공하는 AI 챗봇입니다.</p>
  </div>
</div>
