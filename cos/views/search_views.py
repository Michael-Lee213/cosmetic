from flask import Blueprint, render_template, request, jsonify, session
import pandas as pd
import numpy as np
import pickle
import re
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
from transformers import AutoTokenizer, AutoModel
from langchain.docstore.document import Document
import hashlib
import torch
from sklearn.metrics.pairwise import cosine_similarity

search_bp = Blueprint('search', __name__)

# 데이터 로드
cos_data_file = r'c:\projects\cosmetic\cos\data\Cos_data set_v2.9.csv'
cos_data_df = pd.read_csv(cos_data_file, encoding='utf-8-sig')

# ✅ 데이터 전처리 함수
def preprocess_ingredients(ingredients):
    if pd.isna(ingredients):
        return ''
    ingredients = re.sub(r'[a-zA-Z]', '', ingredients)  # 알파벳 제거
    ingredients = re.sub(r'[_\-]', '', ingredients)  # 언더바 및 대쉬 삭제
    ingredients = re.sub(r'\([^)]*\)', '', ingredients)  # 소괄호 내용 삭제
    ingredients = re.sub(r'\[[^\]]*\]', '', ingredients)  # 대괄호 내용 삭제
    ingredients = re.sub(r'[?]', '', ingredients)  # 특수문자 제거
    return ingredients

# 전처리 및 결측치 제거
cos_data_df = cos_data_df.dropna(subset=['ingredients']).reset_index(drop=True)
cos_data_df['ingredients'] = cos_data_df['ingredients'].apply(preprocess_ingredients)

# ✅ 임베딩 모델 설정
embedding_model = HuggingFaceEmbeddings(model_name="jhgan/ko-sroberta-multitask")

# ✅ 벡터 크기 확인
sample_vector = embedding_model.embed_query("테스트 문장")
vector_dim = len(sample_vector)

# ✅ 성분과 설명을 결합하여 벡터 생성 (가중치 적용)
def get_combined_embedding(text1, text2, weight1=0.9, weight2=0.1):
    vec1 = embedding_model.embed_query(text1) if text1 else np.zeros(vector_dim)
    vec2 = embedding_model.embed_query(text2) if text2 else np.zeros(vector_dim)
    return weight1 * np.array(vec1) + weight2 * np.array(vec2)

# ✅ 피클 파일 로딩 및 FAISS 인덱스 생성
try:
    # 피클 파일 로딩
    with open('vectorstore.pkl', 'rb') as f:
        vectorstore = pickle.load(f)
except FileNotFoundError:
    # FAISS 인덱스 생성
    documents = [
        Document(
            page_content=row['ingredients'],
            metadata={
                'brand_name': row['brand_name'],
                'product_name': row['product_name'],
                'description': row['description'],
                'price': row['price'],
                'image_url': row['image_url'],
                'detail_url': row['detail_url']
            }
        )
        for _, row in cos_data_df.iterrows()
    ]
    
    # FAISS 인덱스 생성
    vectorstore = FAISS.from_documents(documents, embedding=embedding_model)

    # 피클 파일로 저장
    with open('vectorstore.pkl', 'wb') as f:
        pickle.dump(vectorstore, f)

def calculate_similarity(doc, ingredient_list):
    if not ingredient_list:
        return 0.0

    ingredient_vectors = np.array([embedding_model.embed_query(ing) for ing in ingredient_list])
    avg_ingredient_vector = np.mean(ingredient_vectors, axis=0)

    doc_embedding = np.array(embedding_model.embed_query(doc.page_content))
    return cosine_similarity([doc_embedding], [avg_ingredient_vector])[0][0]


# ✅ 유사도 검색 함수 (필터링 추가)
def search_similar_cosmetics(ingredient_list, top_k=5):
    if not ingredient_list:
        return []

    ingredient_vectors = [embedding_model.embed_query(preprocess_ingredients(ing)) for ing in ingredient_list]
    query_vector = np.mean(ingredient_vectors, axis=0)

    results = vectorstore.similarity_search_with_score_by_vector(query_vector, k=top_k)
    hashed_results = {}

    for doc, score in results:
        doc_hash = hashlib.sha256((doc.page_content + doc.metadata['brand_name'] + doc.metadata['product_name']).encode('utf-8')).hexdigest()
        hashed_results[doc_hash] = (doc, score)

    if len(results) < top_k:
        remaining = top_k - len(results)
        candidate_docs = []
        
        for _, row in cos_data_df.iterrows():
            doc_vector = embedding_model.embed_query(row['ingredients'])
            similarity = cosine_similarity([doc_vector], [query_vector])[0][0]
            if similarity > 0.6:  # 유사도가 일정 기준 이상일 때만 추가
                candidate_docs.append((row, similarity))

        candidate_docs = sorted(candidate_docs, key=lambda x: x[1], reverse=True)[:remaining]

        for row, similarity in candidate_docs:
            doc = Document(
                page_content=row['ingredients'],
                metadata={
                    'brand_name': row['brand_name'],
                    'product_name': row['product_name'],
                    'description': row['description'],
                    'price': row['price'],
                    'image_url': row['image_url'],
                    'detail_url': row['detail_url']
                }
            )
            doc_hash = hashlib.sha256((doc.page_content + doc.metadata['brand_name'] + doc.metadata['product_name']).encode('utf-8')).hexdigest()
            if doc_hash not in hashed_results:
                hashed_results[doc_hash] = (doc, similarity)

    return sorted(hashed_results.values(), key=lambda x: x[1], reverse=True)


# ✅ 검색 정확도 계산 함수
def calculate_search_accuracy(recommended_docs, query_ingredients):
    if not recommended_docs or not query_ingredients:
        return 0.0

    query_vectors = [embedding_model.embed_query(ing) for ing in query_ingredients]
    avg_query_vector = np.mean(query_vectors, axis=0)

    similarity_scores = []
    for doc, _ in recommended_docs:
        doc_vector = embedding_model.embed_query(doc.page_content)
        similarity = cosine_similarity([doc_vector], [avg_query_vector])[0][0]
        similarity_scores.append(similarity)

    return np.mean(similarity_scores) if similarity_scores else 0.0


# ✅ 추천 다양성 계산 함수 (유사도 표준편차)
def calculate_recommendation_diversity(recommended_docs):
    if len(recommended_docs) < 2:
        return 0.0  # 비교할 대상이 없으면 0 반환

    similarities = []
    for idx, (doc1, _) in enumerate(recommended_docs):
        vec1 = embedding_model.embed_query(doc1.page_content)
        for doc2, _ in recommended_docs[idx + 1:]:
            vec2 = embedding_model.embed_query(doc2.page_content)
            sim_score = cosine_similarity([vec1], [vec2])[0][0]
            similarities.append(sim_score)

    avg_similarity = np.mean(similarities)
    std_dev = np.std(similarities)

    # 표준편차뿐만 아니라 평균 유사도를 함께 고려하여 다양성 점수 계산
    return 1 - avg_similarity + std_dev

# ✅ 평균 유사도 점수 계산 함수
def calculate_average_similarity(recommended_docs):
    if not recommended_docs:
        return 0.0
    return np.mean([score for _, score in recommended_docs])

recommended_keywords = [
    "토코페롤", "리모넨", "하이드로제네이티드레시틴", "판테놀", "리날룰", "아데노신", "나이아신아마이드", "글리세릴카프릴레이트",
    "스테아릭애씨드", "적색산화철", "글리세릴스테아레이트", "베타인", "아크릴레이트/C10-30알킬아크릴레이트크로스폴리머",
    "시트로넬올", "세라마이드엔피", "알란토인", "제라니올", "스쿠알란", "팔미틱애씨드", "티타늄디옥사이드", "변성알코올",
    "트라이에톡시카프릴릴실레인", "마이카", "펜틸렌글라이콜", "덱스트린", "시트릭애씨드", "다이아이소스테아일말레이트",
    "솔비탄아이소스테아레이트", "콜레스테롤"
]

@search_bp.route('/', methods=['GET', 'POST'])
def search():
    if request.method == 'POST':
        query = request.form.get('keywords', '').strip()
        selected_keywords = request.form.getlist('recommended_keywords')

        if not query and not selected_keywords:
            return render_template("results.html", message="성분을 입력해주세요.")

        query_ingredients = [q.strip() for q in query.split(',') if q.strip()]
        recommended_ingredients = selected_keywords

        # 검색 결과
        query_results = search_similar_cosmetics(query_ingredients, top_k=5)
        recommended_results = search_similar_cosmetics(recommended_ingredients, top_k=5)

        # 결과 통합 및 정렬
        sorted_results = sorted(query_results + recommended_results, key=lambda item: item[1], reverse=True)

        # # ✅ 정확도, 다양성, 평균 유사도 계산
        search_accuracy = calculate_search_accuracy(query_results, query_ingredients)
        recommendation_diversity = calculate_recommendation_diversity(sorted_results)
        average_similarity = calculate_average_similarity(sorted_results)

        # ✅ 정확도, 다양성, 평균 유사도 계산 후 세션 저장
        session['search_accuracy'] = float(calculate_search_accuracy(query_results, query_ingredients))
        session['recommendation_diversity'] = float(calculate_recommendation_diversity(sorted_results))
        session['average_similarity'] = float(calculate_average_similarity(sorted_results))


        print("🔍 [DEBUG] 검색 정확도:", search_accuracy)
        print("🔍 [DEBUG] 추천 다양성:", recommendation_diversity)
        print("🔍 [DEBUG] 평균 유사도:", average_similarity)

        # ✅ 터미널에 검색 결과 출력
        print("\n🔍 검색어:", query_ingredients)
        print("📌 추천 성분:", recommended_ingredients)
        print(f"🔎 총 검색된 제품 수: {len(sorted_results)} 개\n")
        print(f"✅ 검색 정확도: {search_accuracy:.4f}")
        print(f"✅ 추천 다양성 (유사도 표준편차): {recommendation_diversity:.4f}")
        print(f"✅ 평균 유사도 점수: {average_similarity:.4f}\n")


        search_accuracy = calculate_search_accuracy(query_results, query_ingredients)
        recommendation_diversity = calculate_recommendation_diversity(sorted_results)
        average_similarity = calculate_average_similarity(sorted_results)

        # ✔ 출력값이 numpy.float32라면 float() 변환이 필요함!
        # ✔ 출력값이 float이면 session에 바로 저장 가능!

        print(f"🔍 [DEBUG] 검색 정확도 타입: {type(search_accuracy)}")
        print(f"🔍 [DEBUG] 추천 다양성 타입: {type(recommendation_diversity)}")
        print(f"🔍 [DEBUG] 평균 유사도 타입: {type(average_similarity)}")




        for idx, (doc, score) in enumerate(sorted_results[:10]):  # 상위 10개만 출력
            print(f"{idx+1}. {doc.metadata['brand_name']} - {doc.metadata['product_name']}")
            print(f"   ➡️ 설명: {doc.metadata['description']}")
            print(f"   💰 가격: {doc.metadata['price']}원")
            print(f"   🔗 링크: {doc.metadata['detail_url']}")
            print(f"   🧪 성분: {doc.page_content}")  # 성분 정보 출력
            print(f"   ✅ 유사도 점수: {score:.4f}\n")

        return render_template('results.html', query=query, results=sorted_results, selected_ingredients=query_ingredients + recommended_ingredients)

    return render_template('input.html', recommended_keywords=selected_keywords)