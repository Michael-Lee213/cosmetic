import os
import pandas as pd
from gensim.models import Word2Vec

# CSV 데이터 경로 설정
CSV_PATH = os.path.join(os.path.dirname(__file__), "./data/Cos_data set_v2.9.csv")

# CSV 데이터 불러오기
try:
    df = pd.read_csv(CSV_PATH, encoding='utf-8-sig')
    print("✅ CSV 파일 로드 성공")
except FileNotFoundError:
    print(f"❌ CSV 파일을 찾을 수 없음: {CSV_PATH}")
    df = None

# 📌 벡터 DB에서 성분명 가져오기
if df is not None:
    df = df.dropna(subset=['ingredients']).reset_index(drop=True)  # 결측치 제거
    ingredient_list = df['ingredients'].tolist()  # 성분명 리스트 생성
    sentences = [ing.split(', ') for ing in ingredient_list]  # 단어 리스트 변환
else:
    sentences = []

# 📌 Word2Vec 모델 학습
MODEL_PATH = os.path.join(os.path.dirname(__file__), "word2vec_model.bin")

if sentences:
    model = Word2Vec(sentences, vector_size=100, window=5, min_count=1, workers=4)
    model.save(MODEL_PATH)
    print(f"✅ Word2Vec 모델 저장 완료: {MODEL_PATH}")
else:
    print("❌ 학습할 데이터가 없습니다.")

# 📌 저장된 모델 불러오기
try:
    model = Word2Vec.load(MODEL_PATH)
    print("✅ Word2Vec 모델 로드 완료")
except FileNotFoundError:
    print(f"❌ 모델 로드 실패: 파일을 찾을 수 없음 -> {MODEL_PATH}")
    model = None
except Exception as e:
    print(f"❌ 모델 로드 실패: {e}")
    model = None

# 📌 모든 성분 리스트 반환 함수
def get_all_ingredients():
    return model.wv.index_to_key if model else []

# 📌 특정 성분과 유사한 성분 찾기
def get_similar_ingredients(ingredient, topn=10):
    if model is None or ingredient not in model.wv:
        return []
    return model.wv.most_similar(ingredient, topn=topn)