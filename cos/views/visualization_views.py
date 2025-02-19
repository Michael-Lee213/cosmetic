from flask import Blueprint, render_template, jsonify, request, session
import numpy as np
import re
from cos.word2vec_model import get_all_ingredients, model

visualization_bp = Blueprint("visualization", __name__)

@visualization_bp.route("/")
def visualization():
    return render_template("visualization.html")

# ✅ 텍스트 길이를 줄이는 함수 추가
def truncate_text(text, max_length=15):
    return text if len(text) <= max_length else text[:max_length] + "…"

# ✅ 긴 성분을 적절히 나누는 토크나이저 함수 추가
def tokenize_text(text):
    # 특수문자 및 불필요한 문자 제거
    text = re.sub(r"[^가-힣a-zA-Z0-9\s]", "", text)
    # 공백 기준으로 분할하여 단어 리스트 생성
    tokens = text.split()
    # 단어가 없을 경우 원래 텍스트 반환
    return tokens if tokens else [text]

# ✅ GET과 POST 모두 지원 (POST: 특정 성분 검색, GET: 초기 데이터)
@visualization_bp.route("/data", methods=["GET", "POST"])
def visualization_data():
    data = request.get_json() if request.method == "POST" else {}
    selected_ingredient = data.get("selected_ingredient", "").strip()

    print(f"🔍 [DEBUG] 요청된 성분: {selected_ingredient}")

    search_accuracy = session.get("search_accuracy", 0.0)
    recommendation_diversity = session.get("recommendation_diversity", 0.0)
    average_similarity = session.get("average_similarity", 0.0)

    all_ingredients = get_all_ingredients()
    input_ingredients = session.get("selected_ingredients", [])

    # ✅ 토큰화하여 긴 성분을 적절히 분할
    tokenized_selected_ingredient = tokenize_text(selected_ingredient) if selected_ingredient else []

    # ✅ 특정 성분이 입력된 경우 토큰화된 성분을 기준으로 유사한 성분 100개 가져오기
    if tokenized_selected_ingredient and any(token in model.wv for token in tokenized_selected_ingredient):
        try:
            words = []
            for token in tokenized_selected_ingredient:
                if token in model.wv:
                    words.append(token)
                    words += [word for word, _ in model.wv.most_similar(token, topn=49)]  # 각 토큰에서 49개씩 가져오기
            words = list(set(words))[:100]  # 중복 제거 후 100개 제한
        except KeyError:
            print(f"❌ [ERROR] '{selected_ingredient}'의 일부 또는 전체가 Word2Vec 모델에 없음!")
            return jsonify({"error": f"'{selected_ingredient}'의 일부 또는 전체가 Word2Vec 모델에 없습니다."}), 400
    else:
        words = all_ingredients[:100]

    # ✅ 모델에서 유효한 벡터 가져오기
    valid_ingredients = [word for word in words if word in model.wv]

    if valid_ingredients:
        vectors = np.array([model.wv[word] for word in valid_ingredients])
    else:
        np.random.seed(42)
        vectors = np.random.rand(100, 3) * 100
        valid_ingredients = [f"성분{i}" for i in range(100)]

    word2vec_x = vectors[:, 0].tolist()
    word2vec_y = vectors[:, 1].tolist()
    word2vec_z = vectors[:, 2].tolist()

    # ✅ 긴 성분명을 줄여서 표시 (Plotly에 표시할 용도)
    shortened_labels = [truncate_text(word) for word in valid_ingredients]

    # ✅ 중심 성분 위치를 3D 중심에 배치
    if selected_ingredient and any(token in valid_ingredients for token in tokenized_selected_ingredient):
        first_valid_token = next((token for token in tokenized_selected_ingredient if token in valid_ingredients), None)
        if first_valid_token:
            index = valid_ingredients.index(first_valid_token)
            word2vec_x.insert(0, word2vec_x.pop(index))
            word2vec_y.insert(0, word2vec_y.pop(index))
            word2vec_z.insert(0, word2vec_z.pop(index))
            shortened_labels.insert(0, shortened_labels.pop(index))

    return jsonify({
        "selected_ingredient": selected_ingredient,
        "search_accuracy": search_accuracy,
        "recommendation_diversity": recommendation_diversity,
        "average_similarity": average_similarity,
        "word2vec_x": word2vec_x,
        "word2vec_y": word2vec_y,
        "word2vec_z": word2vec_z,
        "word2vec_labels": shortened_labels,
        "hover_labels": valid_ingredients,
        "input_ingredients": input_ingredients
    })
