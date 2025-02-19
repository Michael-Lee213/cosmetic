import matplotlib.pyplot as plt
import io
import base64
import json
from flask import Blueprint, jsonify

chart_bp = Blueprint("chart", __name__)

# 🔹 1. Matplotlib을 이용한 응답 시간 그래프 생성
@chart_bp.route("/response_time_chart")
def response_time_chart():
    """로그된 응답 시간을 Matplotlib을 이용해 시각화하는 엔드포인트"""
    log_file = "cos/response_time_log.json"

    try:
        with open(log_file, "r") as f:
            logs = json.load(f)

        # JSON 파일에서 timestamp와 response_time 데이터 추출
        timestamps = [log["timestamp"] for log in logs]
        response_times = [log["response_time"] for log in logs]

        # 오래된 순으로 정렬
        sorted_data = sorted(zip(timestamps, response_times))
        timestamps, response_times = zip(*sorted_data)

        # Matplotlib 그래프 생성
        plt.figure(figsize=(10, 5))
        plt.plot(
            timestamps,
            response_times,
            marker="o",
            linestyle="-",
            color="b",
            label="Response Time (s)",
        )
        plt.xlabel("Timestamp")
        plt.ylabel("Response Time (seconds)")
        plt.title("Chatbot Response Time Over Time")
        plt.legend()
        plt.grid(True)

        # 그래프를 이미지로 변환 (base64 인코딩)
        img = io.BytesIO()
        plt.savefig(img, format="png")
        img.seek(0)
        graph_url = base64.b64encode(img.getvalue()).decode()
        plt.close()

        return jsonify({"graph_url": f"data:image/png;base64,{graph_url}"})

    except Exception as e:
        return jsonify({"error": f"Error generating chart: {str(e)}"})


# 🔹 2. 이전 모델 응답 시간 데이터를 JSON으로 반환
@chart_bp.route("/previous_response_time_json")
def get_previous_response_times():
    """이전 모델 응답 시간을 JSON으로 반환"""
    log_file = "cos/previous_response_times.json"

    try:
        with open(log_file, "r") as f:
            previous_data = json.load(f)

        # 이전 데이터가 리스트 형태인지를 확인하고, 'response_time' 값만 추출
        if isinstance(previous_data, list):
            previous_response_times = [entry.get("response_time", 0) for entry in previous_data]
        else:
            # 잘못된 형식일 경우 빈 리스트를 반환
            previous_response_times = []

        return jsonify({
            "previous_response_times": previous_response_times
        })

    except json.JSONDecodeError:
        return jsonify({
            "error": "JSON Decode Error: Invalid JSON file format",
            "previous_response_times": []
        })

    except FileNotFoundError:
        return jsonify({
            "error": "File not found",
            "previous_response_times": []
        })

    except Exception as e:
        return jsonify({
            "error": f"Error loading previous response times: {str(e)}",
            "previous_response_times": []
        })