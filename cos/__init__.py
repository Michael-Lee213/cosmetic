from flask import Flask

def create_app():
    app = Flask(__name__)

    #app.secret_key = "your_secret_key_here"  # 🔥 이걸 추가해야 세션이 정상 작동
    app.secret_key="key"


    # 🔹 views 폴더에 있는 블루프린트들 가져오기

    from cos.views.visualization_views import visualization_bp  # 새 Blueprint 추가
    app.register_blueprint(visualization_bp, url_prefix='/visualization')

    from cos.views.chart_views import chart_bp
    app.register_blueprint(chart_bp)

    from cos.views.chatbot_views import chatbot_bp
    app.register_blueprint(chatbot_bp)  # 챗봇 API 등록

    from cos.views.main_views import bp
    app.register_blueprint(bp)  # 메인 페이지 API 등록 (필요하면 추가)

    from cos.views.search_views import search_bp
    app.register_blueprint(search_bp, url_prefix='/')


    return app
