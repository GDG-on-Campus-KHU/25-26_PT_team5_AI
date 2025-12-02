# server.py
from flask import Flask, jsonify
from jobs_crawler import crawl_jobs
from news_crawler import crawl_news

app = Flask(__name__)

@app.get("/api/jobs")
def api_jobs():
    """
    채용 정보 크롤링 후 JSON으로 반환
    """
    data = crawl_jobs()          # list[dict]
    return jsonify(data)         # Java에서 그대로 JSON으로 받기

@app.get("/api/news")
def api_news():
    """
    요즘IT 기사 크롤링 + 요약 후 JSON으로 반환
    """
    data = crawl_news()          # list[dict]
    return jsonify(data)

if __name__ == "__main__":
    # 0.0.0.0 으로 열면 외부(같은 LAN, Docker 등)에서도 접속 가능
    app.run(host="0.0.0.0", port=5000, debug=False)