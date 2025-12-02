# app.py

from typing import List
from fastapi import FastAPI
from pydantic import BaseModel

from jobs_crawler import crawl_jobs
from news_crawler import crawl_news


# === DTO 모델 (자바 record와 필드명/케이스 맞추기) ===

class CrawledJobsDto(BaseModel):
    source: str
    externalId: str | None = None
    title: str | None = None
    company_name: str | None = None
    content: str | None = None
    url: str | None = None
    posted_at: str | None = None
    deadLine: str | None = None
    category: str | None = None
    tech_stack: str | None = None
    location: str | None = None
    exp_level: str | None = None
    thumbnail_url: str | None = None


class CrawledNewsDto(BaseModel):
    source: str
    externalId: str | None = None
    title: str | None = None
    content: str | None = None
    url: str | None = None
    published_at: str | None = None
    category: str | None = None
    reporter: str | None = None
    provider: str | None = None
    thumbnailUrl: str | None = None


app = FastAPI(title="GDG Job/News Crawling API")


@app.post("/crawl/jobs", response_model=List[CrawledJobsDto])
def crawl_jobs_endpoint():
    """
    채용공고 크롤링 후 자바로 넘기는 엔드포인트.
    """
    data = crawl_jobs()
    return data


@app.post("/crawl/news", response_model=List[CrawledNewsDto])
def crawl_news_endpoint():
    """
    뉴스 + 요약 크롤링 후 자바로 넘기는 엔드포인트.
    """
    data = crawl_news()
    return data
