# 1. 요즘 IT
from selenium import webdriver
from selenium.common.exceptions import TimeoutException, NoSuchElementException
import time
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
import pandas as pd
import torch
import numpy as np
from transformers import AutoTokenizer, AutoModel
import kss
import requests
from bs4 import BeautifulSoup
from datetime import date, timedelta,datetime


SOURCE = "YOZM_IT" 
BASE_URL = "https://yozm.wishket.com"

MODEL_NAME = "skt/kobert-base-v1"
device = torch.device("cpu")
yesterday = date.today() - timedelta(days=1)
yesterday_format = yesterday.strftime("%Y.%m.%d")
today = date.today() 
today_format = today.strftime("%Y.%m.%d")

# 1) 모델 & 토크나이저 로드
print("▶ KoBERT 모델 로드 중...")
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
model = AutoModel.from_pretrained(MODEL_NAME)
model.to(device)
model.eval()
print("▶ 로드 완료!")

# 2) 한국어 문장 분리
def split_sentences_kor(text: str):
    """
    긴 한국어 텍스트를 문장 단위 리스트로 분리.
    kss를 사용해서 안전하게 문장 분리.
    """
    sentences = kss.split_sentences(text)
    sentences = [s.strip() for s in sentences if s.strip()]
    return sentences


# 3) 문장들을 KoBERT CLS 임베딩으로 변환
def encode_sentences(sentences, batch_size: int = 8, max_length: int = 256):
    """
    문장 리스트 -> CLS 임베딩 (numpy array: [num_sent, hidden_dim])
    """
    all_embeddings = []

    with torch.no_grad():
        for i in range(0, len(sentences), batch_size):
            batch = sentences[i:i+batch_size]

            enc = tokenizer(
                batch,
                padding=True,
                truncation=True,
                max_length=max_length,
                return_tensors="pt"
            )

            # 🔍 디버그: input_ids 범위 확인
            ids = enc["input_ids"]
            #print("min id:", ids.min().item(), "max id:", ids.max().item())

            # 🔥 중요: token_type_ids 강제로 제거
            if "token_type_ids" in enc:
                #print("-> token_type_ids 제거")
                enc.pop("token_type_ids")

            enc = {k: v.to(device) for k, v in enc.items()}

            outputs = model(**enc)
            # BERT의 [CLS] 토큰 벡터 사용
            cls_embeddings = outputs.last_hidden_state[:, 0, :]  # [batch, hidden]
            all_embeddings.append(cls_embeddings.cpu().numpy())

    return np.vstack(all_embeddings)  # [num_sent, hidden_dim]


# 4) 코사인 유사도로 중요한 문장 top_k 추출
def summarize_kobert(text: str, top_k: int = 5):
    """
    긴 텍스트를
    1) 문장 분리
    2) 각 문장을 KoBERT로 임베딩
    3) 문장 임베딩 vs 문서 평균 임베딩 코사인 유사도 계산
    4) 유사도가 큰 상위 top_k 문장을 원래 순서대로 뽑기
    """
    sentences = split_sentences_kor(text)

    if len(sentences) == 0:
        return "", []

    # 문장 수가 top_k보다 적으면 그냥 전체 반환
    if len(sentences) <= top_k:
        summary = " ".join(sentences)
        return summary, sentences

    print(f"▶ 문장 개수: {len(sentences)}개")
    sent_embs = encode_sentences(sentences)  # [N, D]

    # 문서 임베딩 = 문장 임베딩 평균
    doc_emb = sent_embs.mean(axis=0, keepdims=True)  # [1, D]

    # L2 정규화 후 코사인 유사도 계산
    def l2norm(x, axis):
        return x / (np.linalg.norm(x, axis=axis, keepdims=True) + 1e-8)

    sent_norm = l2norm(sent_embs, axis=1)  # [N, D]
    doc_norm = l2norm(doc_emb, axis=1)     # [1, D]

    sims = (sent_norm @ doc_norm.T).squeeze(1)  # [N]

    # 유사도 높은 상위 top_k 문장 인덱스
    top_idx = sims.argsort()[::-1][:top_k]
    # 문서 원래 순서 유지 (요약문이 자연스럽게 읽히도록)
    top_idx_sorted = sorted(top_idx)

    summary_sentences = [sentences[i] for i in top_idx_sorted]
    summary_text = " ".join(summary_sentences)

    return summary_text, summary_sentences

def crawl_news():
    options = webdriver.ChromeOptions()
    options.add_argument("--start-maximized")
    driver = webdriver.Chrome(options=options)

    provider = "yozmIT"
    url = "https://yozm.wishket.com/magazine/list/new/"
    driver.get(url)
    wait = WebDriverWait(driver, 10)

    # 페이지 로딩 기다리기 (적당한 상위 엘리먼트 기준으로)
    wait.until(EC.presence_of_element_located((By.CSS_SELECTOR, "article")))

    articles = driver.find_elements(By.CSS_SELECTOR, "article")

    results = []

    for art in articles:
        try:

            # 이 article 안에 '1일 전'이 있는지 확인
            date_el = art.find_element(
                By.XPATH,
                ".//span[contains(normalize-space(.), '1일 전') or contains(normalize-space(.), '시간 전')]"
            )
        except:
            # 이 카드에는 '1일 전'이 없음 → 스킵
            continue

        # 제목
        title_el = art.find_element(By.TAG_NAME, "h3")
        title = title_el.text.strip()

        # 날짜 (여기서는 '1일 전'일 것)
        date_text = date_el.text.strip()

        # 링크 (카드 전체를 감싸는 a 태그)
        link_el = art.find_element(
            By.XPATH,
            ".//a[@data-testid='contentsItem-item-link']"
        )
        href = link_el.get_attribute("href")
        article_id = href.rstrip("/").split("/")[-1]
        thumbnail_url = None
        try:
            # column 스타일 카드에 해당
            img_el = art.find_element(
                By.XPATH,
                ".//div[@data-testid='article-column-item--image']//img"
            )
            thumbnail_url = img_el.get_attribute("src")
        except Exception:
            # 위 구조가 없다면, 카드 안의 object-cover 이미지를 fallback으로 사용
            try:
                img_el = art.find_element(
                    By.XPATH,
                    ".//img[contains(@class, 'object-cover')]"
                )
                thumbnail_url = img_el.get_attribute("src")
            except Exception:
                thumbnail_url = None  # 정말 없으면 None

        results.append({
            "source": SOURCE,
            "externalId": article_id,
            "title": title,
            "date": date_text,
            "url": href,
            "provider" : provider,
            'thumbnail_url' : thumbnail_url

        })

    print(results)
    detail_results = []

    for item in results:   # list_results는 앞에서 모아둔 {link, thumbnail, ...}
        driver.get(item["url"])
        def parse_article_detail(driver, timeout=15):
            data = {}
            local_wait = WebDriverWait(driver, timeout)

            # 1) 제목
            # <h1 class="... typo-title3 desktop:typo-title2">...</h1>
            title_el = driver.find_element(
                By.CSS_SELECTOR,
                "h1.typo-title3, h1.typo-title2"
            )
            data["title"] = title_el.text.strip()

            # 2) 글쓴이 (작성자)
            # <span ... data-testid="contents-author-name">FEConf</span>
            author_el = driver.find_element(
                By.CSS_SELECTOR,
                "span[data-testid='contents-author-name']"
            )
            data["author"] = author_el.text.strip()

            # 3) 게시 날짜 (상대 시간: '1일 전')
            # <span class="... typo-body2 ...">1일 전</span>
            date_el = driver.find_element(
                By.XPATH,
                "//span[contains(@class, 'typo-body2') and (contains(normalize-space(.), '1일 전') or contains(normalize-space(.), '시간 전'))]"
            )
            a = date_el.text.strip()
            if a == '1일 전':
                data["posted_at"] = yesterday_format  # 예: '1일 전'
            else:
                data["posted_at"] = today_format

            # 4) 카테고리
            # <a data-testid="category-link" ...><span ...>개발</span></a>
            category_el = driver.find_element(
                By.CSS_SELECTOR,
                "a[data-testid='category-link'] span"
            )
            data["category"] = category_el.text.strip()   # 예: '개발'

            # 5) 본문 내용
            # <section id="article-detail-wrapper"> ... 여기 안의 p, h3, h4, blockquote 등 전체 텍스트
            content_section = local_wait.until(EC.presence_of_element_located((
                By.CSS_SELECTOR,
                "section#article-detail-wrapper"
            )))

            # 👉 문단 개수 너무 빡빡하게 보지 말고, 실패해도 그냥 진행
            try:
                local_wait.until(
                    lambda d: len(
                        d.find_elements(
                            By.CSS_SELECTOR,
                            "section#article-detail-wrapper p.typo-contents2"
                        )
                    ) >= 1     # 최소 1개만 나오면 통과
                )
            except TimeoutException:
                # 문단이 적거나 늦게 떠도 그냥 현재 있는 것만 긁고 넘어가기
                pass

            paragraph_els = content_section.find_elements(
            By.XPATH,
            ".//p | .//h3 | .//h4 | .//blockquote"
            )

            paragraphs = [el.text.strip() for el in paragraph_els if el.text.strip()]

            # 섹션 안의 모든 텍스트를 줄바꿈 포함해서 가져오기
            full_text = content_section.text.strip()
            data["content_raw"] = full_text

            data["content_paragraphs"] = paragraphs

            data["content_raw"] = "\n\n".join(paragraphs)

            return data
        try:
            article_data = parse_article_detail(driver, timeout=15)
        except TimeoutException:
            print("[WARN] 본문 로딩 실패, 스킵:", item["url"])
            continue
        except Exception as e:
            print("[ERROR] 예기치 못한 에러, 스킵:", item["url"], e)
            continue

        detail_results.append({
            "source" : item["source"],
            "externalId":item["externalId"],
            "url": item["url"],
            "title": article_data["title"],
            "reporter": article_data["author"],
            "publishedDate": article_data["posted_at"],
            "category": article_data["category"],
            "content": article_data["content_raw"],  # 나중에 요약 모델에 넣을 원문
            "thumbnailUrl": item["thumbnail_url"],
            "provider" : item["provider"]
        })
    driver.quit()


    # KoBERT 요약
    for article in detail_results:
        text = article.get("content", "")
        if not text:
            article["content"] = ""
            continue
        summary, _ = summarize_kobert(text, top_k=7)
        article["content"] = summary


    return detail_results




'''
def crawl_woowahan():
    SOURCE = "WOOWATECH" 
    url = 'https://techblog.woowahan.com/'
    HEADERS = {"User-Agent": "Mozilla/5.0"}
    provider = "woowahan"

    def parse_post_date(text: str) -> date:
        text = text.strip()
        # 기본 형식: Dec.02.2025
        return datetime.strptime(text, "%b.%d.%Y").date()
    # ───────── 개별 기사 페이지 파싱 함수 ─────────
    def parse_article(url: str) -> dict:
        """
        글 상세 페이지에 들어가서 추가 정보(본문, 태그 등)를 뽑는 예시.
        CSS selector는 실제 페이지 구조에 맞게 조정해야 함!
        """
        res = requests.get(url, headers=HEADERS)
        res.raise_for_status()
        soup = BeautifulSoup(res.text, "html.parser")

        # 1) 본문 내용 추출 (예시 selector)
        #   - 실제로는 F12로 보고 class 이름 확인해서 고쳐줘야 함
        content_el = (
            soup.select_one("div.post-content") or
            soup.select_one("div.entry-content") or
            soup.select_one("article")
        )
        content_text = content_el.get_text("\n", strip=True) if content_el else ""
        # 1) 작성자 + 상세 날짜
        author = None
        detail_date_raw = None
        detail_date = None

        author_box = soup.select_one("div.post-header-author")
        if author_box:
            span_texts = [
                s.get_text(strip=True)
                for s in author_box.find_all("span")
                if s.get_text(strip=True)
            ]
            if len(span_texts) >= 1:
                detail_date_raw = span_texts[0]
            if len(span_texts) >= 2:
                author = span_texts[1]

            # 날짜를 date 객체로 파싱 (실패하면 그냥 raw만 둠)
            if detail_date_raw:
                try:
                    detail_date = datetime.strptime(detail_date_raw, "%Y. %m. %d.").date()
                except ValueError:
                    detail_date = None
        category = None
        category_slug = None
        cat_el = soup.select_one("p.post-header-categories a.cat-tag")
        if cat_el:
            category = cat_el.get_text(strip=True)       # 예: "Infra"
            category_slug = cat_el.get("data-slug")      # 예: "infra"     
 

        return {
            "source" : SOURCE,
            "content": content_text,
            "author": author,
            "category": category,
            # "tags": tags,  # 필요하면 주석 해제 + 위 코드 활성화
        }
    
    # 1) 어제 날짜 문자열 만들기
    yesterday = date.today() - timedelta(days=1)# 수정사항!! 
    #print(yesterday)
    
    #target_date_str = yesterday.strftime("%Y. %m. %d.")  # "2025. 12. 02." 형태
    target_date_str=str(yesterday)
    #print(type(yesterday))
    res = requests.get(url, headers=HEADERS)
    res.raise_for_status()

    soup = BeautifulSoup(res.text, "html.parser")

    posts = []

    # 2) 모든 게시글 카드 찾기
    for item in soup.select("div.post-item"):
        time_tag = item.select_one("time.post-author-date")
        if not time_tag:
            continue

        # ex) "2025. 12. 02."
        date_text = time_tag.get_text(strip=True)
        if len(date_text) == 0:
            continue
        #print(date_text)
        post_date = str(parse_post_date(date_text))
        # 3) 날짜가 어제와 같은 카드만 통과

        if post_date != target_date_str:
            #print(f"post : {post_date}")
            #print(f"target :{target_date_str}")
            continue


        # 4) 제목과 링크 추출
        title_tag = item.select_one("h2.post-title")

        if not title_tag:
            continue

        title = title_tag.get_text(strip=True)


        # 제목을 감싸고 있는 <a> (기사 링크)
        link_tag = title_tag.find_parent("a")
        if not link_tag:
            continue

        url = link_tag["href"]

        article_info = parse_article(url) 
        #print(article_info)

        summary, summary_sents = summarize_kobert(article_info["content"], top_k=7)
        posts.append({
            "source" : SOURCE,
            "externalid": None,
            "provider" : provider,
            "publishedDate": yesterday,
            "title": title,
            "url": url,
            "category": article_info["category"],
            "content": summary,#summary,
            "reporter": article_info["author"],
            "thumbnailUrl": None, 
        })

    return posts
'''

if __name__ == "__main__":
    data = crawl_news()
    # data2 = crawl_woowahan()
    data_f = data # +data2
    df = pd.DataFrame(data_f)
    df.to_json(
        "news_output.json",
        orient="records",
        force_ascii=False,
        indent=2,
    )
    print("news_output.json 저장 완료!")