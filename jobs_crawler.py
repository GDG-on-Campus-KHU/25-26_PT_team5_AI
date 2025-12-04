from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from selenium.common.exceptions import StaleElementReferenceException,TimeoutException, NoSuchElementException, ElementClickInterceptedException
import time
import pandas as pd


# source, externalid, content

url = "https://www.jobkorea.co.kr/recruit/joblist?menucode=local&localorder=1"


def safe_click_search(driver, timeout=10, retries=3):
    wait = WebDriverWait(driver, timeout)

    for attempt in range(retries):
        try:
            # 1) 버튼 다시 찾아오기 (매번 fresh element)
            btn = wait.until(
                EC.element_to_be_clickable((By.ID, "dev-btn-search"))
            )

            # 2) 화면 가운데로 스크롤
            driver.execute_script(
                "arguments[0].scrollIntoView({block: 'center'});", btn
            )
            time.sleep(0.3)

            # 3) 먼저 일반 click 시도
            try:
                btn.click()
            except ElementClickInterceptedException:
                # 4) 뭔가 가로막고 있으면 JS로 강제 클릭
                driver.execute_script("arguments[0].click();", btn)

            return True  # 성공하면 True

        except (StaleElementReferenceException, TimeoutException) as e:
            print(f"[WARN] 검색 버튼 클릭 재시도 ({attempt+1}/{retries}) :", e)
            time.sleep(1)

    print("[ERROR] dev-btn-search 클릭 실패")
    return False


def get_employment_type(driver, timeout=5):
    """
    상세 페이지에서 '고용형태' 값(예: '연수생/교육생', '정규직', '계약직' 등)을 반환.
    못 찾으면 빈 문자열 "" 반환.
    """
    try:
        wait = WebDriverWait(driver, timeout)

        # 1) '고용형태' 라벨이 있는 RecruitmentItem 블럭 찾기
        container = wait.until(
            EC.presence_of_element_located(
                (
                    By.XPATH,
                    "//div[@data-sentry-component='RecruitmentItem']"
                    "[.//span[normalize-space()='고용형태']]"
                )
            )
        )

        # 2) 그 안에서 실제 값(span) 찾기
        #    - 첫 번째 span은 '고용형태' 라벨이므로 제외
        #    - '고용형태'가 아닌 span 중 첫 번째 텍스트를 가져오기
        value_span = container.find_element(
            By.XPATH,
            ".//span[normalize-space()!='고용형태'][1]"
        )

        return value_span.text.strip()

    except (NoSuchElementException, TimeoutException):
        return ""

def get_qualification_experience(driver):
    """
    지원 자격 영역에서 '경력' 항목의 값을 가져온다.
    예:
      - '경력무관'
      - '신입'
      - '경력 (5년이상)'
    """
    try:
        # 1) '경력' 라벨을 가진 QualificationItem 블럭 찾기
        container = driver.find_element(
            By.XPATH,
            "//div[@data-sentry-component='QualificationItem']"
            "[.//span[normalize-space()='경력']]"
        )

        # 2) 안에서 파란 글씨(주요 값, theme-primary) 찾기
        main_span = container.find_element(
            By.XPATH,
            ".//span[@data-accent-color='theme-primary']"
        )
        main_text = main_span.text.strip()  # 예: '경력', '경력무관', '신입'

        # 3) 선택적으로 뒤에 붙는 회색 글씨 (예: '(5년이상)') 찾아서 합치기
        extra_text = ""
        try:
            extra_span = container.find_element(
                By.XPATH,
                ".//span[@data-accent-color='gray900']"
            )
            extra_text = extra_span.text.strip()  # 예: '(5년이상)'
        except NoSuchElementException:
            extra_text = ""

        # 4) 둘 다 있으면 "경력 (5년이상)" 형태로, 없으면 main만 반환
        if extra_text:
            return f"{main_text} {extra_text}"
        return main_text or None

    except NoSuchElementException:
        return None




def crawl_jobs():
    options = webdriver.ChromeOptions()
    options.add_argument("--start-maximized")
    driver = webdriver.Chrome(options=options)

    driver.get(url)
    wait = WebDriverWait(driver, 10)

    roles = ["백엔드개발자", "프론트엔드개발자", "AI/ML엔지니어", "AI/ML연구원", "데이터분석가"]
    results = []
    filtered_results = []
    old_first_card = None

    for role in roles:

        job_button = wait.until(
            EC.element_to_be_clickable(
                (By.XPATH, "//p[contains(@class, 'btn_tit') and contains(normalize-space(.), '직무')]")
            )
        )
        job_button.click()

        ai_label = wait.until(
            EC.element_to_be_clickable(
                (By.XPATH, "//label[@for='duty_step1_10031']")
            )
        )

        ai_label.click()

        # 1) 해당 직무 span을 찾고
        span = wait.until(
            EC.element_to_be_clickable(
                (By.XPATH, f"//span[contains(., '{role}')]")
            )
        )
        # 2) 클릭 가능한 부모(label/li)를 클릭
        #   상황에 따라 span 자체가 클릭 가능하면 그냥 span.click()도 됨
        driver.execute_script("arguments[0].click();", span)
        

            
        #3. "선택된 조건 검색하기" 버튼이 클릭 가능할 때까지 기다렸다가 클릭
        time.sleep(0.8)

        safe_click_search(driver, timeout=10, retries=3)

        # 4. 각 36개의 회사 정보 가져오기
        # 이름, 로고, 링크
        # 검색 후, 결과 리스트가 로딩될 때까지 기다림
        for retry in range(5):  # 최대 5번 재시도
                job_cards = wait.until(
                    EC.presence_of_all_elements_located(
                        (By.CSS_SELECTOR, "#TopHeadlineAGI ul.listBanner.listBanner_3xn:not(.listMajor) > li.devloopArea")
                    )
                )

                if not job_cards:
                    # 카드가 하나도 없으면 잠깐 기다렸다 재시도
                    time.sleep(1)
                    continue


                # 아직도 이전이랑 같으면 조금 기다렸다가 다시 시도
                time.sleep(1)


        print("찾은 공고 카드 개수:", len(job_cards))

        for li in job_cards:
            # 1) 회사 이름 & 회사 링크 (company 영역)
            company_a = li.find_element(By.CSS_SELECTOR, "div.company span.name a")
            company_name = company_a.text.strip() 
            #print(company_name)
            


            # 회사 상세(채용공고) 링크 (상대경로 → 절대경로로 보정)
            company_href = company_a.get_attribute("href")
            if company_href.startswith("/"):
                company_href = url + company_href

            # 2) 회사 로고 이미지
            logo_img = li.find_element(By.CSS_SELECTOR, "div.company span.logo img")
            logo_src = logo_img.get_attribute("src")  # 보통 //imgs.jobkorea... 형태

            # 프로토콜 없는 URL이면 https 붙여주기
            if logo_src.startswith("//"):
                logo_src = "https:" + logo_src

            # 3) 채용 내용 링크 (description 영역 a 태그)
            #    회사 링크랑 같은 경우가 많지만, 구조상 분리해두면 좋음
            desc_a = li.find_element(By.CSS_SELECTOR, "div.description a")
            detail_href = desc_a.get_attribute("href")
            if detail_href.startswith("/"):
                detail_href = url + detail_href

            # 4) (선택) 공고 제목/요약 텍스트도 같이 저장 가능
            summary_text = desc_a.text.strip()  # "폴라리스오피스 신입/경력직 채용 ..." 등

            # 5) (선택) D-Day / 마감일 "~12/10" 같은 것도 따로
            try:
                dday_span = desc_a.find_element(By.CSS_SELECTOR, "span.dday span.deadLine")
                dday_text = dday_span.text.strip()  # "~12/10" or "상시채용"
            except:
                dday_text = None

            results.append(
                {
                    "category": role,
                    "companyName": company_name,
                    "thumbnailUrl": logo_src,
                    "url": detail_href,
                    "summary": summary_text,
                    "dday": dday_text,
                }
            )  
        # 초기화 누르고 다시 
        reset_btn = driver.find_element(By.CSS_SELECTOR, "div.item_reset button[type='reset']")
        reset_btn.click()
        
    # 확인용
    for r in results[:3]:
        print(r) 

    filtered_results = []


    for r in results:
        detail_url = r['url']
        driver.get(detail_url)
        #time.sleep(2)
        # 공고 상세페이지가 로딩될 때까지 대기 (h1이 뜰 때까지)
        wait.until(EC.presence_of_element_located((By.TAG_NAME, "h1")))
        
        emp_type = get_employment_type(driver)
        if "연수생/교육생" in emp_type:
            print("연수생/교육생 공고라 스킵 :", detail_url)
            continue
        
        r["employment_type"] = emp_type


        # ---------------------------------
        # 1) 제목 (h1)
        # ---------------------------------
        try:
            title_el = driver.find_element(By.XPATH, "//h1")
            r["title"] = title_el.text.strip()
        except NoSuchElementException:
            r["title"] = None

        # ---------------------------------
        # 2) 시작일 (라벨: '시작일')
        #    <span>시작일</span> 이 있는 div의 형제 div 안의 span
        # ---------------------------------
        try:
            posted_el = driver.find_element(
                By.XPATH,
                "//span[normalize-space(text())='시작일']/parent::div/following-sibling::div/span"
            )
            r["postedDate"] = posted_el.text.strip()
        except NoSuchElementException:
            r["postedDate"] = None

        # ---------------------------------
        # 3) 마감일 (라벨: '마감일')
        # ---------------------------------
        try:
            deadline_el = driver.find_element(
                By.XPATH,
                "//span[normalize-space(text())='마감일']/parent::div/following-sibling::div/span"
            )
            r["deadline"] = deadline_el.text.strip()
        except NoSuchElementException:
            r["deadline"] = None

        # ---------------------------------
        # 4) 기술 스택 (라벨: '스킬')
        #    예: <span>스킬</span> 옆에 <span>JAVA</span>
        #    스킬이 여러 개면 span들이 여러 개일 수 있어서 모두 모아 join
        # ---------------------------------
        try:
            # '스킬' 이라는 라벨이 들어 있는 블록 전체를 먼저 찾고
            skill_block = driver.find_element(
                By.XPATH,
                "//span[normalize-space(text())='스킬']/ancestor::div[contains(@class,'Flex_display_flex')][1]"
            )
            # 그 안에서 실제 값(회색=gray900) span들만 추출
            skill_spans = skill_block.find_elements(
                By.XPATH,
                ".//span[@data-accent-color='gray900']"
            )
            tech_stack_list = [s.text.strip() for s in skill_spans if s.text.strip()]
            r["techStack"] = ", ".join(tech_stack_list) if tech_stack_list else None
        except NoSuchElementException:
            r["techStack"] = None

        # ---------------------------------
        # 5) 근무지 주소 (라벨: '근무지주소')
        #    네가 보여준 HTML의 주소 span을 타겟팅
        # ---------------------------------
        try:
            addr_el = driver.find_element(
                By.XPATH,
                "//span[normalize-space(text())='근무지주소']"
                "/ancestor::div[contains(@class,'Flex_display_flex')][1]"
                "//span[@data-accent-color='gray900'][1]"
            )
            r["location"] = addr_el.text.strip()
        except NoSuchElementException:
            r["location"] = None

        # ---------------------------------
        # 6) 경력 (exp_level)
        #    JobInfo 박스 안에서 '경력' 라벨 옆의 값 '신입', '경력 3년↑' 등
        # ---------------------------------
        r["expLevel"] = get_qualification_experience(driver)

        filtered_results.append(
            {
            "source" : "jobkorea",
            "externalid":None,
            "title" :r["title"],
            "companyName" : r["companyName"],
            "content" :None,
            "url" : r['url'],
            "postedDate" :r["postedDate"],
            "deadline" : r["deadline"],
            "category" :r["category"],
            "techStack" : r["techStack"],
            "location" :r["location"],
            "expLevel" : r["expLevel"],
            "thumbnailUrl" : r["thumbnailUrl"]
                            })
    
    for idx, item in enumerate(filtered_results,start=1):
        item["externalid"]= f"{idx:03d}"

    driver.quit()
    return filtered_results


if __name__ == "__main__":
    data = crawl_jobs()
    df = pd.DataFrame(data)
    df.to_json("job_output.json", orient="records", force_ascii=False, indent=2)
    print("job_output.json 저장 완료!")