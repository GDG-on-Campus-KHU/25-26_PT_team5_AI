# 1. 파이썬 버전
FROM python:3.10-slim

# 2. 작업 폴더 설정 (컨테이너 내부의 루트 폴더)
WORKDIR /app

# 3. requirements.txt 복사 및 설치
# (requirements.txt는 Dockerfile과 같은 위치에 있으므로 그대로 복사)
COPY requirement.txt .
RUN pip install --no-cache-dir -r requirement.txt

# 4. 소스 코드 전체 복사
# (이때 crawling 폴더도 /app/crawling 으로 복사됨)
COPY . .

# 5. [중요] 실행할 파일이 있는 폴더로 이동!
# 이제부터 명령어를 실행할 위치를 'crawling' 폴더 내부로 변경합니다.
WORKDIR /app/crawling

# 6. 서버 실행
# 위치를 안으로 옮겼으므로 "app:app"으로 바로 실행 가능합니다.
CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8080"]

# Chrome 실행을 위한 필수 패키지 설치
RUN apt-get update && apt-get install -y \
    wget \
    gnupg \
    unzip \
    libnss3 \
    libfontconfig1 \
    libxi6 \
    libxcursor1 \
    libxss1 \
    libxcomposite1 \
    libasound2 \
    libxdamage1 \
    libxtst6 \
    libatk1.0-0 \
    libgtk-3-0 \
    fonts-liberation \
    xdg-utils \
    --no-install-recommends