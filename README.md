## Docker 배포 가이드

1. 필요한 API 키와 설정값을 `.env` 파일에 기입합니다. (이 파일은 이미지에 포함되지 않고 컨테이너에만 주입됩니다.)
2. 최초 또는 의존성이 바뀔 때 `docker compose up --build -d` 를 실행해 이미지를 생성하고 백엔드를 기동합니다. 개발 중에는 `docker compose up --build` 로 포그라운드에서 로그를 확인할 수 있습니다.
3. 기본적으로 `9876` 포트가 열리며 `.env` 에 `FASTAPI_PORT=<포트번호>` 를 추가하면 외부로 노출되는 포트를 바꿀 수 있습니다.
4. 컨테이너 로그는 `docker compose logs -f material-backend` 로 확인하고, 중지할 때는 `docker compose down` 을 실행합니다.

### 수동 이미지 빌드 (선택)

```
docker build -t material-back .
docker run --env-file .env -p 9876:9876 material-back
```

docker compose 를 사용할 수 없는 환경에서 단독 컨테이너를 띄울 때 활용할 수 있습니다.
