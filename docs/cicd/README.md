# CI/CD 配置指南

## 概述

本文档详细描述了GraphRAG知识库系统的持续集成和持续部署(CI/CD)配置，包括代码质量检查、自动化测试、容器构建、安全扫描和部署策略。

## CI/CD架构

```mermaid
flowchart TD
    subgraph "代码仓库"
        GIT[Git Repository]
        BRANCH_MAIN[main分支]
        BRANCH_DEV[develop分支]
        BRANCH_FEATURE[feature分支]
    end
    
    subgraph "CI流水线"
        TRIGGER[触发器]
        LINT[代码检查]
        TEST_UNIT[单元测试]
        TEST_INTEGRATION[集成测试]
        BUILD[构建镜像]
        SECURITY[安全扫描]
        QUALITY[质量门禁]
    end
    
    subgraph "制品仓库"
        DOCKER_REGISTRY[Docker Registry]
        HELM_REPO[Helm Repository]
        ARTIFACT_STORE[制品存储]
    end
    
    subgraph "部署环境"
        ENV_DEV[开发环境]
        ENV_STAGING[测试环境]
        ENV_PROD[生产环境]
    end
    
    subgraph "监控反馈"
        MONITORING[监控系统]
        ALERTS[告警通知]
        ROLLBACK[回滚机制]
    end
    
    GIT --> TRIGGER
    TRIGGER --> LINT
    LINT --> TEST_UNIT
    TEST_UNIT --> TEST_INTEGRATION
    TEST_INTEGRATION --> BUILD
    BUILD --> SECURITY
    SECURITY --> QUALITY
    
    QUALITY --> DOCKER_REGISTRY
    QUALITY --> HELM_REPO
    QUALITY --> ARTIFACT_STORE
    
    BRANCH_FEATURE --> ENV_DEV
    BRANCH_DEV --> ENV_STAGING
    BRANCH_MAIN --> ENV_PROD
    
    ENV_PROD --> MONITORING
    MONITORING --> ALERTS
    ALERTS --> ROLLBACK
```

## GitLab CI/CD配置

### 主配置文件

```yaml
# .gitlab-ci.yml
include:
  - local: '.gitlab-ci/variables.yml'
  - local: '.gitlab-ci/rules.yml'
  - local: '.gitlab-ci/jobs/test.yml'
  - local: '.gitlab-ci/jobs/build.yml'
  - local: '.gitlab-ci/jobs/security.yml'
  - local: '.gitlab-ci/jobs/deploy.yml'

stages:
  - validate
  - test
  - build
  - security
  - deploy
  - post-deploy

# 全局变量
variables:
  DOCKER_DRIVER: overlay2
  DOCKER_TLS_CERTDIR: "/certs"
  REGISTRY: registry.example.com
  PROJECT_NAME: graphrag
  KUBECONFIG: /tmp/kubeconfig
  HELM_EXPERIMENTAL_OCI: 1

# 默认设置
default:
  image: python:3.11-slim
  before_script:
    - apt-get update -qq && apt-get install -y -qq git curl
    - pip install --upgrade pip

# 缓存配置
cache:
  key: ${CI_COMMIT_REF_SLUG}
  paths:
    - .pip-cache/
    - node_modules/
    - .pytest_cache/
```

### 变量配置

```yaml
# .gitlab-ci/variables.yml
variables:
  # Docker配置
  DOCKER_REGISTRY: "registry.example.com"
  DOCKER_NAMESPACE: "graphrag"
  
  # 应用版本
  APP_VERSION: "${CI_COMMIT_TAG:-${CI_COMMIT_SHORT_SHA}}"
  
  # 环境配置
  DEV_NAMESPACE: "graphrag-dev"
  STAGING_NAMESPACE: "graphrag-staging"
  PROD_NAMESPACE: "graphrag"
  
  # 测试配置
  POSTGRES_DB: "test_graphrag"
  POSTGRES_USER: "test"
  POSTGRES_PASSWORD: "test"
  REDIS_URL: "redis://redis:6379/0"
  
  # 质量门禁
  COVERAGE_THRESHOLD: "80"
  SECURITY_THRESHOLD: "HIGH"
  
  # 通知配置
  SLACK_WEBHOOK: "${SLACK_WEBHOOK_URL}"
  EMAIL_RECIPIENTS: "devops@example.com"
```

### 规则配置

```yaml
# .gitlab-ci/rules.yml
.rules:
  # 主分支规则
  main_branch: &main_branch_rules
    - if: '$CI_COMMIT_BRANCH == "main"'
      when: always
  
  # 开发分支规则
  develop_branch: &develop_branch_rules
    - if: '$CI_COMMIT_BRANCH == "develop"'
      when: always
  
  # 功能分支规则
  feature_branch: &feature_branch_rules
    - if: '$CI_COMMIT_BRANCH =~ /^feature\/.*/'
      when: always
  
  # 标签规则
  tag_rules: &tag_rules
    - if: '$CI_COMMIT_TAG'
      when: always
  
  # 合并请求规则
  merge_request: &merge_request_rules
    - if: '$CI_PIPELINE_SOURCE == "merge_request_event"'
      when: always
  
  # 手动触发规则
  manual_deploy: &manual_deploy_rules
    - if: '$CI_COMMIT_BRANCH == "main"'
      when: manual
      allow_failure: false
```

## 测试阶段配置

### 代码质量检查

```yaml
# .gitlab-ci/jobs/test.yml
# 代码格式检查
lint:code:
  stage: validate
  script:
    - pip install black isort flake8 mypy
    - black --check src/ tests/
    - isort --check-only src/ tests/
    - flake8 src/ tests/
    - mypy src/
  rules:
    - <<: *merge_request_rules
    - <<: *feature_branch_rules
    - <<: *develop_branch_rules
    - <<: *main_branch_rules

# 依赖安全检查
lint:dependencies:
  stage: validate
  script:
    - pip install safety pip-audit
    - safety check --json --output safety-report.json
    - pip-audit --format=json --output=pip-audit-report.json
  artifacts:
    reports:
      security:
        - safety-report.json
        - pip-audit-report.json
    expire_in: 1 week
  allow_failure: true
  rules:
    - <<: *merge_request_rules
    - <<: *develop_branch_rules
    - <<: *main_branch_rules

# 单元测试
test:unit:
  stage: test
  services:
    - name: postgres:13
      alias: postgres
    - name: redis:6
      alias: redis
  variables:
    POSTGRES_DB: $POSTGRES_DB
    POSTGRES_USER: $POSTGRES_USER
    POSTGRES_PASSWORD: $POSTGRES_PASSWORD
  script:
    - pip install -r requirements.txt
    - pip install pytest pytest-cov pytest-xdist pytest-mock
    - pytest tests/unit/ 
        --cov=src/ 
        --cov-report=xml 
        --cov-report=html 
        --cov-report=term
        --junit-xml=junit-report.xml
        -n auto
  coverage: '/TOTAL.+?(\d+\%)$/'
  artifacts:
    reports:
      junit: junit-report.xml
      coverage_report:
        coverage_format: cobertura
        path: coverage.xml
    paths:
      - htmlcov/
    expire_in: 1 week
  rules:
    - <<: *merge_request_rules
    - <<: *feature_branch_rules
    - <<: *develop_branch_rules
    - <<: *main_branch_rules

# 集成测试
test:integration:
  stage: test
  image: docker/compose:latest
  services:
    - docker:dind
  variables:
    DOCKER_HOST: tcp://docker:2376
    DOCKER_TLS_CERTDIR: "/certs"
    DOCKER_TLS_VERIFY: 1
    DOCKER_CERT_PATH: "$DOCKER_TLS_CERTDIR/client"
  script:
    - docker-compose -f docker-compose.test.yml build
    - docker-compose -f docker-compose.test.yml up --abort-on-container-exit
    - docker-compose -f docker-compose.test.yml down -v
  artifacts:
    reports:
      junit: test-results/integration-report.xml
    expire_in: 1 week
  rules:
    - <<: *develop_branch_rules
    - <<: *main_branch_rules

# API测试
test:api:
  stage: test
  image: postman/newman:latest
  script:
    - newman run tests/api/GraphRAG-API.postman_collection.json 
        --environment tests/api/test-environment.json
        --reporters junit,cli
        --reporter-junit-export newman-report.xml
  artifacts:
    reports:
      junit: newman-report.xml
    expire_in: 1 week
  rules:
    - <<: *develop_branch_rules
    - <<: *main_branch_rules

# 性能测试
test:performance:
  stage: test
  image: grafana/k6:latest
  script:
    - k6 run --out junit=performance-report.xml tests/performance/load-test.js
  artifacts:
    reports:
      junit: performance-report.xml
    expire_in: 1 week
  allow_failure: true
  rules:
    - <<: *main_branch_rules
    - when: manual
```

## 构建阶段配置

### 容器镜像构建

```yaml
# .gitlab-ci/jobs/build.yml
# API服务构建
build:api:
  stage: build
  image: docker:latest
  services:
    - docker:dind
  variables:
    IMAGE_TAG: "${DOCKER_REGISTRY}/${DOCKER_NAMESPACE}/api:${APP_VERSION}"
    LATEST_TAG: "${DOCKER_REGISTRY}/${DOCKER_NAMESPACE}/api:latest"
  before_script:
    - echo $CI_REGISTRY_PASSWORD | docker login -u $CI_REGISTRY_USER --password-stdin $DOCKER_REGISTRY
  script:
    # 构建多阶段镜像
    - docker build 
        --target production
        --build-arg BUILD_DATE=$(date -u +'%Y-%m-%dT%H:%M:%SZ')
        --build-arg VCS_REF=$CI_COMMIT_SHA
        --build-arg VERSION=$APP_VERSION
        -t $IMAGE_TAG
        -t $LATEST_TAG
        -f docker/Dockerfile.api .
    
    # 推送镜像
    - docker push $IMAGE_TAG
    - docker push $LATEST_TAG
    
    # 生成镜像清单
    - docker inspect $IMAGE_TAG > api-image-manifest.json
  artifacts:
    paths:
      - api-image-manifest.json
    expire_in: 1 week
  rules:
    - <<: *develop_branch_rules
    - <<: *main_branch_rules

# Worker服务构建
build:worker:
  stage: build
  image: docker:latest
  services:
    - docker:dind
  variables:
    IMAGE_TAG: "${DOCKER_REGISTRY}/${DOCKER_NAMESPACE}/worker:${APP_VERSION}"
    LATEST_TAG: "${DOCKER_REGISTRY}/${DOCKER_NAMESPACE}/worker:latest"
  before_script:
    - echo $CI_REGISTRY_PASSWORD | docker login -u $CI_REGISTRY_USER --password-stdin $DOCKER_REGISTRY
  script:
    - docker build 
        --target production
        --build-arg BUILD_DATE=$(date -u +'%Y-%m-%dT%H:%M:%SZ')
        --build-arg VCS_REF=$CI_COMMIT_SHA
        --build-arg VERSION=$APP_VERSION
        -t $IMAGE_TAG
        -t $LATEST_TAG
        -f docker/Dockerfile.worker .
    - docker push $IMAGE_TAG
    - docker push $LATEST_TAG
    - docker inspect $IMAGE_TAG > worker-image-manifest.json
  artifacts:
    paths:
      - worker-image-manifest.json
    expire_in: 1 week
  rules:
    - <<: *develop_branch_rules
    - <<: *main_branch_rules

# Web界面构建
build:web:
  stage: build
  image: node:18-alpine
  cache:
    key: ${CI_COMMIT_REF_SLUG}-node
    paths:
      - web/node_modules/
  script:
    - cd web/
    - npm ci --cache .npm --prefer-offline
    - npm run build
    - npm run test:unit
  artifacts:
    paths:
      - web/dist/
    expire_in: 1 week
  rules:
    - <<: *develop_branch_rules
    - <<: *main_branch_rules

# Helm Chart打包
build:helm:
  stage: build
  image: alpine/helm:latest
  script:
    - helm dependency update helm/graphrag/
    - helm package helm/graphrag/ --version $APP_VERSION --app-version $APP_VERSION
    - helm push graphrag-${APP_VERSION}.tgz oci://${DOCKER_REGISTRY}/helm
  artifacts:
    paths:
      - graphrag-*.tgz
    expire_in: 1 week
  rules:
    - <<: *main_branch_rules
    - <<: *tag_rules
```

## 安全扫描配置

### 容器安全扫描

```yaml
# .gitlab-ci/jobs/security.yml
# 容器漏洞扫描
security:container:
  stage: security
  image: aquasec/trivy:latest
  variables:
    API_IMAGE: "${DOCKER_REGISTRY}/${DOCKER_NAMESPACE}/api:${APP_VERSION}"
    WORKER_IMAGE: "${DOCKER_REGISTRY}/${DOCKER_NAMESPACE}/worker:${APP_VERSION}"
  script:
    # 扫描API镜像
    - trivy image 
        --format json 
        --output api-security-report.json 
        --severity HIGH,CRITICAL 
        $API_IMAGE
    
    # 扫描Worker镜像
    - trivy image 
        --format json 
        --output worker-security-report.json 
        --severity HIGH,CRITICAL 
        $WORKER_IMAGE
    
    # 生成汇总报告
    - trivy image --format table $API_IMAGE
    - trivy image --format table $WORKER_IMAGE
  artifacts:
    reports:
      security:
        - api-security-report.json
        - worker-security-report.json
    expire_in: 1 week
  allow_failure: true
  rules:
    - <<: *develop_branch_rules
    - <<: *main_branch_rules

# 代码安全扫描
security:sast:
  stage: security
  image: python:3.11
  script:
    - pip install bandit semgrep
    
    # Bandit扫描
    - bandit -r src/ -f json -o bandit-report.json
    
    # Semgrep扫描
    - semgrep --config=auto --json --output=semgrep-report.json src/
    
    # 生成汇总报告
    - bandit -r src/ -f txt
  artifacts:
    reports:
      security:
        - bandit-report.json
        - semgrep-report.json
    expire_in: 1 week
  allow_failure: true
  rules:
    - <<: *merge_request_rules
    - <<: *develop_branch_rules
    - <<: *main_branch_rules

# 依赖漏洞扫描
security:dependencies:
  stage: security
  image: python:3.11
  script:
    - pip install safety pip-audit cyclonedx-bom
    
    # 生成SBOM
    - cyclonedx-py -o sbom.json
    
    # Safety扫描
    - safety check --json --output safety-report.json
    
    # pip-audit扫描
    - pip-audit --format=json --output=pip-audit-report.json
  artifacts:
    reports:
      security:
        - safety-report.json
        - pip-audit-report.json
      sbom: sbom.json
    expire_in: 1 week
  allow_failure: true
  rules:
    - <<: *develop_branch_rules
    - <<: *main_branch_rules

# 密钥扫描
security:secrets:
  stage: security
  image: trufflesecurity/trufflehog:latest
  script:
    - trufflehog git file://. --json > secrets-report.json
    - trufflehog git file://. --only-verified
  artifacts:
    reports:
      security: secrets-report.json
    expire_in: 1 week
  allow_failure: true
  rules:
    - <<: *merge_request_rules
    - <<: *develop_branch_rules
    - <<: *main_branch_rules
```

## 部署阶段配置

### 环境部署

```yaml
# .gitlab-ci/jobs/deploy.yml
# 开发环境部署
deploy:dev:
  stage: deploy
  image: bitnami/kubectl:latest
  environment:
    name: development
    url: https://dev-api.graphrag.example.com
    deployment_tier: development
  variables:
    NAMESPACE: $DEV_NAMESPACE
    VALUES_FILE: "helm/values-dev.yaml"
  before_script:
    - echo $KUBE_CONFIG_DEV | base64 -d > $KUBECONFIG
    - helm repo add bitnami https://charts.bitnami.com/bitnami
    - helm repo update
  script:
    - helm upgrade --install graphrag-dev helm/graphrag/
        --namespace $NAMESPACE
        --create-namespace
        --values $VALUES_FILE
        --set image.tag=$APP_VERSION
        --set environment=development
        --wait --timeout=10m
    
    # 健康检查
    - kubectl rollout status deployment/graphrag-api -n $NAMESPACE
    - kubectl rollout status deployment/graphrag-worker -n $NAMESPACE
    
    # 运行烟雾测试
    - kubectl run smoke-test --rm -i --restart=Never 
        --image=curlimages/curl 
        -- curl -f http://graphrag-api-service.$NAMESPACE.svc.cluster.local/health
  rules:
    - <<: *feature_branch_rules
    - <<: *develop_branch_rules

# 测试环境部署
deploy:staging:
  stage: deploy
  image: bitnami/kubectl:latest
  environment:
    name: staging
    url: https://staging-api.graphrag.example.com
    deployment_tier: staging
  variables:
    NAMESPACE: $STAGING_NAMESPACE
    VALUES_FILE: "helm/values-staging.yaml"
  before_script:
    - echo $KUBE_CONFIG_STAGING | base64 -d > $KUBECONFIG
    - helm repo add bitnami https://charts.bitnami.com/bitnami
    - helm repo update
  script:
    - helm upgrade --install graphrag-staging helm/graphrag/
        --namespace $NAMESPACE
        --create-namespace
        --values $VALUES_FILE
        --set image.tag=$APP_VERSION
        --set environment=staging
        --wait --timeout=15m
    
    # 健康检查
    - kubectl rollout status deployment/graphrag-api -n $NAMESPACE
    - kubectl rollout status deployment/graphrag-worker -n $NAMESPACE
    
    # 集成测试
    - kubectl apply -f k8s/test-job.yaml -n $NAMESPACE
    - kubectl wait --for=condition=complete job/integration-test -n $NAMESPACE --timeout=300s
  after_script:
    - kubectl delete job/integration-test -n $NAMESPACE --ignore-not-found
  rules:
    - <<: *develop_branch_rules

# 生产环境部署
deploy:production:
  stage: deploy
  image: bitnami/kubectl:latest
  environment:
    name: production
    url: https://api.graphrag.example.com
    deployment_tier: production
  variables:
    NAMESPACE: $PROD_NAMESPACE
    VALUES_FILE: "helm/values-production.yaml"
  before_script:
    - echo $KUBE_CONFIG_PROD | base64 -d > $KUBECONFIG
    - helm repo add bitnami https://charts.bitnami.com/bitnami
    - helm repo update
  script:
    # 备份当前版本
    - kubectl get deployment graphrag-api -n $NAMESPACE -o yaml > backup-api-deployment.yaml
    - kubectl get deployment graphrag-worker -n $NAMESPACE -o yaml > backup-worker-deployment.yaml
    
    # 部署新版本
    - helm upgrade --install graphrag helm/graphrag/
        --namespace $NAMESPACE
        --create-namespace
        --values $VALUES_FILE
        --set image.tag=$APP_VERSION
        --set environment=production
        --wait --timeout=20m
    
    # 健康检查
    - kubectl rollout status deployment/graphrag-api -n $NAMESPACE
    - kubectl rollout status deployment/graphrag-worker -n $NAMESPACE
    
    # 生产验证测试
    - kubectl apply -f k8s/production-test-job.yaml -n $NAMESPACE
    - kubectl wait --for=condition=complete job/production-test -n $NAMESPACE --timeout=600s
  after_script:
    - kubectl delete job/production-test -n $NAMESPACE --ignore-not-found
  artifacts:
    paths:
      - backup-*.yaml
    expire_in: 1 week
  rules:
    - <<: *manual_deploy_rules

# 回滚部署
rollback:production:
  stage: deploy
  image: bitnami/kubectl:latest
  environment:
    name: production
    url: https://api.graphrag.example.com
    deployment_tier: production
  variables:
    NAMESPACE: $PROD_NAMESPACE
  before_script:
    - echo $KUBE_CONFIG_PROD | base64 -d > $KUBECONFIG
  script:
    - helm rollback graphrag -n $NAMESPACE
    - kubectl rollout status deployment/graphrag-api -n $NAMESPACE
    - kubectl rollout status deployment/graphrag-worker -n $NAMESPACE
  when: manual
  rules:
    - <<: *main_branch_rules
```

## 部署后验证

### 自动化测试

```yaml
# 部署后验证
post-deploy:verify:
  stage: post-deploy
  image: curlimages/curl
  variables:
    API_URL: "${CI_ENVIRONMENT_URL}"
  script:
    # 健康检查
    - curl -f $API_URL/health
    
    # API功能测试
    - curl -f -X POST $API_URL/api/v1/query 
        -H "Content-Type: application/json"
        -d '{"query": "test", "limit": 1}'
    
    # 性能基准测试
    - for i in {1..10}; do
        time curl -s $API_URL/health > /dev/null
      done
  rules:
    - <<: *develop_branch_rules
    - <<: *main_branch_rules

# 监控检查
post-deploy:monitoring:
  stage: post-deploy
  image: prom/prometheus:latest
  script:
    # 检查Prometheus指标
    - promtool query instant 'up{job="graphrag-api"}' --server=$PROMETHEUS_URL
    - promtool query instant 'rate(http_requests_total[5m])' --server=$PROMETHEUS_URL
  allow_failure: true
  rules:
    - <<: *main_branch_rules
```

## 通知配置

### Slack通知

```yaml
# 成功通知
notify:success:
  stage: post-deploy
  image: curlimages/curl
  script:
    - |
      curl -X POST -H 'Content-type: application/json' \
        --data "{
          \"text\": \"✅ GraphRAG部署成功\",
          \"attachments\": [{
            \"color\": \"good\",
            \"fields\": [
              {\"title\": \"环境\", \"value\": \"$CI_ENVIRONMENT_NAME\", \"short\": true},
              {\"title\": \"版本\", \"value\": \"$APP_VERSION\", \"short\": true},
              {\"title\": \"分支\", \"value\": \"$CI_COMMIT_REF_NAME\", \"short\": true},
              {\"title\": \"提交\", \"value\": \"$CI_COMMIT_SHORT_SHA\", \"short\": true}
            ]
          }]
        }" \
        $SLACK_WEBHOOK
  when: on_success
  rules:
    - <<: *main_branch_rules

# 失败通知
notify:failure:
  stage: post-deploy
  image: curlimages/curl
  script:
    - |
      curl -X POST -H 'Content-type: application/json' \
        --data "{
          \"text\": \"❌ GraphRAG部署失败\",
          \"attachments\": [{
            \"color\": \"danger\",
            \"fields\": [
              {\"title\": \"环境\", \"value\": \"$CI_ENVIRONMENT_NAME\", \"short\": true},
              {\"title\": \"版本\", \"value\": \"$APP_VERSION\", \"short\": true},
              {\"title\": \"分支\", \"value\": \"$CI_COMMIT_REF_NAME\", \"short\": true},
              {\"title\": \"流水线\", \"value\": \"$CI_PIPELINE_URL\", \"short\": false}
            ]
          }]
        }" \
        $SLACK_WEBHOOK
  when: on_failure
  rules:
    - <<: *main_branch_rules
```

## GitHub Actions配置

### 主工作流

```yaml
# .github/workflows/ci-cd.yml
name: CI/CD Pipeline

on:
  push:
    branches: [ main, develop ]
  pull_request:
    branches: [ main, develop ]
  release:
    types: [ published ]

env:
  REGISTRY: ghcr.io
  IMAGE_NAME: ${{ github.repository }}

jobs:
  # 代码质量检查
  lint:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
      
      - name: Set up Python
        uses: actions/setup-python@v4
        with:
          python-version: '3.11'
          
      - name: Install dependencies
        run: |
          python -m pip install --upgrade pip
          pip install black isort flake8 mypy
          
      - name: Run linting
        run: |
          black --check src/ tests/
          isort --check-only src/ tests/
          flake8 src/ tests/
          mypy src/

  # 单元测试
  test:
    runs-on: ubuntu-latest
    services:
      postgres:
        image: postgres:13
        env:
          POSTGRES_PASSWORD: postgres
          POSTGRES_DB: test_graphrag
        options: >-
          --health-cmd pg_isready
          --health-interval 10s
          --health-timeout 5s
          --health-retries 5
        ports:
          - 5432:5432
          
      redis:
        image: redis:6
        options: >-
          --health-cmd "redis-cli ping"
          --health-interval 10s
          --health-timeout 5s
          --health-retries 5
        ports:
          - 6379:6379
    
    steps:
      - uses: actions/checkout@v4
      
      - name: Set up Python
        uses: actions/setup-python@v4
        with:
          python-version: '3.11'
          
      - name: Install dependencies
        run: |
          python -m pip install --upgrade pip
          pip install -r requirements.txt
          pip install pytest pytest-cov
          
      - name: Run tests
        env:
          DATABASE_URL: postgresql://postgres:postgres@localhost:5432/test_graphrag
          REDIS_URL: redis://localhost:6379/0
        run: |
          pytest tests/ --cov=src/ --cov-report=xml
          
      - name: Upload coverage to Codecov
        uses: codecov/codecov-action@v3
        with:
          file: ./coverage.xml

  # 构建镜像
  build:
    needs: [lint, test]
    runs-on: ubuntu-latest
    permissions:
      contents: read
      packages: write
    
    steps:
      - uses: actions/checkout@v4
      
      - name: Log in to Container Registry
        uses: docker/login-action@v3
        with:
          registry: ${{ env.REGISTRY }}
          username: ${{ github.actor }}
          password: ${{ secrets.GITHUB_TOKEN }}
          
      - name: Extract metadata
        id: meta
        uses: docker/metadata-action@v5
        with:
          images: ${{ env.REGISTRY }}/${{ env.IMAGE_NAME }}
          tags: |
            type=ref,event=branch
            type=ref,event=pr
            type=semver,pattern={{version}}
            type=semver,pattern={{major}}.{{minor}}
            
      - name: Build and push Docker image
        uses: docker/build-push-action@v5
        with:
          context: .
          file: ./docker/Dockerfile.api
          push: true
          tags: ${{ steps.meta.outputs.tags }}
          labels: ${{ steps.meta.outputs.labels }}

  # 安全扫描
  security:
    needs: [build]
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
      
      - name: Run Trivy vulnerability scanner
        uses: aquasecurity/trivy-action@master
        with:
          image-ref: ${{ env.REGISTRY }}/${{ env.IMAGE_NAME }}:${{ github.sha }}
          format: 'sarif'
          output: 'trivy-results.sarif'
          
      - name: Upload Trivy scan results to GitHub Security tab
        uses: github/codeql-action/upload-sarif@v2
        with:
          sarif_file: 'trivy-results.sarif'

  # 部署到开发环境
  deploy-dev:
    needs: [build, security]
    runs-on: ubuntu-latest
    if: github.ref == 'refs/heads/develop'
    environment: development
    
    steps:
      - uses: actions/checkout@v4
      
      - name: Configure kubectl
        uses: azure/k8s-set-context@v3
        with:
          method: kubeconfig
          kubeconfig: ${{ secrets.KUBE_CONFIG_DEV }}
          
      - name: Deploy to development
        run: |
          helm upgrade --install graphrag-dev helm/graphrag/ \
            --namespace graphrag-dev \
            --create-namespace \
            --set image.tag=${{ github.sha }} \
            --set environment=development \
            --values helm/values-dev.yaml

  # 部署到生产环境
  deploy-prod:
    needs: [build, security]
    runs-on: ubuntu-latest
    if: github.ref == 'refs/heads/main'
    environment: production
    
    steps:
      - uses: actions/checkout@v4
      
      - name: Configure kubectl
        uses: azure/k8s-set-context@v3
        with:
          method: kubeconfig
          kubeconfig: ${{ secrets.KUBE_CONFIG_PROD }}
          
      - name: Deploy to production
        run: |
          helm upgrade --install graphrag helm/graphrag/ \
            --namespace graphrag \
            --create-namespace \
            --set image.tag=${{ github.sha }} \
            --set environment=production \
            --values helm/values-production.yaml \
            --wait --timeout=20m
```

## 质量门禁配置

### SonarQube集成

```yaml
# sonar-project.properties
sonar.projectKey=graphrag
sonar.projectName=GraphRAG Knowledge Base
sonar.projectVersion=1.0

# 源代码配置
sonar.sources=src/
sonar.tests=tests/
sonar.python.coverage.reportPaths=coverage.xml

# 质量门禁
sonar.qualitygate.wait=true

# 排除文件
sonar.exclusions=**/*_pb2.py,**/migrations/**,**/venv/**

# 代码重复检测
sonar.cpd.python.minimumtokens=50

# 测试覆盖率
sonar.coverage.exclusions=tests/**,**/__init__.py
```

### 质量检查作业

```yaml
# .gitlab-ci/jobs/quality.yml
quality:sonarqube:
  stage: test
  image: sonarsource/sonar-scanner-cli:latest
  variables:
    SONAR_USER_HOME: "${CI_PROJECT_DIR}/.sonar"
    GIT_DEPTH: "0"
  cache:
    key: "${CI_JOB_NAME}"
    paths:
      - .sonar/cache
  script:
    - sonar-scanner
  allow_failure: true
  rules:
    - <<: *merge_request_rules
    - <<: *main_branch_rules

quality:gate:
  stage: test
  image: python:3.11
  script:
    # 检查测试覆盖率
    - |
      COVERAGE=$(grep -o 'TOTAL.*[0-9]\+%' coverage-report.txt | grep -o '[0-9]\+')
      if [ "$COVERAGE" -lt "$COVERAGE_THRESHOLD" ]; then
        echo "Coverage $COVERAGE% is below threshold $COVERAGE_THRESHOLD%"
        exit 1
      fi
    
    # 检查安全扫描结果
    - |
      HIGH_VULNS=$(jq '.Results[].Vulnerabilities[] | select(.Severity=="HIGH" or .Severity=="CRITICAL")' trivy-report.json | wc -l)
      if [ "$HIGH_VULNS" -gt "0" ]; then
        echo "Found $HIGH_VULNS high/critical vulnerabilities"
        exit 1
      fi
  dependencies:
    - test:unit
    - security:container
  rules:
    - <<: *main_branch_rules
```

## 监控和告警

### 部署监控

```yaml
# 部署监控作业
monitor:deployment:
  stage: post-deploy
  image: prom/prometheus:latest
  script:
    # 检查服务可用性
    - |
      for i in {1..30}; do
        if promtool query instant 'up{job="graphrag-api"}' --server=$PROMETHEUS_URL | grep -q "1"; then
          echo "Service is up"
          break
        fi
        echo "Waiting for service to be ready..."
        sleep 10
      done
    
    # 检查错误率
    - |
      ERROR_RATE=$(promtool query instant 'rate(http_requests_total{status=~"5.."}[5m]) / rate(http_requests_total[5m])' --server=$PROMETHEUS_URL)
      if (( $(echo "$ERROR_RATE > 0.01" | bc -l) )); then
        echo "Error rate too high: $ERROR_RATE"
        exit 1
      fi
  rules:
    - <<: *main_branch_rules
```

### 告警配置

```yaml
# alerting-rules.yml
groups:
- name: deployment.rules
  rules:
  - alert: DeploymentFailed
    expr: kube_deployment_status_replicas_available / kube_deployment_spec_replicas < 0.8
    for: 5m
    labels:
      severity: critical
    annotations:
      summary: "Deployment {{ $labels.deployment }} has insufficient replicas"
      
  - alert: HighErrorRate
    expr: rate(http_requests_total{status=~"5.."}[5m]) / rate(http_requests_total[5m]) > 0.05
    for: 2m
    labels:
      severity: warning
    annotations:
      summary: "High error rate detected"
      
  - alert: SlowResponse
    expr: histogram_quantile(0.95, rate(http_request_duration_seconds_bucket[5m])) > 2
    for: 5m
    labels:
      severity: warning
    annotations:
      summary: "Slow response time detected"
```

## 最佳实践

### 1. 分支策略
- **main分支**: 生产环境代码，只接受来自develop的合并
- **develop分支**: 开发环境代码，功能集成分支
- **feature分支**: 功能开发分支，从develop创建
- **hotfix分支**: 紧急修复分支，从main创建

### 2. 版本管理
- 使用语义化版本控制 (SemVer)
- 自动生成版本号和变更日志
- 标签触发生产部署

### 3. 安全实践
- 密钥管理使用专用工具 (Vault, K8s Secrets)
- 镜像签名和验证
- 最小权限原则
- 定期安全扫描

### 4. 性能优化
- 并行执行测试和构建
- 缓存依赖和构建产物
- 增量构建和测试
- 智能触发条件

### 5. 监控和反馈
- 实时部署状态监控
- 自动回滚机制
- 详细的日志和指标
- 及时的通知和告警

## 故障排查

### 常见问题

1. **构建失败**
   ```bash
   # 检查构建日志
   gitlab-ci-multi-runner logs
   
   # 本地复现构建
   docker run --rm -v $(pwd):/app -w /app python:3.11 bash -c "pip install -r requirements.txt && python -m pytest"
   ```

2. **部署失败**
   ```bash
   # 检查Kubernetes状态
   kubectl get pods -n graphrag
   kubectl describe pod <pod-name> -n graphrag
   kubectl logs <pod-name> -n graphrag
   
   # 检查Helm部署
   helm status graphrag -n graphrag
   helm history graphrag -n graphrag
   ```

3. **测试失败**
   ```bash
   # 本地运行测试
   pytest tests/ -v --tb=short
   
   # 检查测试环境
   docker-compose -f docker-compose.test.yml up
   ```

### 回滚策略

```bash
# Helm回滚
helm rollback graphrag <revision> -n graphrag

# Kubernetes回滚
kubectl rollout undo deployment/graphrag-api -n graphrag

# 数据库回滚 (需要备份)
pg_restore -h localhost -U postgres -d graphrag backup.sql
```

## 总结

本CI/CD配置提供了完整的自动化流水线，包括：

- 代码质量检查和测试
- 安全扫描和漏洞检测
- 自动化构建和部署
- 监控和告警机制
- 回滚和故障恢复

通过这套配置，可以确保GraphRAG系统的高质量交付和稳定运行。