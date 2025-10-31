# GraphRAG CI/CD 测试集成指南

本指南详细说明了如何在 GraphRAG 项目中设置和配置持续集成/持续部署（CI/CD）流水线，确保代码质量和自动化测试。

## 📚 目录

- [概述](#概述)
- [GitHub Actions](#github-actions)
- [测试策略](#测试策略)
- [环境配置](#环境配置)
- [工作流程](#工作流程)
- [质量门禁](#质量门禁)
- [部署流程](#部署流程)
- [监控和报告](#监控和报告)
- [故障排除](#故障排除)

## 🎯 概述

GraphRAG 项目采用多阶段的 CI/CD 流水线，确保：

- **代码质量**: 自动化代码检查和格式化
- **测试覆盖**: 全面的测试执行和覆盖率报告
- **安全扫描**: 依赖漏洞和代码安全检查
- **性能监控**: 性能回归检测
- **自动部署**: 安全可靠的部署流程

### CI/CD 架构

```mermaid
graph LR
    A[代码提交] --> B[代码检查]
    B --> C[单元测试]
    C --> D[集成测试]
    D --> E[安全扫描]
    E --> F[性能测试]
    F --> G[构建镜像]
    G --> H[部署测试环境]
    H --> I[E2E测试]
    I --> J[部署生产环境]
```

## 🔄 GitHub Actions

### 主工作流配置

创建 `.github/workflows/ci.yml`:

```yaml
name: CI/CD Pipeline

on:
  push:
    branches: [ main, develop ]
  pull_request:
    branches: [ main, develop ]
  schedule:
    # 每天凌晨2点运行完整测试
    - cron: '0 2 * * *'

env:
  PYTHON_VERSION: '3.11'
  NODE_VERSION: '18'

jobs:
  # 代码质量检查
  code-quality:
    name: Code Quality
    runs-on: ubuntu-latest
    steps:
      - name: Checkout code
        uses: actions/checkout@v4
        
      - name: Set up Python
        uses: actions/setup-python@v4
        with:
          python-version: ${{ env.PYTHON_VERSION }}
          
      - name: Cache dependencies
        uses: actions/cache@v3
        with:
          path: ~/.cache/pip
          key: ${{ runner.os }}-pip-${{ hashFiles('**/requirements*.txt') }}
          
      - name: Install dependencies
        run: |
          python -m pip install --upgrade pip
          pip install -r requirements-dev.txt
          
      - name: Code formatting check
        run: |
          black --check app/ tests/
          isort --check-only app/ tests/
          
      - name: Linting
        run: |
          flake8 app/ tests/
          pylint app/
          
      - name: Type checking
        run: |
          mypy app/
          
      - name: Security scan
        run: |
          bandit -r app/
          safety check

  # 单元测试
  unit-tests:
    name: Unit Tests
    runs-on: ubuntu-latest
    needs: code-quality
    strategy:
      matrix:
        python-version: ['3.10', '3.11', '3.12']
    
    steps:
      - name: Checkout code
        uses: actions/checkout@v4
        
      - name: Set up Python ${{ matrix.python-version }}
        uses: actions/setup-python@v4
        with:
          python-version: ${{ matrix.python-version }}
          
      - name: Install dependencies
        run: |
          python -m pip install --upgrade pip
          pip install -r requirements-test.txt
          
      - name: Run unit tests
        run: |
          pytest tests/unit/ \
            --cov=app \
            --cov-report=xml \
            --cov-report=html \
            --junitxml=test-results.xml \
            -v
            
      - name: Upload coverage to Codecov
        uses: codecov/codecov-action@v3
        with:
          file: ./coverage.xml
          flags: unittests
          name: codecov-umbrella
          
      - name: Upload test results
        uses: actions/upload-artifact@v3
        if: always()
        with:
          name: test-results-${{ matrix.python-version }}
          path: |
            test-results.xml
            htmlcov/

  # 集成测试
  integration-tests:
    name: Integration Tests
    runs-on: ubuntu-latest
    needs: unit-tests
    
    services:
      postgres:
        image: postgres:15
        env:
          POSTGRES_PASSWORD: test_password
          POSTGRES_USER: test_user
          POSTGRES_DB: graphrag_test
        options: >-
          --health-cmd pg_isready
          --health-interval 10s
          --health-timeout 5s
          --health-retries 5
        ports:
          - 5432:5432
          
      neo4j:
        image: neo4j:5.15
        env:
          NEO4J_AUTH: neo4j/test_password
          NEO4J_PLUGINS: '["apoc"]'
        options: >-
          --health-cmd "cypher-shell -u neo4j -p test_password 'RETURN 1'"
          --health-interval 10s
          --health-timeout 5s
          --health-retries 5
        ports:
          - 7687:7687
          - 7474:7474
          
      redis:
        image: redis:7-alpine
        options: >-
          --health-cmd "redis-cli ping"
          --health-interval 10s
          --health-timeout 5s
          --health-retries 5
        ports:
          - 6379:6379
    
    steps:
      - name: Checkout code
        uses: actions/checkout@v4
        
      - name: Set up Python
        uses: actions/setup-python@v4
        with:
          python-version: ${{ env.PYTHON_VERSION }}
          
      - name: Install dependencies
        run: |
          python -m pip install --upgrade pip
          pip install -r requirements-test.txt
          
      - name: Wait for services
        run: |
          sleep 30
          
      - name: Run database migrations
        env:
          DATABASE_URL: postgresql://test_user:test_password@localhost:5432/graphrag_test
          NEO4J_URI: bolt://localhost:7687
          NEO4J_USER: neo4j
          NEO4J_PASSWORD: test_password
        run: |
          python -m alembic upgrade head
          
      - name: Run integration tests
        env:
          DATABASE_URL: postgresql://test_user:test_password@localhost:5432/graphrag_test
          NEO4J_URI: bolt://localhost:7687
          NEO4J_USER: neo4j
          NEO4J_PASSWORD: test_password
          REDIS_URL: redis://localhost:6379/0
        run: |
          pytest tests/integration/ \
            --cov=app \
            --cov-append \
            --cov-report=xml \
            --junitxml=integration-results.xml \
            -v
            
      - name: Upload integration test results
        uses: actions/upload-artifact@v3
        if: always()
        with:
          name: integration-test-results
          path: integration-results.xml

  # 性能测试
  performance-tests:
    name: Performance Tests
    runs-on: ubuntu-latest
    needs: integration-tests
    if: github.event_name == 'schedule' || contains(github.event.head_commit.message, '[perf]')
    
    steps:
      - name: Checkout code
        uses: actions/checkout@v4
        
      - name: Set up Python
        uses: actions/setup-python@v4
        with:
          python-version: ${{ env.PYTHON_VERSION }}
          
      - name: Install dependencies
        run: |
          python -m pip install --upgrade pip
          pip install -r requirements-test.txt
          
      - name: Run performance tests
        run: |
          pytest tests/ -m performance \
            --benchmark-only \
            --benchmark-json=benchmark.json
            
      - name: Upload benchmark results
        uses: actions/upload-artifact@v3
        with:
          name: benchmark-results
          path: benchmark.json

  # 安全扫描
  security-scan:
    name: Security Scan
    runs-on: ubuntu-latest
    needs: code-quality
    
    steps:
      - name: Checkout code
        uses: actions/checkout@v4
        
      - name: Run Trivy vulnerability scanner
        uses: aquasecurity/trivy-action@master
        with:
          scan-type: 'fs'
          scan-ref: '.'
          format: 'sarif'
          output: 'trivy-results.sarif'
          
      - name: Upload Trivy scan results
        uses: github/codeql-action/upload-sarif@v2
        with:
          sarif_file: 'trivy-results.sarif'
          
      - name: Dependency vulnerability scan
        run: |
          pip install safety
          safety check --json --output safety-report.json || true
          
      - name: Upload safety report
        uses: actions/upload-artifact@v3
        with:
          name: safety-report
          path: safety-report.json

  # 构建Docker镜像
  build-image:
    name: Build Docker Image
    runs-on: ubuntu-latest
    needs: [unit-tests, integration-tests]
    if: github.ref == 'refs/heads/main' || github.ref == 'refs/heads/develop'
    
    steps:
      - name: Checkout code
        uses: actions/checkout@v4
        
      - name: Set up Docker Buildx
        uses: docker/setup-buildx-action@v3
        
      - name: Login to Container Registry
        uses: docker/login-action@v3
        with:
          registry: ghcr.io
          username: ${{ github.actor }}
          password: ${{ secrets.GITHUB_TOKEN }}
          
      - name: Extract metadata
        id: meta
        uses: docker/metadata-action@v5
        with:
          images: ghcr.io/${{ github.repository }}
          tags: |
            type=ref,event=branch
            type=ref,event=pr
            type=sha,prefix={{branch}}-
            
      - name: Build and push
        uses: docker/build-push-action@v5
        with:
          context: .
          push: true
          tags: ${{ steps.meta.outputs.tags }}
          labels: ${{ steps.meta.outputs.labels }}
          cache-from: type=gha
          cache-to: type=gha,mode=max

  # 部署到测试环境
  deploy-staging:
    name: Deploy to Staging
    runs-on: ubuntu-latest
    needs: build-image
    if: github.ref == 'refs/heads/develop'
    environment: staging
    
    steps:
      - name: Deploy to staging
        run: |
          echo "Deploying to staging environment..."
          # 这里添加实际的部署脚本
          
  # 部署到生产环境
  deploy-production:
    name: Deploy to Production
    runs-on: ubuntu-latest
    needs: build-image
    if: github.ref == 'refs/heads/main'
    environment: production
    
    steps:
      - name: Deploy to production
        run: |
          echo "Deploying to production environment..."
          # 这里添加实际的部署脚本
```

### 测试报告工作流

创建 `.github/workflows/test-report.yml`:

```yaml
name: Test Report

on:
  workflow_run:
    workflows: ["CI/CD Pipeline"]
    types:
      - completed

jobs:
  test-report:
    name: Generate Test Report
    runs-on: ubuntu-latest
    if: github.event.workflow_run.conclusion != 'skipped'
    
    steps:
      - name: Download test results
        uses: actions/github-script@v6
        with:
          script: |
            let allArtifacts = await github.rest.actions.listWorkflowRunArtifacts({
               owner: context.repo.owner,
               repo: context.repo.repo,
               run_id: context.payload.workflow_run.id,
            });
            
            let matchArtifacts = allArtifacts.data.artifacts.filter((artifact) => {
              return artifact.name.includes("test-results")
            });
            
            for (const artifact of matchArtifacts) {
              let download = await github.rest.actions.downloadArtifact({
                 owner: context.repo.owner,
                 repo: context.repo.repo,
                 artifact_id: artifact.id,
                 archive_format: 'zip',
              });
              
              let fs = require('fs');
              fs.writeFileSync(`${artifact.name}.zip`, Buffer.from(download.data));
            }
            
      - name: Extract test results
        run: |
          for file in *.zip; do
            unzip -o "$file"
          done
          
      - name: Publish test results
        uses: dorny/test-reporter@v1
        if: always()
        with:
          name: Test Results
          path: '**/*test-results.xml'
          reporter: java-junit
          fail-on-error: true
```

## 🧪 测试策略

### 测试阶段划分

1. **快速反馈阶段** (< 5分钟)
   - 代码格式检查
   - 静态分析
   - 单元测试

2. **集成验证阶段** (5-15分钟)
   - 集成测试
   - API测试
   - 数据库测试

3. **全面验证阶段** (15-30分钟)
   - E2E测试
   - 性能测试
   - 安全扫描

### 测试并行化

```yaml
strategy:
  matrix:
    test-group: [unit, integration, api, database]
    python-version: ['3.10', '3.11']
  fail-fast: false
  max-parallel: 4
```

### 条件测试执行

```yaml
# 只在特定条件下运行性能测试
- name: Run performance tests
  if: |
    github.event_name == 'schedule' || 
    contains(github.event.head_commit.message, '[perf]') ||
    github.ref == 'refs/heads/main'
```

## ⚙️ 环境配置

### 环境变量管理

```yaml
env:
  # 全局环境变量
  PYTHON_VERSION: '3.11'
  DATABASE_URL: postgresql://test_user:test_password@localhost:5432/graphrag_test
  NEO4J_URI: bolt://localhost:7687
  NEO4J_USER: neo4j
  NEO4J_PASSWORD: test_password
  REDIS_URL: redis://localhost:6379/0
  
  # 测试配置
  TESTING: true
  LOG_LEVEL: DEBUG
  
  # 外部服务（使用secrets）
  OPENAI_API_KEY: ${{ secrets.OPENAI_API_KEY }}
  EMBEDDING_SERVICE_URL: ${{ secrets.EMBEDDING_SERVICE_URL }}
```

### Secrets 配置

在 GitHub 仓库设置中配置以下 secrets：

- `OPENAI_API_KEY`: OpenAI API密钥
- `DOCKER_REGISTRY_TOKEN`: Docker镜像仓库访问令牌
- `STAGING_DEPLOY_KEY`: 测试环境部署密钥
- `PRODUCTION_DEPLOY_KEY`: 生产环境部署密钥

### 服务依赖

```yaml
services:
  postgres:
    image: postgres:15
    env:
      POSTGRES_PASSWORD: test_password
      POSTGRES_USER: test_user
      POSTGRES_DB: graphrag_test
    options: >-
      --health-cmd pg_isready
      --health-interval 10s
      --health-timeout 5s
      --health-retries 5
    ports:
      - 5432:5432
```

## 🔄 工作流程

### Pull Request 流程

```mermaid
graph TD
    A[创建PR] --> B[代码检查]
    B --> C{检查通过?}
    C -->|否| D[修复问题]
    D --> B
    C -->|是| E[运行测试]
    E --> F{测试通过?}
    F -->|否| G[修复测试]
    G --> E
    F -->|是| H[代码审查]
    H --> I{审查通过?}
    I -->|否| J[修改代码]
    J --> B
    I -->|是| K[合并PR]
```

### 分支策略

- **main**: 生产分支，自动部署到生产环境
- **develop**: 开发分支，自动部署到测试环境
- **feature/***: 功能分支，运行完整测试套件
- **hotfix/***: 热修复分支，快速测试和部署

### 部署策略

```yaml
# 蓝绿部署
deploy-blue-green:
  steps:
    - name: Deploy to blue environment
      run: deploy-blue.sh
      
    - name: Run smoke tests
      run: pytest tests/smoke/ --env=blue
      
    - name: Switch traffic to blue
      if: success()
      run: switch-traffic.sh blue
      
    - name: Cleanup green environment
      run: cleanup-green.sh
```

## 🚪 质量门禁

### 覆盖率要求

```yaml
- name: Check coverage threshold
  run: |
    pytest --cov=app --cov-fail-under=80
```

### 性能基准

```yaml
- name: Performance regression check
  run: |
    pytest tests/performance/ \
      --benchmark-compare=baseline.json \
      --benchmark-compare-fail=mean:10%
```

### 安全扫描

```yaml
- name: Security gate
  run: |
    bandit -r app/ -f json -o bandit-report.json
    python scripts/check_security_gate.py bandit-report.json
```

### 代码质量

```yaml
- name: Code quality gate
  run: |
    # 复杂度检查
    radon cc app/ --min B
    
    # 重复代码检查
    pylint app/ --disable=all --enable=duplicate-code
    
    # 技术债务检查
    sonar-scanner
```

## 📊 监控和报告

### 测试趋势报告

```python
# scripts/generate_test_report.py
import json
import matplotlib.pyplot as plt
from datetime import datetime, timedelta

def generate_test_trend_report():
    """生成测试趋势报告"""
    # 获取最近30天的测试数据
    test_data = get_test_history(days=30)
    
    # 生成图表
    dates = [data['date'] for data in test_data]
    pass_rates = [data['pass_rate'] for data in test_data]
    coverage = [data['coverage'] for data in test_data]
    
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8))
    
    # 测试通过率趋势
    ax1.plot(dates, pass_rates, 'g-', label='通过率')
    ax1.set_ylabel('通过率 (%)')
    ax1.set_title('测试通过率趋势')
    ax1.legend()
    
    # 代码覆盖率趋势
    ax2.plot(dates, coverage, 'b-', label='覆盖率')
    ax2.set_ylabel('覆盖率 (%)')
    ax2.set_xlabel('日期')
    ax2.set_title('代码覆盖率趋势')
    ax2.legend()
    
    plt.tight_layout()
    plt.savefig('test_trend_report.png')
    
    return 'test_trend_report.png'
```

### 性能监控

```yaml
- name: Performance monitoring
  run: |
    # 运行性能测试
    pytest tests/performance/ --benchmark-json=perf.json
    
    # 上传到性能监控系统
    python scripts/upload_performance_data.py perf.json
```

### 通知配置

```yaml
- name: Notify on failure
  if: failure()
  uses: 8398a7/action-slack@v3
  with:
    status: failure
    channel: '#ci-cd'
    webhook_url: ${{ secrets.SLACK_WEBHOOK }}
```

## 🔧 故障排除

### 常见问题

#### 1. 测试超时

```yaml
- name: Run tests with timeout
  timeout-minutes: 30
  run: pytest tests/ --timeout=300
```

#### 2. 资源不足

```yaml
# 使用更大的运行器
runs-on: ubuntu-latest-4-cores
```

#### 3. 网络问题

```yaml
- name: Retry on network failure
  uses: nick-invision/retry@v2
  with:
    timeout_minutes: 10
    max_attempts: 3
    command: pytest tests/integration/
```

#### 4. 依赖冲突

```yaml
- name: Clear pip cache
  run: |
    pip cache purge
    pip install --no-cache-dir -r requirements.txt
```

### 调试技巧

#### 1. 启用调试模式

```yaml
- name: Debug test failure
  if: failure()
  run: |
    pytest tests/ -vvv --tb=long --capture=no
```

#### 2. 保存调试信息

```yaml
- name: Save debug artifacts
  if: failure()
  uses: actions/upload-artifact@v3
  with:
    name: debug-logs
    path: |
      logs/
      *.log
      core.*
```

#### 3. SSH调试

```yaml
- name: Setup tmate session
  if: failure()
  uses: mxschmitt/action-tmate@v3
  timeout-minutes: 30
```

### 性能优化

#### 1. 缓存优化

```yaml
- name: Cache dependencies
  uses: actions/cache@v3
  with:
    path: |
      ~/.cache/pip
      ~/.cache/pytest
    key: ${{ runner.os }}-${{ hashFiles('**/requirements*.txt') }}
```

#### 2. 并行执行

```yaml
- name: Run tests in parallel
  run: |
    pytest tests/ -n auto --dist worksteal
```

#### 3. 增量测试

```yaml
- name: Run only changed tests
  run: |
    pytest --testmon tests/
```

## 📈 持续改进

### 测试指标收集

```python
# scripts/collect_test_metrics.py
def collect_test_metrics():
    """收集测试指标"""
    metrics = {
        'timestamp': datetime.now().isoformat(),
        'total_tests': get_total_test_count(),
        'pass_rate': get_pass_rate(),
        'coverage': get_coverage_percentage(),
        'execution_time': get_execution_time(),
        'flaky_tests': get_flaky_tests(),
        'slow_tests': get_slow_tests()
    }
    
    # 发送到监控系统
    send_metrics_to_monitoring(metrics)
    
    return metrics
```

### 自动化改进建议

```python
# scripts/analyze_test_results.py
def analyze_test_results():
    """分析测试结果并提供改进建议"""
    analysis = {
        'slow_tests': identify_slow_tests(),
        'flaky_tests': identify_flaky_tests(),
        'low_coverage_areas': identify_low_coverage(),
        'duplicate_tests': identify_duplicate_tests(),
        'suggestions': generate_improvement_suggestions()
    }
    
    # 生成报告
    generate_analysis_report(analysis)
    
    return analysis
```

---

**最后更新**: 2024年
**维护者**: GraphRAG Team