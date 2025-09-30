# 部署指南

## 概述

本文档提供了GraphRAG知识库系统的完整部署指南，包括本地开发环境、测试环境、预发布环境和生产环境的部署策略。

## 部署架构

```mermaid
graph TB
    subgraph "负载均衡层"
        LB[Nginx/HAProxy<br/>负载均衡器]
    end
    
    subgraph "应用层"
        API1[FastAPI实例1<br/>8000端口]
        API2[FastAPI实例2<br/>8001端口]
        API3[FastAPI实例3<br/>8002端口]
        
        WORKER1[Celery Worker 1]
        WORKER2[Celery Worker 2]
        WORKER3[Celery Worker 3]
    end
    
    subgraph "数据存储层"
        PG[(PostgreSQL<br/>主数据库)]
        PG_SLAVE[(PostgreSQL<br/>只读副本)]
        NEO4J[(Neo4j<br/>图数据库)]
        WEAVIATE[(Weaviate<br/>向量数据库)]
        REDIS[(Redis<br/>缓存/队列)]
        MINIO[(MinIO<br/>对象存储)]
    end
    
    subgraph "监控层"
        PROMETHEUS[Prometheus<br/>指标收集]
        GRAFANA[Grafana<br/>监控面板]
        LOKI[Loki<br/>日志聚合]
    end
    
    LB --> API1
    LB --> API2
    LB --> API3
    
    API1 --> PG
    API2 --> PG
    API3 --> PG
    
    API1 --> PG_SLAVE
    API2 --> PG_SLAVE
    API3 --> PG_SLAVE
    
    API1 --> NEO4J
    API2 --> NEO4J
    API3 --> NEO4J
    
    API1 --> WEAVIATE
    API2 --> WEAVIATE
    API3 --> WEAVIATE
    
    API1 --> REDIS
    API2 --> REDIS
    API3 --> REDIS
    
    API1 --> MINIO
    API2 --> MINIO
    API3 --> MINIO
    
    WORKER1 --> REDIS
    WORKER2 --> REDIS
    WORKER3 --> REDIS
    
    WORKER1 --> PG
    WORKER2 --> PG
    WORKER3 --> PG
    
    PROMETHEUS --> API1
    PROMETHEUS --> API2
    PROMETHEUS --> API3
    PROMETHEUS --> PG
    PROMETHEUS --> NEO4J
    PROMETHEUS --> REDIS
    
    GRAFANA --> PROMETHEUS
    LOKI --> API1
    LOKI --> API2
    LOKI --> API3
```

## 环境分类

### 1. 本地开发环境

#### Docker Compose配置

```yaml
# docker-compose.dev.yml
version: '3.8'

services:
  # 应用服务
  api:
    build:
      context: .
      dockerfile: Dockerfile.dev
    ports:
      - "8000:8000"
    volumes:
      - .:/app
      - ./logs:/app/logs
      - ./uploads:/app/uploads
    environment:
      - DEBUG=true
      - DATABASE_URL=postgresql+asyncpg://postgres:password@postgres:5432/graphrag_dev
      - REDIS_URL=redis://redis:6379/0
      - NEO4J_URI=bolt://neo4j:7687
      - NEO4J_USER=neo4j
      - NEO4J_PASSWORD=password
      - WEAVIATE_URL=http://weaviate:8080
      - MINIO_ENDPOINT=minio:9000
      - MINIO_ACCESS_KEY=minioadmin
      - MINIO_SECRET_KEY=minioadmin
    depends_on:
      - postgres
      - redis
      - neo4j
      - weaviate
      - minio
    command: uvicorn src.main:app --host 0.0.0.0 --port 8000 --reload

  # Worker服务
  worker:
    build:
      context: .
      dockerfile: Dockerfile.dev
    volumes:
      - .:/app
      - ./logs:/app/logs
      - ./uploads:/app/uploads
    environment:
      - DEBUG=true
      - DATABASE_URL=postgresql+asyncpg://postgres:password@postgres:5432/graphrag_dev
      - REDIS_URL=redis://redis:6379/0
      - NEO4J_URI=bolt://neo4j:7687
      - NEO4J_USER=neo4j
      - NEO4J_PASSWORD=password
      - WEAVIATE_URL=http://weaviate:8080
      - MINIO_ENDPOINT=minio:9000
      - MINIO_ACCESS_KEY=minioadmin
      - MINIO_SECRET_KEY=minioadmin
    depends_on:
      - postgres
      - redis
      - neo4j
      - weaviate
      - minio
    command: celery -A src.workers.celery_app worker --loglevel=info --concurrency=2

  # 数据库服务
  postgres:
    image: postgres:15
    environment:
      POSTGRES_DB: graphrag_dev
      POSTGRES_USER: postgres
      POSTGRES_PASSWORD: password
    ports:
      - "5432:5432"
    volumes:
      - postgres_data:/var/lib/postgresql/data
      - ./scripts/init_db.sql:/docker-entrypoint-initdb.d/init_db.sql

  redis:
    image: redis:7-alpine
    ports:
      - "6379:6379"
    volumes:
      - redis_data:/data

  neo4j:
    image: neo4j:5
    environment:
      NEO4J_AUTH: neo4j/password
      NEO4J_PLUGINS: '["apoc"]'
      NEO4J_dbms_security_procedures_unrestricted: apoc.*
    ports:
      - "7474:7474"
      - "7687:7687"
    volumes:
      - neo4j_data:/data
      - neo4j_logs:/logs

  weaviate:
    image: semitechnologies/weaviate:1.22.4
    ports:
      - "8080:8080"
    environment:
      QUERY_DEFAULTS_LIMIT: 25
      AUTHENTICATION_ANONYMOUS_ACCESS_ENABLED: 'true'
      PERSISTENCE_DATA_PATH: '/var/lib/weaviate'
      DEFAULT_VECTORIZER_MODULE: 'none'
      ENABLE_MODULES: 'text2vec-openai,text2vec-cohere,text2vec-huggingface'
      CLUSTER_HOSTNAME: 'node1'
    volumes:
      - weaviate_data:/var/lib/weaviate

  minio:
    image: minio/minio:latest
    ports:
      - "9000:9000"
      - "9001:9001"
    environment:
      MINIO_ROOT_USER: minioadmin
      MINIO_ROOT_PASSWORD: minioadmin
    volumes:
      - minio_data:/data
    command: server /data --console-address ":9001"

volumes:
  postgres_data:
  redis_data:
  neo4j_data:
  neo4j_logs:
  weaviate_data:
  minio_data:
```

#### 开发环境Dockerfile

```dockerfile
# Dockerfile.dev
FROM python:3.11-slim

WORKDIR /app

# 安装系统依赖
RUN apt-get update && apt-get install -y \
    build-essential \
    curl \
    git \
    && rm -rf /var/lib/apt/lists/*

# 安装Python依赖
COPY requirements.txt requirements-dev.txt ./
RUN pip install --no-cache-dir -r requirements.txt -r requirements-dev.txt

# 复制应用代码
COPY . .

# 设置环境变量
ENV PYTHONPATH=/app
ENV PYTHONUNBUFFERED=1

# 暴露端口
EXPOSE 8000

# 默认命令
CMD ["uvicorn", "src.main:app", "--host", "0.0.0.0", "--port", "8000", "--reload"]
```

### 2. 生产环境部署

#### Kubernetes部署配置

##### Namespace配置

```yaml
# k8s/namespace.yaml
apiVersion: v1
kind: Namespace
metadata:
  name: graphrag-prod
  labels:
    name: graphrag-prod
    environment: production
```

##### ConfigMap配置

```yaml
# k8s/configmap.yaml
apiVersion: v1
kind: ConfigMap
metadata:
  name: graphrag-config
  namespace: graphrag-prod
data:
  API_HOST: "0.0.0.0"
  API_PORT: "8000"
  API_WORKERS: "4"
  LOG_LEVEL: "INFO"
  ALLOWED_ORIGINS: "https://graphrag.example.com"
  MAX_FILE_SIZE: "104857600"
  ALLOWED_FILE_TYPES: ".pdf,.txt,.md,.docx,.html"
  EMBEDDING_MODEL: "sentence-transformers/all-MiniLM-L6-v2"
  LLM_MODEL: "gpt-3.5-turbo"
```

##### Secret配置

```yaml
# k8s/secret.yaml
apiVersion: v1
kind: Secret
metadata:
  name: graphrag-secrets
  namespace: graphrag-prod
type: Opaque
data:
  DATABASE_URL: <base64-encoded-database-url>
  REDIS_URL: <base64-encoded-redis-url>
  NEO4J_URI: <base64-encoded-neo4j-uri>
  NEO4J_USER: <base64-encoded-neo4j-user>
  NEO4J_PASSWORD: <base64-encoded-neo4j-password>
  WEAVIATE_URL: <base64-encoded-weaviate-url>
  WEAVIATE_API_KEY: <base64-encoded-weaviate-key>
  MINIO_ENDPOINT: <base64-encoded-minio-endpoint>
  MINIO_ACCESS_KEY: <base64-encoded-minio-access-key>
  MINIO_SECRET_KEY: <base64-encoded-minio-secret-key>
  OPENAI_API_KEY: <base64-encoded-openai-key>
  SECRET_KEY: <base64-encoded-secret-key>
```

##### API服务部署

```yaml
# k8s/api-deployment.yaml
apiVersion: apps/v1
kind: Deployment
metadata:
  name: graphrag-api
  namespace: graphrag-prod
  labels:
    app: graphrag-api
    version: v1
spec:
  replicas: 3
  selector:
    matchLabels:
      app: graphrag-api
  template:
    metadata:
      labels:
        app: graphrag-api
        version: v1
    spec:
      containers:
      - name: api
        image: graphrag/api:latest
        ports:
        - containerPort: 8000
        env:
        - name: DEBUG
          value: "false"
        envFrom:
        - configMapRef:
            name: graphrag-config
        - secretRef:
            name: graphrag-secrets
        resources:
          requests:
            memory: "512Mi"
            cpu: "250m"
          limits:
            memory: "1Gi"
            cpu: "500m"
        livenessProbe:
          httpGet:
            path: /health
            port: 8000
          initialDelaySeconds: 30
          periodSeconds: 10
        readinessProbe:
          httpGet:
            path: /ready
            port: 8000
          initialDelaySeconds: 5
          periodSeconds: 5
        volumeMounts:
        - name: logs
          mountPath: /app/logs
        - name: uploads
          mountPath: /app/uploads
      volumes:
      - name: logs
        emptyDir: {}
      - name: uploads
        persistentVolumeClaim:
          claimName: graphrag-uploads-pvc

---
apiVersion: v1
kind: Service
metadata:
  name: graphrag-api-service
  namespace: graphrag-prod
spec:
  selector:
    app: graphrag-api
  ports:
  - protocol: TCP
    port: 80
    targetPort: 8000
  type: ClusterIP

---
apiVersion: v1
kind: PersistentVolumeClaim
metadata:
  name: graphrag-uploads-pvc
  namespace: graphrag-prod
spec:
  accessModes:
    - ReadWriteMany
  resources:
    requests:
      storage: 100Gi
  storageClassName: nfs-client
```

##### Worker服务部署

```yaml
# k8s/worker-deployment.yaml
apiVersion: apps/v1
kind: Deployment
metadata:
  name: graphrag-worker
  namespace: graphrag-prod
  labels:
    app: graphrag-worker
    version: v1
spec:
  replicas: 5
  selector:
    matchLabels:
      app: graphrag-worker
  template:
    metadata:
      labels:
        app: graphrag-worker
        version: v1
    spec:
      containers:
      - name: worker
        image: graphrag/worker:latest
        env:
        - name: DEBUG
          value: "false"
        - name: CELERY_CONCURRENCY
          value: "4"
        envFrom:
        - configMapRef:
            name: graphrag-config
        - secretRef:
            name: graphrag-secrets
        resources:
          requests:
            memory: "1Gi"
            cpu: "500m"
          limits:
            memory: "2Gi"
            cpu: "1000m"
        volumeMounts:
        - name: logs
          mountPath: /app/logs
        - name: uploads
          mountPath: /app/uploads
        command: ["celery"]
        args: ["-A", "src.workers.celery_app", "worker", "--loglevel=info", "--concurrency=4"]
      volumes:
      - name: logs
        emptyDir: {}
      - name: uploads
        persistentVolumeClaim:
          claimName: graphrag-uploads-pvc
```

##### Ingress配置

```yaml
# k8s/ingress.yaml
apiVersion: networking.k8s.io/v1
kind: Ingress
metadata:
  name: graphrag-ingress
  namespace: graphrag-prod
  annotations:
    kubernetes.io/ingress.class: nginx
    cert-manager.io/cluster-issuer: letsencrypt-prod
    nginx.ingress.kubernetes.io/proxy-body-size: "100m"
    nginx.ingress.kubernetes.io/proxy-read-timeout: "300"
    nginx.ingress.kubernetes.io/proxy-send-timeout: "300"
    nginx.ingress.kubernetes.io/rate-limit: "100"
    nginx.ingress.kubernetes.io/rate-limit-window: "1m"
spec:
  tls:
  - hosts:
    - graphrag.example.com
    secretName: graphrag-tls
  rules:
  - host: graphrag.example.com
    http:
      paths:
      - path: /
        pathType: Prefix
        backend:
          service:
            name: graphrag-api-service
            port:
              number: 80
```

#### 生产环境Dockerfile

```dockerfile
# Dockerfile.prod
FROM python:3.11-slim as builder

WORKDIR /app

# 安装构建依赖
RUN apt-get update && apt-get install -y \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# 安装Python依赖
COPY requirements.txt .
RUN pip install --no-cache-dir --user -r requirements.txt

# 生产镜像
FROM python:3.11-slim

WORKDIR /app

# 安装运行时依赖
RUN apt-get update && apt-get install -y \
    curl \
    && rm -rf /var/lib/apt/lists/* \
    && groupadd -r appuser && useradd -r -g appuser appuser

# 复制Python包
COPY --from=builder /root/.local /root/.local

# 复制应用代码
COPY src/ ./src/
COPY alembic/ ./alembic/
COPY alembic.ini .

# 设置环境变量
ENV PATH=/root/.local/bin:$PATH
ENV PYTHONPATH=/app
ENV PYTHONUNBUFFERED=1

# 创建必要目录
RUN mkdir -p logs uploads && chown -R appuser:appuser /app

# 切换到非root用户
USER appuser

# 健康检查
HEALTHCHECK --interval=30s --timeout=30s --start-period=5s --retries=3 \
    CMD curl -f http://localhost:8000/health || exit 1

# 暴露端口
EXPOSE 8000

# 启动命令
CMD ["gunicorn", "src.main:app", "-w", "4", "-k", "uvicorn.workers.UvicornWorker", "--bind", "0.0.0.0:8000"]
```

### 3. 数据库部署

#### PostgreSQL高可用配置

```yaml
# k8s/postgres-ha.yaml
apiVersion: postgresql.cnpg.io/v1
kind: Cluster
metadata:
  name: postgres-cluster
  namespace: graphrag-prod
spec:
  instances: 3
  
  postgresql:
    parameters:
      max_connections: "200"
      shared_buffers: "256MB"
      effective_cache_size: "1GB"
      maintenance_work_mem: "64MB"
      checkpoint_completion_target: "0.9"
      wal_buffers: "16MB"
      default_statistics_target: "100"
      random_page_cost: "1.1"
      effective_io_concurrency: "200"
      work_mem: "4MB"
      min_wal_size: "1GB"
      max_wal_size: "4GB"
  
  bootstrap:
    initdb:
      database: graphrag_prod
      owner: graphrag_user
      secret:
        name: postgres-credentials
  
  storage:
    size: 500Gi
    storageClass: fast-ssd
  
  monitoring:
    enabled: true
  
  backup:
    retentionPolicy: "30d"
    barmanObjectStore:
      destinationPath: "s3://graphrag-backups/postgres"
      s3Credentials:
        accessKeyId:
          name: backup-credentials
          key: ACCESS_KEY_ID
        secretAccessKey:
          name: backup-credentials
          key: SECRET_ACCESS_KEY
      wal:
        retention: "7d"
      data:
        retention: "30d"
```

#### Neo4j集群配置

```yaml
# k8s/neo4j-cluster.yaml
apiVersion: v1
kind: ConfigMap
metadata:
  name: neo4j-config
  namespace: graphrag-prod
data:
  neo4j.conf: |
    dbms.mode=CORE
    causal_clustering.minimum_core_cluster_size_at_formation=3
    causal_clustering.minimum_core_cluster_size_at_runtime=3
    causal_clustering.initial_discovery_members=neo4j-core-0.neo4j-core:5000,neo4j-core-1.neo4j-core:5000,neo4j-core-2.neo4j-core:5000
    dbms.connector.bolt.listen_address=0.0.0.0:7687
    dbms.connector.http.listen_address=0.0.0.0:7474
    dbms.security.auth_enabled=true
    dbms.logs.query.enabled=true
    dbms.memory.heap.initial_size=2G
    dbms.memory.heap.max_size=2G
    dbms.memory.pagecache.size=1G

---
apiVersion: apps/v1
kind: StatefulSet
metadata:
  name: neo4j-core
  namespace: graphrag-prod
spec:
  serviceName: neo4j-core
  replicas: 3
  selector:
    matchLabels:
      app: neo4j-core
  template:
    metadata:
      labels:
        app: neo4j-core
    spec:
      containers:
      - name: neo4j
        image: neo4j:5-enterprise
        ports:
        - containerPort: 7474
        - containerPort: 7687
        - containerPort: 5000
        - containerPort: 6000
        - containerPort: 7000
        env:
        - name: NEO4J_AUTH
          valueFrom:
            secretKeyRef:
              name: neo4j-auth
              key: auth
        - name: NEO4J_ACCEPT_LICENSE_AGREEMENT
          value: "yes"
        volumeMounts:
        - name: neo4j-data
          mountPath: /data
        - name: neo4j-config
          mountPath: /var/lib/neo4j/conf
        resources:
          requests:
            memory: "4Gi"
            cpu: "1000m"
          limits:
            memory: "6Gi"
            cpu: "2000m"
      volumes:
      - name: neo4j-config
        configMap:
          name: neo4j-config
  volumeClaimTemplates:
  - metadata:
      name: neo4j-data
    spec:
      accessModes: ["ReadWriteOnce"]
      resources:
        requests:
          storage: 200Gi
      storageClassName: fast-ssd
```

### 4. 监控和日志

#### Prometheus配置

```yaml
# k8s/prometheus-config.yaml
apiVersion: v1
kind: ConfigMap
metadata:
  name: prometheus-config
  namespace: graphrag-prod
data:
  prometheus.yml: |
    global:
      scrape_interval: 15s
      evaluation_interval: 15s
    
    rule_files:
      - "alert_rules.yml"
    
    alerting:
      alertmanagers:
        - static_configs:
            - targets:
              - alertmanager:9093
    
    scrape_configs:
      - job_name: 'graphrag-api'
        static_configs:
          - targets: ['graphrag-api-service:80']
        metrics_path: /metrics
        scrape_interval: 30s
      
      - job_name: 'postgres'
        static_configs:
          - targets: ['postgres-cluster-rw:5432']
        metrics_path: /metrics
      
      - job_name: 'neo4j'
        static_configs:
          - targets: ['neo4j-core:7474']
        metrics_path: /db/manage/server/jmx/domain/org.neo4j
      
      - job_name: 'redis'
        static_configs:
          - targets: ['redis:6379']
      
      - job_name: 'kubernetes-pods'
        kubernetes_sd_configs:
          - role: pod
        relabel_configs:
          - source_labels: [__meta_kubernetes_pod_annotation_prometheus_io_scrape]
            action: keep
            regex: true

  alert_rules.yml: |
    groups:
      - name: graphrag_alerts
        rules:
          - alert: APIHighErrorRate
            expr: rate(http_requests_total{status=~"5.."}[5m]) > 0.1
            for: 5m
            labels:
              severity: critical
            annotations:
              summary: "GraphRAG API高错误率"
              description: "API错误率超过10%，持续5分钟"
          
          - alert: DatabaseConnectionHigh
            expr: pg_stat_activity_count > 150
            for: 2m
            labels:
              severity: warning
            annotations:
              summary: "数据库连接数过高"
              description: "PostgreSQL连接数超过150"
          
          - alert: DiskSpaceHigh
            expr: (node_filesystem_avail_bytes / node_filesystem_size_bytes) < 0.1
            for: 1m
            labels:
              severity: critical
            annotations:
              summary: "磁盘空间不足"
              description: "磁盘使用率超过90%"
```

#### Grafana仪表板配置

```json
{
  "dashboard": {
    "title": "GraphRAG System Overview",
    "panels": [
      {
        "title": "API请求率",
        "type": "graph",
        "targets": [
          {
            "expr": "rate(http_requests_total[5m])",
            "legendFormat": "{{method}} {{endpoint}}"
          }
        ]
      },
      {
        "title": "API响应时间",
        "type": "graph",
        "targets": [
          {
            "expr": "histogram_quantile(0.95, rate(http_request_duration_seconds_bucket[5m]))",
            "legendFormat": "95th percentile"
          }
        ]
      },
      {
        "title": "数据库连接",
        "type": "singlestat",
        "targets": [
          {
            "expr": "pg_stat_activity_count",
            "legendFormat": "Active Connections"
          }
        ]
      },
      {
        "title": "Celery任务队列",
        "type": "graph",
        "targets": [
          {
            "expr": "celery_tasks_total",
            "legendFormat": "{{state}}"
          }
        ]
      }
    ]
  }
}
```

### 5. 备份和恢复策略

#### 数据库备份脚本

```bash
#!/bin/bash
# scripts/backup.sh

set -e

# 配置
BACKUP_DIR="/backups"
DATE=$(date +%Y%m%d_%H%M%S)
RETENTION_DAYS=30

# PostgreSQL备份
echo "开始PostgreSQL备份..."
pg_dump $DATABASE_URL | gzip > "$BACKUP_DIR/postgres_$DATE.sql.gz"

# Neo4j备份
echo "开始Neo4j备份..."
neo4j-admin backup --backup-dir="$BACKUP_DIR/neo4j_$DATE" --name=graph.db

# 压缩Neo4j备份
tar -czf "$BACKUP_DIR/neo4j_$DATE.tar.gz" -C "$BACKUP_DIR" "neo4j_$DATE"
rm -rf "$BACKUP_DIR/neo4j_$DATE"

# 上传到云存储
echo "上传备份到云存储..."
aws s3 cp "$BACKUP_DIR/postgres_$DATE.sql.gz" "s3://graphrag-backups/postgres/"
aws s3 cp "$BACKUP_DIR/neo4j_$DATE.tar.gz" "s3://graphrag-backups/neo4j/"

# 清理本地旧备份
echo "清理本地旧备份..."
find "$BACKUP_DIR" -name "postgres_*.sql.gz" -mtime +$RETENTION_DAYS -delete
find "$BACKUP_DIR" -name "neo4j_*.tar.gz" -mtime +$RETENTION_DAYS -delete

echo "备份完成: $DATE"
```

#### 恢复脚本

```bash
#!/bin/bash
# scripts/restore.sh

set -e

BACKUP_DATE=$1
BACKUP_DIR="/backups"

if [ -z "$BACKUP_DATE" ]; then
    echo "使用方法: $0 <backup_date>"
    echo "示例: $0 20240101_120000"
    exit 1
fi

# 从云存储下载备份
echo "从云存储下载备份..."
aws s3 cp "s3://graphrag-backups/postgres/postgres_$BACKUP_DATE.sql.gz" "$BACKUP_DIR/"
aws s3 cp "s3://graphrag-backups/neo4j/neo4j_$BACKUP_DATE.tar.gz" "$BACKUP_DIR/"

# 恢复PostgreSQL
echo "恢复PostgreSQL..."
gunzip -c "$BACKUP_DIR/postgres_$BACKUP_DATE.sql.gz" | psql $DATABASE_URL

# 恢复Neo4j
echo "恢复Neo4j..."
systemctl stop neo4j
tar -xzf "$BACKUP_DIR/neo4j_$BACKUP_DATE.tar.gz" -C /var/lib/neo4j/data/
systemctl start neo4j

echo "恢复完成: $BACKUP_DATE"
```

### 6. 部署自动化

#### Helm Chart配置

```yaml
# helm/graphrag/Chart.yaml
apiVersion: v2
name: graphrag
description: GraphRAG Knowledge Base System
type: application
version: 1.0.0
appVersion: "1.0.0"

dependencies:
  - name: postgresql
    version: 12.1.2
    repository: https://charts.bitnami.com/bitnami
  - name: redis
    version: 17.4.3
    repository: https://charts.bitnami.com/bitnami
```

```yaml
# helm/graphrag/values.yaml
replicaCount: 3

image:
  repository: graphrag/api
  pullPolicy: IfNotPresent
  tag: "latest"

service:
  type: ClusterIP
  port: 80

ingress:
  enabled: true
  className: "nginx"
  annotations:
    cert-manager.io/cluster-issuer: letsencrypt-prod
  hosts:
    - host: graphrag.example.com
      paths:
        - path: /
          pathType: Prefix
  tls:
    - secretName: graphrag-tls
      hosts:
        - graphrag.example.com

resources:
  limits:
    cpu: 500m
    memory: 1Gi
  requests:
    cpu: 250m
    memory: 512Mi

autoscaling:
  enabled: true
  minReplicas: 3
  maxReplicas: 10
  targetCPUUtilizationPercentage: 80

postgresql:
  enabled: true
  auth:
    postgresPassword: "secure-password"
    database: "graphrag_prod"
  primary:
    persistence:
      enabled: true
      size: 500Gi

redis:
  enabled: true
  auth:
    enabled: true
    password: "secure-redis-password"
  master:
    persistence:
      enabled: true
      size: 50Gi
```

#### 部署脚本

```bash
#!/bin/bash
# scripts/deploy.sh

set -e

ENVIRONMENT=${1:-production}
NAMESPACE="graphrag-${ENVIRONMENT}"

echo "部署GraphRAG到环境: $ENVIRONMENT"

# 创建命名空间
kubectl create namespace $NAMESPACE --dry-run=client -o yaml | kubectl apply -f -

# 部署Helm Chart
helm upgrade --install graphrag ./helm/graphrag \
    --namespace $NAMESPACE \
    --values ./helm/graphrag/values-${ENVIRONMENT}.yaml \
    --wait \
    --timeout 10m

# 等待部署完成
kubectl rollout status deployment/graphrag-api -n $NAMESPACE

# 运行数据库迁移
kubectl run migration-job \
    --image=graphrag/api:latest \
    --restart=Never \
    --namespace=$NAMESPACE \
    --command -- alembic upgrade head

# 验证部署
kubectl get pods -n $NAMESPACE
kubectl get services -n $NAMESPACE
kubectl get ingress -n $NAMESPACE

echo "部署完成!"
```

## 运维指南

### 1. 日常维护任务

- **监控检查**: 每日检查Grafana仪表板和告警
- **日志审查**: 定期审查应用和系统日志
- **性能优化**: 监控资源使用情况，及时调整配置
- **安全更新**: 定期更新系统和依赖包
- **备份验证**: 定期验证备份的完整性和可恢复性

### 2. 扩容策略

- **水平扩容**: 增加API和Worker实例数量
- **垂直扩容**: 增加单个实例的CPU和内存
- **数据库扩容**: 增加数据库连接池大小和存储容量
- **缓存扩容**: 增加Redis集群节点

### 3. 故障排除

- **API响应慢**: 检查数据库连接、缓存命中率、资源使用情况
- **任务队列堆积**: 增加Worker数量，检查任务执行时间
- **数据库连接超时**: 检查连接池配置，数据库性能
- **内存泄漏**: 监控内存使用趋势，重启相关服务

### 4. 安全最佳实践

- **网络隔离**: 使用网络策略限制Pod间通信
- **访问控制**: 实施RBAC和最小权限原则
- **数据加密**: 启用传输和存储加密
- **定期审计**: 定期进行安全审计和漏洞扫描

## 总结

这个部署指南提供了：

1. **完整的部署架构**: 从开发到生产的全环境覆盖
2. **详细的配置文件**: Kubernetes、Docker、Helm等配置
3. **自动化部署流程**: CI/CD集成和自动化脚本
4. **监控和日志方案**: Prometheus、Grafana、日志聚合
5. **备份恢复策略**: 数据保护和灾难恢复
6. **运维最佳实践**: 日常维护和故障排除指南

通过遵循这些部署策略和最佳实践，可以确保GraphRAG系统的高可用性、可扩展性和安全性。