#!/bin/bash

# GraphRAG 知识库系统 - 基础服务健康检查脚本
# 功能: 测试 PostgreSQL、Neo4j、Redis、MinIO、Weaviate、MinerU 等服务是否正常运行
# 作者: GraphRAG Team
# 使用方法: ./scripts/test-services.sh

set -e  # 遇到错误立即退出

# 颜色定义
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# 日志函数
log_info() {
    echo -e "${BLUE}[INFO]${NC} $1"
}

log_success() {
    echo -e "${GREEN}[SUCCESS]${NC} $1"
}

log_warning() {
    echo -e "${YELLOW}[WARNING]${NC} $1"
}

log_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

# 检查命令是否存在
check_command() {
    if ! command -v $1 &> /dev/null; then
        log_error "$1 命令未找到，请先安装"
        return 1
    fi
}

# 等待服务启动
wait_for_service() {
    local service_name=$1
    local check_command=$2
    local max_attempts=30
    local attempt=1
    
    log_info "等待 $service_name 服务启动..."
    
    while [ $attempt -le $max_attempts ]; do
        if eval $check_command &> /dev/null; then
            log_success "$service_name 服务已启动"
            return 0
        fi
        
        echo -n "."
        sleep 2
        attempt=$((attempt + 1))
    done
    
    log_error "$service_name 服务启动超时"
    return 1
}

# 测试 PostgreSQL 连接
test_postgres() {
    log_info "测试 PostgreSQL 连接..."
    
    # 检查 psql 命令
    if ! check_command psql; then
        log_warning "psql 未安装，跳过 PostgreSQL 详细测试"
        # 使用 curl 进行基本连接测试
        if nc -z localhost 5432 2>/dev/null; then
            log_success "PostgreSQL 端口 5432 可访问"
            return 0
        else
            log_error "PostgreSQL 端口 5432 不可访问"
            return 1
        fi
    fi
    
    # 使用 psql 测试连接
    export PGPASSWORD="graphrag123"
    if psql -h localhost -p 5432 -U graphrag -d graphrag -c "SELECT version();" &> /dev/null; then
        log_success "PostgreSQL 连接测试成功"
        
        # 获取版本信息
        version=$(psql -h localhost -p 5432 -U graphrag -d graphrag -t -c "SELECT version();" 2>/dev/null | head -1 | xargs)
        log_info "PostgreSQL 版本: $version"
        return 0
    else
        log_error "PostgreSQL 连接测试失败"
        return 1
    fi
}

# 测试 Neo4j 连接
test_neo4j() {
    log_info "测试 Neo4j 连接..."
    
    # 测试 HTTP 接口
    if curl -f -s -u neo4j:neo4j123 http://localhost:7474/db/data/ &> /dev/null; then
        log_success "Neo4j HTTP 接口连接成功"
        
        # 获取版本信息
        version=$(curl -s -u neo4j:neo4j123 http://localhost:7474/db/data/ | grep -o '"neo4j_version":"[^"]*' | cut -d'"' -f4)
        if [ ! -z "$version" ]; then
            log_info "Neo4j 版本: $version"
        fi
        return 0
    else
        log_error "Neo4j HTTP 接口连接失败"
        return 1
    fi
}

# 测试 Redis 连接
test_redis() {
    log_info "测试 Redis 连接..."
    
    # 检查 redis-cli 命令
    if ! check_command redis-cli; then
        log_warning "redis-cli 未安装，跳过 Redis 详细测试"
        # 使用 nc 进行基本连接测试
        if nc -z localhost 6379 2>/dev/null; then
            log_success "Redis 端口 6379 可访问"
            return 0
        else
            log_error "Redis 端口 6379 不可访问"
            return 1
        fi
    fi
    
    # 使用 redis-cli 测试连接
    if redis-cli -h localhost -p 6379 -a redis123 ping | grep -q "PONG"; then
        log_success "Redis 连接测试成功"
        
        # 获取版本信息
        version=$(redis-cli -h localhost -p 6379 -a redis123 info server | grep "redis_version" | cut -d: -f2 | tr -d '\r')
        log_info "Redis 版本: $version"
        return 0
    else
        log_error "Redis 连接测试失败"
        return 1
    fi
}

# 测试 MinIO 连接
test_minio() {
    log_info "测试 MinIO 连接..."
    
    # 测试 API 接口
    if curl -f -s http://localhost:9000/minio/health/live &> /dev/null; then
        log_success "MinIO API 接口连接成功"
        return 0
    else
        log_error "MinIO API 接口连接失败"
        return 1
    fi
}

# 测试 Weaviate 连接
test_weaviate() {
    log_info "测试 Weaviate 连接..."
    
    # 测试健康检查接口
    if curl -f -s http://localhost:8080/v1/.well-known/ready &> /dev/null; then
        log_success "Weaviate 连接测试成功"
        
        # 获取版本信息
        version=$(curl -s http://localhost:8080/v1/meta | grep -o '"version":"[^"]*' | cut -d'"' -f4)
        if [ ! -z "$version" ]; then
            log_info "Weaviate 版本: $version"
        fi
        return 0
    else
        log_error "Weaviate 连接测试失败"
        return 1
    fi
}

# 测试 MinerU 连接
test_mineru() {
    log_info "测试 MinerU 连接..."
    
    # 测试 Streamlit 健康检查
    if curl -f -s http://localhost:8501/_stcore/health &> /dev/null; then
        log_success "MinerU 连接测试成功"
        return 0
    else
        log_error "MinerU 连接测试失败"
        return 1
    fi
}

# 主函数
main() {
    echo "=========================================="
    echo "GraphRAG 基础服务健康检查"
    echo "=========================================="
    
    # 检查 Docker Compose 是否运行
    if ! docker-compose ps | grep -q "Up"; then
        log_error "Docker Compose 服务未运行，请先执行: docker-compose up -d"
        exit 1
    fi
    
    log_info "开始检查各个服务状态..."
    echo ""
    
    # 服务测试结果统计
    total_services=6
    passed_services=0
    
    # 测试各个服务
    services=(
        "PostgreSQL:test_postgres"
        "Neo4j:test_neo4j" 
        "Redis:test_redis"
        "MinIO:test_minio"
        "Weaviate:test_weaviate"
        "MinerU:test_mineru"
    )
    
    for service_test in "${services[@]}"; do
        service_name=$(echo $service_test | cut -d: -f1)
        test_function=$(echo $service_test | cut -d: -f2)
        
        echo "----------------------------------------"
        if $test_function; then
            passed_services=$((passed_services + 1))
        fi
        echo ""
    done
    
    # 输出测试结果摘要
    echo "=========================================="
    echo "测试结果摘要"
    echo "=========================================="
    log_info "总服务数: $total_services"
    log_success "通过测试: $passed_services"
    
    if [ $passed_services -eq $total_services ]; then
        log_success "所有服务运行正常！"
        exit 0
    else
        failed_services=$((total_services - passed_services))
        log_error "失败服务数: $failed_services"
        log_warning "请检查失败的服务并重新启动"
        exit 1
    fi
}

# 执行主函数
main "$@"