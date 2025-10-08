#!/bin/bash

# GraphRAG 知识库系统启动脚本
# ============================
#
# 本脚本用于快速启动 GraphRAG 系统及其所有依赖服务
# 包括环境检查、服务启动、健康检查和初始化等功能
#
# 使用方法：
# 1. 启动所有服务：./scripts/start.sh
# 2. 启动特定服务：./scripts/start.sh --service api
# 3. 开发模式启动：./scripts/start.sh --dev
# 4. 生产模式启动：./scripts/start.sh --prod
# 5. 查看帮助：./scripts/start.sh --help
#
# 作者: GraphRAG Team
# 创建时间: 2024
# 版本: 1.0.0

set -e  # 遇到错误立即退出

# 颜色定义
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
PURPLE='\033[0;35m'
CYAN='\033[0;36m'
NC='\033[0m' # No Color

# 日志函数
log_info() {
    echo -e "${GREEN}[INFO]${NC} $1"
}

log_warn() {
    echo -e "${YELLOW}[WARN]${NC} $1"
}

log_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

log_debug() {
    if [[ "$DEBUG" == "true" ]]; then
        echo -e "${BLUE}[DEBUG]${NC} $1"
    fi
}

log_step() {
    echo -e "${PURPLE}[STEP]${NC} $1"
}

# 显示横幅
show_banner() {
    echo -e "${CYAN}"
    cat << "EOF"
   ____                 _     ____      _    ____ 
  / ___|_ __ __ _ _ __ | |__ |  _ \    / \  / ___|
 | |  _| '__/ _` | '_ \| '_ \| |_) |  / _ \| |  _ 
 | |_| | | | (_| | |_) | | | |  _ <  / ___ \ |_| |
  \____|_|  \__,_| .__/|_| |_|_| \_\/_/   \_\____|
                 |_|                              
    知识库系统 - Knowledge Base System
EOF
    echo -e "${NC}"
    echo -e "${GREEN}GraphRAG 知识库系统启动脚本${NC}"
    echo -e "${GREEN}版本: 1.0.0${NC}"
    echo -e "${GREEN}作者: GraphRAG Team${NC}"
    echo ""
}

# 显示帮助信息
show_help() {
    cat << EOF
GraphRAG 知识库系统启动脚本

使用方法:
    $0 [选项]

选项:
    --help, -h          显示此帮助信息
    --dev               开发模式启动（启用热重载和调试）
    --prod              生产模式启动（优化性能）
    --service SERVICE   启动特定服务 (api, postgres, neo4j, redis, weaviate, minio)
    --no-build          跳过构建步骤
    --no-init           跳过初始化步骤
    --no-health         跳过健康检查
    --logs              启动后显示日志
    --stop              停止所有服务
    --restart           重启所有服务
    --status            显示服务状态
    --clean             清理所有数据（危险操作）

示例:
    $0                  # 启动所有服务
    $0 --dev            # 开发模式启动
    $0 --prod           # 生产模式启动
    $0 --service api    # 只启动 API 服务
    $0 --logs           # 启动并显示日志
    $0 --stop           # 停止所有服务
    $0 --status         # 显示服务状态

环境变量:
    DEBUG=true          启用调试模式
    COMPOSE_FILE        指定 docker-compose 文件路径
    API_PORT            API 服务端口（默认: 8000）
    POSTGRES_PORT       PostgreSQL 端口（默认: 5432）
    NEO4J_HTTP_PORT     Neo4j HTTP 端口（默认: 7475）
    NEO4J_BOLT_PORT     Neo4j Bolt 端口（默认: 7688）

EOF
}

# 检查依赖
check_dependencies() {
    log_step "检查系统依赖..."
    
    # 检查 Docker
    if ! command -v docker &> /dev/null; then
        log_error "Docker 未安装，请先安装 Docker"
        exit 1
    fi
    
    # 检查 Docker Compose
    if ! command -v docker-compose &> /dev/null && ! docker compose version &> /dev/null; then
        log_error "Docker Compose 未安装，请先安装 Docker Compose"
        exit 1
    fi
    
    # 检查 curl
    if ! command -v curl &> /dev/null; then
        log_warn "curl 未安装，健康检查功能将受限"
    fi
    
    log_info "系统依赖检查完成"
}

# 检查环境文件
check_env_file() {
    log_step "检查环境配置..."
    
    if [[ ! -f ".env" ]]; then
        log_warn ".env 文件不存在，创建默认配置..."
        cat > .env << EOF
# GraphRAG 系统环境配置
# ===================

# 应用配置
APP_NAME=GraphRAG Knowledge Base
VERSION=1.0.0
ENVIRONMENT=development
DEBUG=true

# 数据库配置
POSTGRES_DB=graphrag
POSTGRES_USER=graphrag
POSTGRES_PASSWORD=graphrag123

# Neo4j 配置
NEO4J_USER=neo4j
NEO4J_PASSWORD=neo4j123
NEO4J_DATABASE=graphrag

# Redis 配置
REDIS_PASSWORD=redis123

# MinIO 配置
MINIO_ROOT_USER=minioadmin
MINIO_ROOT_PASSWORD=minioadmin123
MINIO_BUCKET=graphrag

# API 配置
SECRET_KEY=your-secret-key-change-in-production
ACCESS_TOKEN_EXPIRE_MINUTES=30

# 日志配置
LOG_LEVEL=INFO
LOG_FORMAT=structured

# OpenAI 配置（可选）
# OPENAI_API_KEY=your-openai-api-key
# OPENAI_BASE_URL=https://api.openai.com/v1
# OPENAI_MODEL=gpt-3.5-turbo
# OPENAI_EMBEDDING_MODEL=text-embedding-ada-002
EOF
        log_info "已创建默认 .env 文件，请根据需要修改配置"
    else
        log_info "环境配置文件存在"
    fi
}

# 创建必要的目录
create_directories() {
    log_step "创建必要的目录..."
    
    directories=(
        "logs"
        "data"
        "uploads"
        "init-scripts/postgres"
        "nginx/conf.d"
        "nginx/ssl"
    )
    
    for dir in "${directories[@]}"; do
        if [[ ! -d "$dir" ]]; then
            mkdir -p "$dir"
            log_debug "创建目录: $dir"
        fi
    done
    
    log_info "目录创建完成"
}

# 构建服务
build_services() {
    if [[ "$NO_BUILD" == "true" ]]; then
        log_info "跳过构建步骤"
        return
    fi
    
    log_step "构建 Docker 镜像..."
    
    if [[ -n "$SERVICE" ]]; then
        docker-compose build "$SERVICE"
    else
        docker-compose build
    fi
    
    log_info "镜像构建完成"
}

# 启动服务
start_services() {
    log_step "启动服务..."
    
    local compose_args=""
    
    if [[ "$MODE" == "dev" ]]; then
        compose_args="$compose_args -f docker-compose.yml -f docker-compose.dev.yml"
    elif [[ "$MODE" == "prod" ]]; then
        compose_args="$compose_args -f docker-compose.yml -f docker-compose.prod.yml"
    fi
    
    if [[ -n "$SERVICE" ]]; then
        docker-compose $compose_args up -d "$SERVICE"
        log_info "服务 $SERVICE 启动完成"
    else
        docker-compose $compose_args up -d
        log_info "所有服务启动完成"
    fi
}

# 等待服务就绪
wait_for_services() {
    if [[ "$NO_HEALTH" == "true" ]]; then
        log_info "跳过健康检查"
        return
    fi
    
    log_step "等待服务就绪..."
    
    local services=("postgres" "neo4j" "redis" "weaviate" "minio")
    local max_attempts=30
    local attempt=1
    
    for service in "${services[@]}"; do
        log_info "等待 $service 服务就绪..."
        
        while [[ $attempt -le $max_attempts ]]; do
            if docker-compose ps "$service" | grep -q "healthy\|Up"; then
                log_info "$service 服务已就绪"
                break
            fi
            
            if [[ $attempt -eq $max_attempts ]]; then
                log_error "$service 服务启动超时"
                return 1
            fi
            
            log_debug "等待 $service 服务... ($attempt/$max_attempts)"
            sleep 5
            ((attempt++))
        done
        
        attempt=1
    done
    
    # 等待 API 服务
    if [[ -z "$SERVICE" ]] || [[ "$SERVICE" == "api" ]]; then
        log_info "等待 API 服务就绪..."
        local api_port=${API_PORT:-8000}
        
        while [[ $attempt -le $max_attempts ]]; do
            if curl -f -s "http://localhost:$api_port/health" > /dev/null 2>&1; then
                log_info "API 服务已就绪"
                break
            fi
            
            if [[ $attempt -eq $max_attempts ]]; then
                log_error "API 服务启动超时"
                return 1
            fi
            
            log_debug "等待 API 服务... ($attempt/$max_attempts)"
            sleep 5
            ((attempt++))
        done
    fi
    
    log_info "所有服务已就绪"
}

# 初始化数据库
initialize_databases() {
    if [[ "$NO_INIT" == "true" ]]; then
        log_info "跳过初始化步骤"
        return
    fi
    
    log_step "初始化数据库..."
    
    # 初始化 Neo4j
    log_info "初始化 Neo4j 图数据库..."
    if [[ -f "scripts/init_neo4j.py" ]]; then
        docker-compose exec -T api python scripts/init_neo4j.py || log_warn "Neo4j 初始化失败"
    else
        log_warn "Neo4j 初始化脚本不存在"
    fi
    
    log_info "数据库初始化完成"
}

# 显示服务状态
show_status() {
    log_step "显示服务状态..."
    
    echo -e "${CYAN}服务状态:${NC}"
    docker-compose ps
    
    echo -e "\n${CYAN}服务健康状态:${NC}"
    local api_port=${API_PORT:-8000}
    
    if curl -f -s "http://localhost:$api_port/health" > /dev/null 2>&1; then
        echo -e "API 服务: ${GREEN}健康${NC}"
        
        # 获取详细健康状态
        local health_response=$(curl -s "http://localhost:$api_port/health" | python3 -m json.tool 2>/dev/null || echo "{}")
        echo "$health_response" | grep -E "(status|service|version)" || true
    else
        echo -e "API 服务: ${RED}不健康${NC}"
    fi
    
    echo -e "\n${CYAN}访问地址:${NC}"
    echo -e "API 文档: ${BLUE}http://localhost:$api_port/docs${NC}"
    echo -e "API 健康检查: ${BLUE}http://localhost:$api_port/health${NC}"
    echo -e "Neo4j 浏览器: ${BLUE}http://localhost:7475${NC}"
    echo -e "MinIO 控制台: ${BLUE}http://localhost:9001${NC}"
}

# 显示日志
show_logs() {
    log_step "显示服务日志..."
    
    if [[ -n "$SERVICE" ]]; then
        docker-compose logs -f "$SERVICE"
    else
        docker-compose logs -f
    fi
}

# 停止服务
stop_services() {
    log_step "停止服务..."
    
    if [[ -n "$SERVICE" ]]; then
        docker-compose stop "$SERVICE"
        log_info "服务 $SERVICE 已停止"
    else
        docker-compose down
        log_info "所有服务已停止"
    fi
}

# 重启服务
restart_services() {
    log_step "重启服务..."
    
    stop_services
    sleep 2
    start_services
    wait_for_services
    
    log_info "服务重启完成"
}

# 清理数据
clean_data() {
    log_step "清理数据..."
    
    read -p "$(echo -e ${RED}警告: 此操作将删除所有数据，是否继续？ [y/N]: ${NC})" -n 1 -r
    echo
    
    if [[ $REPLY =~ ^[Yy]$ ]]; then
        docker-compose down -v --remove-orphans
        docker system prune -f
        log_info "数据清理完成"
    else
        log_info "取消清理操作"
    fi
}

# 主函数
main() {
    # 解析命令行参数
    while [[ $# -gt 0 ]]; do
        case $1 in
            --help|-h)
                show_help
                exit 0
                ;;
            --dev)
                MODE="dev"
                DEBUG="true"
                shift
                ;;
            --prod)
                MODE="prod"
                DEBUG="false"
                shift
                ;;
            --service)
                SERVICE="$2"
                shift 2
                ;;
            --no-build)
                NO_BUILD="true"
                shift
                ;;
            --no-init)
                NO_INIT="true"
                shift
                ;;
            --no-health)
                NO_HEALTH="true"
                shift
                ;;
            --logs)
                SHOW_LOGS="true"
                shift
                ;;
            --stop)
                ACTION="stop"
                shift
                ;;
            --restart)
                ACTION="restart"
                shift
                ;;
            --status)
                ACTION="status"
                shift
                ;;
            --clean)
                ACTION="clean"
                shift
                ;;
            *)
                log_error "未知选项: $1"
                show_help
                exit 1
                ;;
        esac
    done
    
    # 显示横幅
    show_banner
    
    # 执行操作
    case "${ACTION:-start}" in
        stop)
            stop_services
            ;;
        restart)
            restart_services
            ;;
        status)
            show_status
            ;;
        clean)
            clean_data
            ;;
        start|*)
            # 检查依赖
            check_dependencies
            
            # 检查环境文件
            check_env_file
            
            # 创建目录
            create_directories
            
            # 构建服务
            build_services
            
            # 启动服务
            start_services
            
            # 等待服务就绪
            wait_for_services
            
            # 初始化数据库
            initialize_databases
            
            # 显示状态
            show_status
            
            # 显示日志
            if [[ "$SHOW_LOGS" == "true" ]]; then
                show_logs
            fi
            ;;
    esac
    
    log_info "操作完成"
}

# 捕获中断信号
trap 'log_warn "操作被中断"; exit 1' INT TERM

# 执行主函数
main "$@"