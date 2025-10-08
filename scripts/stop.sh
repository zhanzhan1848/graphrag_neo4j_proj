#!/bin/bash

# GraphRAG 知识库系统停止脚本
# ============================
#
# 本脚本用于优雅地停止 GraphRAG 系统及其所有依赖服务
# 包括数据保存、连接关闭、资源清理等功能
#
# 使用方法：
# 1. 停止所有服务：./scripts/stop.sh
# 2. 停止特定服务：./scripts/stop.sh --service api
# 3. 强制停止：./scripts/stop.sh --force
# 4. 停止并清理：./scripts/stop.sh --clean
# 5. 查看帮助：./scripts/stop.sh --help
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
    echo -e "${GREEN}GraphRAG 知识库系统停止脚本${NC}"
    echo -e "${GREEN}版本: 1.0.0${NC}"
    echo -e "${GREEN}作者: GraphRAG Team${NC}"
    echo ""
}

# 显示帮助信息
show_help() {
    cat << EOF
GraphRAG 知识库系统停止脚本

使用方法:
    $0 [选项]

选项:
    --help, -h          显示此帮助信息
    --service SERVICE   停止特定服务 (api, postgres, neo4j, redis, weaviate, minio)
    --force             强制停止所有服务（不等待优雅关闭）
    --clean             停止服务并清理数据卷
    --remove-images     停止服务并删除镜像
    --timeout SECONDS   设置停止超时时间（默认: 30秒）

示例:
    $0                      # 优雅停止所有服务
    $0 --service api        # 只停止 API 服务
    $0 --force              # 强制停止所有服务
    $0 --clean              # 停止并清理数据
    $0 --timeout 60         # 设置60秒超时

环境变量:
    DEBUG=true              启用调试模式
    COMPOSE_FILE            指定 docker-compose 文件路径

EOF
}

# 检查服务状态
check_services() {
    log_step "检查服务状态..."
    
    if ! docker-compose ps --quiet > /dev/null 2>&1; then
        log_warn "没有运行的服务"
        return 1
    fi
    
    local running_services=$(docker-compose ps --services --filter "status=running")
    
    if [[ -z "$running_services" ]]; then
        log_info "没有运行的服务"
        return 1
    fi
    
    log_info "运行中的服务:"
    echo "$running_services" | while read -r service; do
        echo -e "  - ${BLUE}$service${NC}"
    done
    
    return 0
}

# 优雅停止 API 服务
graceful_stop_api() {
    log_step "优雅停止 API 服务..."
    
    local api_port=${API_PORT:-8000}
    local timeout=${TIMEOUT:-30}
    
    # 检查 API 服务是否运行
    if ! curl -f -s "http://localhost:$api_port/health" > /dev/null 2>&1; then
        log_info "API 服务未运行"
        return 0
    fi
    
    # 发送关闭信号
    log_info "发送关闭信号到 API 服务..."
    if curl -f -s -X POST "http://localhost:$api_port/admin/shutdown" > /dev/null 2>&1; then
        log_info "API 服务收到关闭信号"
    else
        log_warn "无法发送关闭信号，将使用 Docker 停止"
    fi
    
    # 等待服务关闭
    local attempt=1
    local max_attempts=$((timeout / 2))
    
    while [[ $attempt -le $max_attempts ]]; do
        if ! curl -f -s "http://localhost:$api_port/health" > /dev/null 2>&1; then
            log_info "API 服务已优雅关闭"
            return 0
        fi
        
        log_debug "等待 API 服务关闭... ($attempt/$max_attempts)"
        sleep 2
        ((attempt++))
    done
    
    log_warn "API 服务未在超时时间内关闭，将强制停止"
    return 1
}

# 保存数据库状态
save_database_state() {
    log_step "保存数据库状态..."
    
    # 保存 PostgreSQL 状态
    if docker-compose ps postgres | grep -q "Up"; then
        log_info "保存 PostgreSQL 数据..."
        docker-compose exec -T postgres pg_dump -U graphrag graphrag > "backups/postgres_$(date +%Y%m%d_%H%M%S).sql" 2>/dev/null || log_warn "PostgreSQL 备份失败"
    fi
    
    # 保存 Neo4j 状态
    if docker-compose ps neo4j | grep -q "Up"; then
        log_info "保存 Neo4j 数据..."
        # Neo4j 数据通过数据卷持久化，无需额外操作
        log_debug "Neo4j 数据已通过数据卷持久化"
    fi
    
    log_info "数据库状态保存完成"
}

# 停止特定服务
stop_service() {
    local service=$1
    local timeout=${TIMEOUT:-30}
    
    log_step "停止服务: $service"
    
    if ! docker-compose ps "$service" | grep -q "Up"; then
        log_info "服务 $service 未运行"
        return 0
    fi
    
    # 特殊处理 API 服务
    if [[ "$service" == "api" ]] && [[ "$FORCE" != "true" ]]; then
        graceful_stop_api || true
    fi
    
    # 停止服务
    if [[ "$FORCE" == "true" ]]; then
        docker-compose kill "$service"
        log_info "强制停止服务: $service"
    else
        docker-compose stop -t "$timeout" "$service"
        log_info "优雅停止服务: $service"
    fi
}

# 停止所有服务
stop_all_services() {
    local timeout=${TIMEOUT:-30}
    
    log_step "停止所有服务..."
    
    # 获取运行中的服务列表
    local running_services=$(docker-compose ps --services --filter "status=running")
    
    if [[ -z "$running_services" ]]; then
        log_info "没有运行的服务需要停止"
        return 0
    fi
    
    # 优雅停止 API 服务
    if echo "$running_services" | grep -q "api" && [[ "$FORCE" != "true" ]]; then
        graceful_stop_api || true
    fi
    
    # 保存数据库状态
    if [[ "$FORCE" != "true" ]]; then
        save_database_state || true
    fi
    
    # 停止所有服务
    if [[ "$FORCE" == "true" ]]; then
        docker-compose kill
        log_info "强制停止所有服务"
    else
        docker-compose stop -t "$timeout"
        log_info "优雅停止所有服务"
    fi
    
    # 移除容器
    if [[ "$CLEAN" == "true" ]]; then
        docker-compose down -v --remove-orphans
        log_info "清理容器和数据卷"
    else
        docker-compose down --remove-orphans
        log_info "移除容器"
    fi
}

# 清理镜像
clean_images() {
    if [[ "$REMOVE_IMAGES" != "true" ]]; then
        return 0
    fi
    
    log_step "清理 Docker 镜像..."
    
    # 获取项目相关的镜像
    local project_images=$(docker images --filter "reference=graphrag*" -q)
    
    if [[ -n "$project_images" ]]; then
        echo "$project_images" | xargs docker rmi -f
        log_info "清理项目镜像完成"
    else
        log_info "没有项目镜像需要清理"
    fi
    
    # 清理悬空镜像
    docker image prune -f > /dev/null 2>&1 || true
    log_info "清理悬空镜像完成"
}

# 显示停止后状态
show_final_status() {
    log_step "显示最终状态..."
    
    echo -e "${CYAN}服务状态:${NC}"
    docker-compose ps || echo "没有运行的容器"
    
    echo -e "\n${CYAN}数据卷状态:${NC}"
    docker volume ls --filter "name=graphrag" || echo "没有相关数据卷"
    
    if [[ "$CLEAN" == "true" ]]; then
        echo -e "\n${YELLOW}注意: 所有数据已被清理${NC}"
    else
        echo -e "\n${GREEN}数据已保留，下次启动时将恢复${NC}"
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
            --service)
                SERVICE="$2"
                shift 2
                ;;
            --force)
                FORCE="true"
                shift
                ;;
            --clean)
                CLEAN="true"
                shift
                ;;
            --remove-images)
                REMOVE_IMAGES="true"
                shift
                ;;
            --timeout)
                TIMEOUT="$2"
                shift 2
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
    
    # 检查服务状态
    if ! check_services; then
        exit 0
    fi
    
    # 确认操作
    if [[ "$CLEAN" == "true" ]] || [[ "$REMOVE_IMAGES" == "true" ]]; then
        echo -e "${RED}警告: 此操作将删除数据，是否继续？ [y/N]: ${NC}"
        read -n 1 -r
        echo
        
        if [[ ! $REPLY =~ ^[Yy]$ ]]; then
            log_info "取消操作"
            exit 0
        fi
    fi
    
    # 执行停止操作
    if [[ -n "$SERVICE" ]]; then
        stop_service "$SERVICE"
    else
        stop_all_services
    fi
    
    # 清理镜像
    clean_images
    
    # 显示最终状态
    show_final_status
    
    log_info "停止操作完成"
}

# 捕获中断信号
trap 'log_warn "操作被中断"; exit 1' INT TERM

# 执行主函数
main "$@"