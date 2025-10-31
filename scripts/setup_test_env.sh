#!/bin/bash
# GraphRAG 测试环境设置脚本
# ============================
#
# 自动设置和配置测试环境，包括数据库初始化、依赖安装等
#
# 使用方法:
#   ./scripts/setup_test_env.sh
#   ./scripts/setup_test_env.sh --reset
#   ./scripts/setup_test_env.sh --docker
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

# 项目配置
PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PYTHON_CMD="python3"
PIP_CMD="pip3"

# 默认配置
RESET_ENV=false
USE_DOCKER=false
INSTALL_DEPS=true
INIT_DB=true
CREATE_TEST_DATA=false

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

log_header() {
    echo -e "${PURPLE}================================${NC}"
    echo -e "${PURPLE} $1${NC}"
    echo -e "${PURPLE}================================${NC}"
}

# 显示帮助信息
show_help() {
    cat << EOF
GraphRAG 测试环境设置脚本

使用方法:
    $0 [选项]

选项:
    --reset             重置测试环境
    --docker            使用Docker环境
    --no-deps           跳过依赖安装
    --no-db             跳过数据库初始化
    --test-data         创建测试数据
    -h, --help          显示此帮助信息

示例:
    $0                  # 标准设置
    $0 --reset          # 重置环境
    $0 --docker         # 使用Docker
    $0 --test-data      # 包含测试数据

EOF
}

# 检查系统要求
check_system_requirements() {
    log_header "检查系统要求"
    
    # 检查Python
    if ! command -v $PYTHON_CMD &> /dev/null; then
        log_error "Python3 未找到，请先安装Python3"
        exit 1
    fi
    
    local python_version=$($PYTHON_CMD --version 2>&1 | cut -d' ' -f2)
    log_info "Python版本: $python_version"
    
    # 检查pip
    if ! command -v $PIP_CMD &> /dev/null; then
        log_error "pip3 未找到，请先安装pip3"
        exit 1
    fi
    
    # 检查Docker（如果需要）
    if [ "$USE_DOCKER" = true ]; then
        if ! command -v docker &> /dev/null; then
            log_error "Docker 未找到，请先安装Docker"
            exit 1
        fi
        
        if ! command -v docker-compose &> /dev/null; then
            log_error "docker-compose 未找到，请先安装docker-compose"
            exit 1
        fi
    fi
    
    log_success "系统要求检查完成"
}

# 创建虚拟环境
setup_virtual_environment() {
    log_header "设置虚拟环境"
    
    cd "$PROJECT_ROOT"
    
    # 检查是否已在虚拟环境中
    if [[ "$VIRTUAL_ENV" != "" ]]; then
        log_info "已在虚拟环境中: $VIRTUAL_ENV"
        return 0
    fi
    
    # 创建虚拟环境
    if [ ! -d "venv" ]; then
        log_info "创建虚拟环境..."
        $PYTHON_CMD -m venv venv
    fi
    
    # 激活虚拟环境
    log_info "激活虚拟环境..."
    source venv/bin/activate
    
    log_success "虚拟环境设置完成"
}

# 安装依赖包
install_dependencies() {
    if [ "$INSTALL_DEPS" = false ]; then
        log_info "跳过依赖安装"
        return 0
    fi
    
    log_header "安装依赖包"
    
    cd "$PROJECT_ROOT"
    
    # 升级pip
    log_info "升级pip..."
    $PIP_CMD install --upgrade pip
    
    # 安装生产依赖
    if [ -f "requirements.txt" ]; then
        log_info "安装生产依赖..."
        $PIP_CMD install -r requirements.txt
    fi
    
    # 安装开发依赖
    if [ -f "requirements-dev.txt" ]; then
        log_info "安装开发依赖..."
        $PIP_CMD install -r requirements-dev.txt
    fi
    
    # 安装测试依赖
    if [ -f "requirements-test.txt" ]; then
        log_info "安装测试依赖..."
        $PIP_CMD install -r requirements-test.txt
    else
        # 如果没有测试依赖文件，安装基本测试包
        log_info "安装基本测试依赖..."
        $PIP_CMD install pytest pytest-cov pytest-html pytest-asyncio pytest-mock pytest-xdist
    fi
    
    log_success "依赖包安装完成"
}

# 设置环境变量
setup_environment_variables() {
    log_header "设置环境变量"
    
    cd "$PROJECT_ROOT"
    
    # 创建测试环境配置文件
    if [ ! -f ".env.test" ]; then
        log_info "创建测试环境配置文件..."
        cat > .env.test << EOF
# GraphRAG 测试环境配置
# =====================

# 数据库配置
DATABASE_URL=postgresql://test_user:test_password@localhost:5432/graphrag_test
NEO4J_URI=bolt://localhost:7687
NEO4J_USER=neo4j
NEO4J_PASSWORD=test_password

# Redis配置
REDIS_URL=redis://localhost:6379/1

# 测试配置
TESTING=true
DEBUG=true
LOG_LEVEL=DEBUG

# API配置
API_HOST=127.0.0.1
API_PORT=8001

# 文件存储
UPLOAD_DIR=./test_uploads
MAX_FILE_SIZE=100MB

# 嵌入模型配置
EMBEDDING_MODEL=sentence-transformers/all-MiniLM-L6-v2
EMBEDDING_DIMENSION=384

# LLM配置
LLM_PROVIDER=openai
OPENAI_API_KEY=test_key_for_testing

# 测试数据配置
TEST_DATA_DIR=./tests/fixtures/data
CLEANUP_TEST_DATA=true

EOF
        log_success "测试环境配置文件已创建"
    else
        log_info "测试环境配置文件已存在"
    fi
}

# 启动Docker服务
start_docker_services() {
    if [ "$USE_DOCKER" = false ]; then
        return 0
    fi
    
    log_header "启动Docker服务"
    
    cd "$PROJECT_ROOT"
    
    # 检查docker-compose文件
    if [ ! -f "docker-compose.test.yml" ]; then
        log_info "创建测试Docker配置..."
        cat > docker-compose.test.yml << EOF
version: '3.8'

services:
  postgres-test:
    image: postgres:15
    environment:
      POSTGRES_DB: graphrag_test
      POSTGRES_USER: test_user
      POSTGRES_PASSWORD: test_password
    ports:
      - "5433:5432"
    volumes:
      - postgres_test_data:/var/lib/postgresql/data
    healthcheck:
      test: ["CMD-SHELL", "pg_isready -U test_user -d graphrag_test"]
      interval: 10s
      timeout: 5s
      retries: 5

  neo4j-test:
    image: neo4j:5.15
    environment:
      NEO4J_AUTH: neo4j/test_password
      NEO4J_PLUGINS: '["apoc"]'
      NEO4J_dbms_security_procedures_unrestricted: apoc.*
    ports:
      - "7688:7687"
      - "7475:7474"
    volumes:
      - neo4j_test_data:/data
    healthcheck:
      test: ["CMD", "cypher-shell", "-u", "neo4j", "-p", "test_password", "RETURN 1"]
      interval: 10s
      timeout: 5s
      retries: 5

  redis-test:
    image: redis:7-alpine
    ports:
      - "6380:6379"
    healthcheck:
      test: ["CMD", "redis-cli", "ping"]
      interval: 10s
      timeout: 5s
      retries: 5

volumes:
  postgres_test_data:
  neo4j_test_data:
EOF
    fi
    
    # 启动服务
    log_info "启动测试数据库服务..."
    docker-compose -f docker-compose.test.yml up -d
    
    # 等待服务就绪
    log_info "等待服务启动..."
    sleep 30
    
    # 检查服务状态
    if docker-compose -f docker-compose.test.yml ps | grep -q "Up"; then
        log_success "Docker服务启动完成"
    else
        log_error "Docker服务启动失败"
        docker-compose -f docker-compose.test.yml logs
        exit 1
    fi
}

# 初始化数据库
initialize_database() {
    if [ "$INIT_DB" = false ]; then
        log_info "跳过数据库初始化"
        return 0
    fi
    
    log_header "初始化数据库"
    
    cd "$PROJECT_ROOT"
    
    # 设置环境变量
    export TESTING=true
    
    # 运行数据库迁移
    log_info "运行数据库迁移..."
    if [ -f "alembic.ini" ]; then
        $PYTHON_CMD -m alembic upgrade head
    else
        # 如果没有alembic，直接创建表
        $PYTHON_CMD -c "
import sys
sys.path.insert(0, '.')
from app.core.database import engine, Base
from app.models.database import *
Base.metadata.create_all(bind=engine)
print('数据库表创建完成')
"
    fi
    
    # 初始化Neo4j约束和索引
    log_info "初始化Neo4j约束和索引..."
    $PYTHON_CMD -c "
import sys
sys.path.insert(0, '.')
from app.core.neo4j_client import Neo4jClient
from app.services.graph_index_service import GraphIndexService

try:
    client = Neo4jClient()
    index_service = GraphIndexService(client)
    index_service.create_indexes()
    index_service.create_constraints()
    print('Neo4j索引和约束创建完成')
except Exception as e:
    print(f'Neo4j初始化失败: {e}')
"
    
    log_success "数据库初始化完成"
}

# 创建测试数据
create_test_data() {
    if [ "$CREATE_TEST_DATA" = false ]; then
        return 0
    fi
    
    log_header "创建测试数据"
    
    cd "$PROJECT_ROOT"
    
    # 创建测试数据目录
    mkdir -p tests/fixtures/data
    
    # 运行测试数据生成脚本
    log_info "生成测试数据..."
    $PYTHON_CMD -c "
import sys
sys.path.insert(0, '.')
from tests.fixtures.test_data import TestDataFactory
from app.core.database import SessionLocal

with SessionLocal() as session:
    factory = TestDataFactory(session)
    
    # 创建示例文档
    doc = factory.create_document(
        title='测试文档',
        content='这是一个用于测试的示例文档。',
        file_type='text'
    )
    
    # 创建示例实体
    entity = factory.create_entity(
        name='测试实体',
        entity_type='PERSON',
        document_id=doc.id
    )
    
    session.commit()
    print('测试数据创建完成')
"
    
    log_success "测试数据创建完成"
}

# 验证测试环境
verify_test_environment() {
    log_header "验证测试环境"
    
    cd "$PROJECT_ROOT"
    
    # 运行基本测试
    log_info "运行环境验证测试..."
    $PYTHON_CMD -m pytest tests/ -k "test_database_connection or test_neo4j_connection" -v --tb=short
    
    if [ $? -eq 0 ]; then
        log_success "测试环境验证通过"
    else
        log_error "测试环境验证失败"
        return 1
    fi
}

# 重置测试环境
reset_test_environment() {
    log_header "重置测试环境"
    
    cd "$PROJECT_ROOT"
    
    # 停止Docker服务
    if [ "$USE_DOCKER" = true ]; then
        log_info "停止Docker服务..."
        docker-compose -f docker-compose.test.yml down -v
    fi
    
    # 清理数据库
    log_info "清理测试数据库..."
    $PYTHON_CMD -c "
import sys
sys.path.insert(0, '.')
try:
    from app.core.database import engine, Base
    Base.metadata.drop_all(bind=engine)
    print('PostgreSQL数据库清理完成')
except Exception as e:
    print(f'PostgreSQL清理失败: {e}')

try:
    from app.core.neo4j_client import Neo4jClient
    client = Neo4jClient()
    with client.driver.session() as session:
        session.run('MATCH (n) DETACH DELETE n')
    print('Neo4j数据库清理完成')
except Exception as e:
    print(f'Neo4j清理失败: {e}')
"
    
    # 清理测试文件
    rm -rf htmlcov/ .coverage coverage.xml test_report.html
    rm -rf tests/fixtures/data/*
    rm -rf test_uploads/
    
    log_success "测试环境重置完成"
}

# 显示设置摘要
show_setup_summary() {
    log_header "设置摘要"
    
    echo -e "${CYAN}测试环境设置完成！${NC}"
    echo ""
    echo -e "${YELLOW}可用命令:${NC}"
    echo "  ./scripts/run_tests.sh --unit      # 运行单元测试"
    echo "  ./scripts/run_tests.sh --all       # 运行所有测试"
    echo "  ./scripts/run_tests.sh --coverage  # 生成覆盖率报告"
    echo "  make test                          # 使用Makefile运行测试"
    echo ""
    echo -e "${YELLOW}配置文件:${NC}"
    echo "  .env.test                          # 测试环境配置"
    echo "  pytest.ini                         # pytest配置"
    echo "  .coveragerc                        # 覆盖率配置"
    echo ""
    echo -e "${YELLOW}测试目录:${NC}"
    echo "  tests/unit/                        # 单元测试"
    echo "  tests/integration/                 # 集成测试"
    echo "  tests/fixtures/                    # 测试数据和fixtures"
    echo ""
}

# 主函数
main() {
    # 解析命令行参数
    while [[ $# -gt 0 ]]; do
        case $1 in
            --reset)
                RESET_ENV=true
                shift
                ;;
            --docker)
                USE_DOCKER=true
                shift
                ;;
            --no-deps)
                INSTALL_DEPS=false
                shift
                ;;
            --no-db)
                INIT_DB=false
                shift
                ;;
            --test-data)
                CREATE_TEST_DATA=true
                shift
                ;;
            -h|--help)
                show_help
                exit 0
                ;;
            *)
                log_error "未知选项: $1"
                show_help
                exit 1
                ;;
        esac
    done
    
    # 重置环境（如果需要）
    if [ "$RESET_ENV" = true ]; then
        reset_test_environment
    fi
    
    # 执行设置步骤
    check_system_requirements
    setup_virtual_environment
    install_dependencies
    setup_environment_variables
    start_docker_services
    initialize_database
    create_test_data
    
    # 验证环境
    if verify_test_environment; then
        show_setup_summary
        log_success "测试环境设置完成！"
        exit 0
    else
        log_error "测试环境设置失败"
        exit 1
    fi
}

# 脚本入口
main "$@"