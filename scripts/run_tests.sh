#!/bin/bash
# GraphRAG 测试运行脚本
# ======================
#
# 提供便捷的测试运行功能，支持多种测试场景
#
# 使用方法:
#   ./scripts/run_tests.sh --help
#   ./scripts/run_tests.sh --unit
#   ./scripts/run_tests.sh --integration
#   ./scripts/run_tests.sh --all --coverage
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
PYTEST_CMD="python -m pytest"

# 默认配置
VERBOSE=false
COVERAGE=false
PARALLEL=false
REPORT=false
CLEAN=false
CHECK_ENV=false

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
GraphRAG 测试运行脚本

使用方法:
    $0 [选项] [测试类型]

测试类型:
    --unit              运行单元测试
    --integration       运行集成测试
    --performance       运行性能测试
    --slow              运行慢速测试
    --all               运行所有测试
    --api               运行API测试
    --database          运行数据库测试
    --marker MARKER     运行指定标记的测试

选项:
    -v, --verbose       详细输出
    -c, --coverage      生成覆盖率报告
    -p, --parallel      并行运行测试
    -r, --report        生成HTML报告
    --clean             清理测试文件
    --check             检查测试环境
    -h, --help          显示此帮助信息

示例:
    $0 --unit --verbose
    $0 --all --coverage --report
    $0 --marker database --verbose
    $0 --check
    $0 --clean

EOF
}

# 检查Python环境
check_python() {
    if ! command -v $PYTHON_CMD &> /dev/null; then
        log_error "Python3 未找到，请确保已安装Python3"
        exit 1
    fi
    
    local python_version=$($PYTHON_CMD --version 2>&1 | cut -d' ' -f2)
    log_info "Python版本: $python_version"
}

# 检查依赖包
check_dependencies() {
    log_info "检查测试依赖..."
    
    local required_packages=("pytest" "pytest-cov" "pytest-html" "pytest-asyncio")
    local missing_packages=()
    
    for package in "${required_packages[@]}"; do
        if ! $PYTHON_CMD -c "import ${package//-/_}" &> /dev/null; then
            missing_packages+=("$package")
        fi
    done
    
    if [ ${#missing_packages[@]} -ne 0 ]; then
        log_error "缺少必要的包: ${missing_packages[*]}"
        log_info "请运行: pip install ${missing_packages[*]}"
        return 1
    fi
    
    log_success "所有依赖包已安装"
    return 0
}

# 检查数据库连接
check_database() {
    log_info "检查数据库连接..."
    
    if $PYTHON_CMD -c "
import sys
sys.path.insert(0, '$PROJECT_ROOT')
try:
    from app.core.database import engine
    with engine.connect() as conn:
        conn.execute('SELECT 1')
    print('数据库连接正常')
except Exception as e:
    print(f'数据库连接失败: {e}')
    sys.exit(1)
" 2>/dev/null; then
        log_success "数据库连接正常"
        return 0
    else
        log_error "数据库连接失败"
        return 1
    fi
}

# 检查测试环境
check_environment() {
    log_header "检查测试环境"
    
    check_python
    
    if ! check_dependencies; then
        exit 1
    fi
    
    if ! check_database; then
        log_warning "数据库连接失败，某些测试可能无法运行"
    fi
    
    log_success "测试环境检查完成"
}

# 清理测试文件
clean_test_files() {
    log_header "清理测试文件"
    
    cd "$PROJECT_ROOT"
    
    # 清理覆盖率文件
    rm -rf htmlcov/
    rm -f .coverage coverage.xml
    
    # 清理测试报告
    rm -f test_report.html tests.log
    
    # 清理Python缓存
    find . -type d -name "__pycache__" -exec rm -rf {} + 2>/dev/null || true
    find . -type f -name "*.pyc" -delete 2>/dev/null || true
    
    # 清理pytest缓存
    rm -rf .pytest_cache/
    
    log_success "测试文件清理完成"
}

# 构建pytest命令
build_pytest_command() {
    local test_type="$1"
    local cmd="$PYTEST_CMD"
    
    # 基本选项
    if [ "$VERBOSE" = true ]; then
        cmd="$cmd -v"
    fi
    
    # 覆盖率选项
    if [ "$COVERAGE" = true ]; then
        cmd="$cmd --cov=app --cov-report=term-missing"
        if [ "$REPORT" = true ]; then
            cmd="$cmd --cov-report=html:htmlcov"
        fi
    fi
    
    # 并行选项
    if [ "$PARALLEL" = true ]; then
        cmd="$cmd -n auto"
    fi
    
    # HTML报告
    if [ "$REPORT" = true ]; then
        cmd="$cmd --html=test_report.html --self-contained-html"
    fi
    
    # 测试路径和标记
    case "$test_type" in
        "unit")
            cmd="$cmd tests/unit/"
            ;;
        "integration")
            cmd="$cmd tests/integration/ -m integration"
            ;;
        "performance")
            cmd="$cmd tests/ -m performance"
            ;;
        "slow")
            cmd="$cmd tests/ -m slow"
            ;;
        "api")
            cmd="$cmd tests/ -m api"
            ;;
        "database")
            cmd="$cmd tests/ -m database"
            ;;
        "all")
            cmd="$cmd tests/"
            ;;
        *)
            cmd="$cmd tests/ -m $test_type"
            ;;
    esac
    
    echo "$cmd"
}

# 运行测试
run_tests() {
    local test_type="$1"
    local test_name=""
    
    case "$test_type" in
        "unit") test_name="单元测试" ;;
        "integration") test_name="集成测试" ;;
        "performance") test_name="性能测试" ;;
        "slow") test_name="慢速测试" ;;
        "api") test_name="API测试" ;;
        "database") test_name="数据库测试" ;;
        "all") test_name="所有测试" ;;
        *) test_name="标记为 $test_type 的测试" ;;
    esac
    
    log_header "运行 $test_name"
    
    cd "$PROJECT_ROOT"
    
    local cmd=$(build_pytest_command "$test_type")
    log_info "执行命令: $cmd"
    
    local start_time=$(date +%s)
    
    if eval "$cmd"; then
        local end_time=$(date +%s)
        local duration=$((end_time - start_time))
        log_success "$test_name 完成 (耗时: ${duration}秒)"
        
        # 显示报告位置
        if [ "$COVERAGE" = true ] && [ "$REPORT" = true ]; then
            log_info "覆盖率报告: htmlcov/index.html"
        fi
        
        if [ "$REPORT" = true ]; then
            log_info "测试报告: test_report.html"
        fi
        
        return 0
    else
        local end_time=$(date +%s)
        local duration=$((end_time - start_time))
        log_error "$test_name 失败 (耗时: ${duration}秒)"
        return 1
    fi
}

# 显示测试统计
show_test_stats() {
    if [ -f ".coverage" ]; then
        log_info "生成覆盖率统计..."
        $PYTHON_CMD -m coverage report --show-missing
    fi
}

# 主函数
main() {
    local test_type=""
    local custom_marker=""
    
    # 解析命令行参数
    while [[ $# -gt 0 ]]; do
        case $1 in
            --unit)
                test_type="unit"
                shift
                ;;
            --integration)
                test_type="integration"
                shift
                ;;
            --performance)
                test_type="performance"
                shift
                ;;
            --slow)
                test_type="slow"
                shift
                ;;
            --all)
                test_type="all"
                shift
                ;;
            --api)
                test_type="api"
                shift
                ;;
            --database)
                test_type="database"
                shift
                ;;
            --marker)
                test_type="$2"
                shift 2
                ;;
            -v|--verbose)
                VERBOSE=true
                shift
                ;;
            -c|--coverage)
                COVERAGE=true
                shift
                ;;
            -p|--parallel)
                PARALLEL=true
                shift
                ;;
            -r|--report)
                REPORT=true
                shift
                ;;
            --clean)
                CLEAN=true
                shift
                ;;
            --check)
                CHECK_ENV=true
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
    
    # 执行相应操作
    if [ "$CLEAN" = true ]; then
        clean_test_files
        exit 0
    fi
    
    if [ "$CHECK_ENV" = true ]; then
        check_environment
        exit 0
    fi
    
    # 如果没有指定测试类型，默认运行单元测试
    if [ -z "$test_type" ]; then
        test_type="unit"
        log_warning "未指定测试类型，默认运行单元测试"
    fi
    
    # 检查环境
    check_environment
    
    # 运行测试
    if run_tests "$test_type"; then
        show_test_stats
        log_success "测试运行完成"
        exit 0
    else
        log_error "测试运行失败"
        exit 1
    fi
}

# 脚本入口
main "$@"