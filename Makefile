# GraphRAG 项目 Makefile
# =======================
# 
# 提供便捷的开发和测试命令
# 
# 使用方法:
#   make help          # 显示帮助信息
#   make test          # 运行所有测试
#   make test-unit     # 运行单元测试
#   make test-integration  # 运行集成测试
#   make coverage      # 生成覆盖率报告
#   make lint          # 代码检查
#   make format        # 代码格式化
#   make clean         # 清理临时文件

.PHONY: help test test-unit test-integration test-performance test-slow coverage lint format clean install dev-install docker-build docker-up docker-down docs serve-docs

# 默认目标
.DEFAULT_GOAL := help

# 颜色定义
BLUE := \033[36m
GREEN := \033[32m
YELLOW := \033[33m
RED := \033[31m
RESET := \033[0m

# 项目配置
PROJECT_NAME := GraphRAG
PYTHON := python3
PIP := pip3
PYTEST := python -m pytest
COVERAGE := python -m coverage

help: ## 显示帮助信息
	@echo "$(BLUE)$(PROJECT_NAME) 开发工具$(RESET)"
	@echo ""
	@echo "$(GREEN)可用命令:$(RESET)"
	@awk 'BEGIN {FS = ":.*?## "} /^[a-zA-Z_-]+:.*?## / {printf "  $(YELLOW)%-20s$(RESET) %s\n", $$1, $$2}' $(MAKEFILE_LIST)

# 安装依赖
install: ## 安装生产依赖
	@echo "$(BLUE)安装生产依赖...$(RESET)"
	$(PIP) install -r requirements.txt

dev-install: ## 安装开发依赖
	@echo "$(BLUE)安装开发依赖...$(RESET)"
	$(PIP) install -r requirements.txt
	$(PIP) install pytest pytest-cov pytest-html pytest-asyncio pytest-xdist pytest-benchmark pytest-mock faker

# 测试相关
test: ## 运行所有测试
	@echo "$(BLUE)运行所有测试...$(RESET)"
	$(PYTHON) tests/test_runner.py --all

test-unit: ## 运行单元测试
	@echo "$(BLUE)运行单元测试...$(RESET)"
	$(PYTHON) tests/test_runner.py --unit

test-integration: ## 运行集成测试
	@echo "$(BLUE)运行集成测试...$(RESET)"
	$(PYTHON) tests/test_runner.py --integration

test-performance: ## 运行性能测试
	@echo "$(BLUE)运行性能测试...$(RESET)"
	$(PYTHON) tests/test_runner.py --performance

test-slow: ## 运行慢速测试
	@echo "$(BLUE)运行慢速测试...$(RESET)"
	$(PYTHON) tests/test_runner.py --slow

test-api: ## 运行API测试
	@echo "$(BLUE)运行API测试...$(RESET)"
	$(PYTHON) tests/test_runner.py --marker api

test-database: ## 运行数据库测试
	@echo "$(BLUE)运行数据库测试...$(RESET)"
	$(PYTHON) tests/test_runner.py --marker database

test-watch: ## 监视文件变化并自动运行测试
	@echo "$(BLUE)监视模式运行测试...$(RESET)"
	$(PYTEST) --looponfail tests/

# 覆盖率相关
coverage: ## 生成覆盖率报告
	@echo "$(BLUE)生成覆盖率报告...$(RESET)"
	$(PYTHON) tests/test_runner.py --all --coverage
	@echo "$(GREEN)覆盖率报告已生成: htmlcov/index.html$(RESET)"

coverage-report: ## 显示覆盖率报告
	@echo "$(BLUE)显示覆盖率报告...$(RESET)"
	$(COVERAGE) report

coverage-html: ## 生成HTML覆盖率报告
	@echo "$(BLUE)生成HTML覆盖率报告...$(RESET)"
	$(COVERAGE) html
	@echo "$(GREEN)HTML报告已生成: htmlcov/index.html$(RESET)"

# 代码质量
lint: ## 代码检查
	@echo "$(BLUE)运行代码检查...$(RESET)"
	@if command -v flake8 >/dev/null 2>&1; then \
		echo "运行 flake8..."; \
		flake8 app tests; \
	else \
		echo "$(YELLOW)flake8 未安装，跳过检查$(RESET)"; \
	fi
	@if command -v pylint >/dev/null 2>&1; then \
		echo "运行 pylint..."; \
		pylint app; \
	else \
		echo "$(YELLOW)pylint 未安装，跳过检查$(RESET)"; \
	fi

format: ## 代码格式化
	@echo "$(BLUE)格式化代码...$(RESET)"
	@if command -v black >/dev/null 2>&1; then \
		echo "运行 black..."; \
		black app tests; \
	else \
		echo "$(YELLOW)black 未安装，跳过格式化$(RESET)"; \
	fi
	@if command -v isort >/dev/null 2>&1; then \
		echo "运行 isort..."; \
		isort app tests; \
	else \
		echo "$(YELLOW)isort 未安装，跳过导入排序$(RESET)"; \
	fi

# 环境检查
check-env: ## 检查测试环境
	@echo "$(BLUE)检查测试环境...$(RESET)"
	$(PYTHON) tests/test_runner.py --check

# 清理
clean: ## 清理临时文件
	@echo "$(BLUE)清理临时文件...$(RESET)"
	$(PYTHON) tests/test_runner.py --clean
	find . -type f -name "*.pyc" -delete
	find . -type d -name "__pycache__" -exec rm -rf {} + 2>/dev/null || true
	find . -type d -name "*.egg-info" -exec rm -rf {} + 2>/dev/null || true
	rm -rf build/ dist/ .eggs/
	rm -f .coverage coverage.xml tests.log
	@echo "$(GREEN)清理完成$(RESET)"

clean-all: clean ## 深度清理（包括虚拟环境）
	@echo "$(BLUE)深度清理...$(RESET)"
	rm -rf venv/ env/ .venv/
	rm -rf node_modules/
	@echo "$(GREEN)深度清理完成$(RESET)"

# Docker 相关
docker-build: ## 构建Docker镜像
	@echo "$(BLUE)构建Docker镜像...$(RESET)"
	docker build -t graphrag:latest .

docker-up: ## 启动Docker服务
	@echo "$(BLUE)启动Docker服务...$(RESET)"
	docker-compose up -d

docker-down: ## 停止Docker服务
	@echo "$(BLUE)停止Docker服务...$(RESET)"
	docker-compose down

docker-logs: ## 查看Docker日志
	@echo "$(BLUE)查看Docker日志...$(RESET)"
	docker-compose logs -f

# 数据库相关
db-init: ## 初始化数据库
	@echo "$(BLUE)初始化数据库...$(RESET)"
	$(PYTHON) scripts/init_neo4j.py

db-reset: ## 重置数据库
	@echo "$(BLUE)重置数据库...$(RESET)"
	@echo "$(RED)警告: 这将删除所有数据!$(RESET)"
	@read -p "确认继续? [y/N] " confirm && [ "$$confirm" = "y" ]
	docker-compose down -v
	docker-compose up -d

# 开发服务器
dev: ## 启动开发服务器
	@echo "$(BLUE)启动开发服务器...$(RESET)"
	$(PYTHON) -m uvicorn app.main:app --reload --host 0.0.0.0 --port 8000

dev-debug: ## 启动调试模式开发服务器
	@echo "$(BLUE)启动调试模式开发服务器...$(RESET)"
	$(PYTHON) -m uvicorn app.main:app --reload --host 0.0.0.0 --port 8000 --log-level debug

# 文档相关
docs: ## 生成API文档
	@echo "$(BLUE)生成API文档...$(RESET)"
	@if command -v sphinx-build >/dev/null 2>&1; then \
		sphinx-build -b html docs/ docs/_build/html; \
	else \
		echo "$(YELLOW)Sphinx 未安装，跳过文档生成$(RESET)"; \
	fi

serve-docs: ## 启动文档服务器
	@echo "$(BLUE)启动文档服务器...$(RESET)"
	@if [ -d "docs/_build/html" ]; then \
		cd docs/_build/html && $(PYTHON) -m http.server 8080; \
	else \
		echo "$(RED)文档未生成，请先运行 make docs$(RESET)"; \
	fi

# 性能分析
profile: ## 运行性能分析
	@echo "$(BLUE)运行性能分析...$(RESET)"
	$(PYTHON) -m cProfile -o profile.stats -m pytest tests/unit/ -m performance
	@if command -v snakeviz >/dev/null 2>&1; then \
		snakeviz profile.stats; \
	else \
		echo "$(YELLOW)snakeviz 未安装，无法可视化性能数据$(RESET)"; \
	fi

# 安全检查
security: ## 安全检查
	@echo "$(BLUE)运行安全检查...$(RESET)"
	@if command -v bandit >/dev/null 2>&1; then \
		bandit -r app/; \
	else \
		echo "$(YELLOW)bandit 未安装，跳过安全检查$(RESET)"; \
	fi
	@if command -v safety >/dev/null 2>&1; then \
		safety check; \
	else \
		echo "$(YELLOW)safety 未安装，跳过依赖安全检查$(RESET)"; \
	fi

# 发布相关
build: ## 构建分发包
	@echo "$(BLUE)构建分发包...$(RESET)"
	$(PYTHON) setup.py sdist bdist_wheel

release: build ## 发布到PyPI
	@echo "$(BLUE)发布到PyPI...$(RESET)"
	@if command -v twine >/dev/null 2>&1; then \
		twine upload dist/*; \
	else \
		echo "$(RED)twine 未安装，无法发布$(RESET)"; \
	fi

# 快捷命令
quick-test: test-unit ## 快速测试（单元测试）

full-test: test coverage lint ## 完整测试（测试+覆盖率+代码检查）

ci: clean test coverage lint ## CI流水线命令

# 状态信息
status: ## 显示项目状态
	@echo "$(BLUE)项目状态:$(RESET)"
	@echo "Python版本: $(shell $(PYTHON) --version)"
	@echo "项目目录: $(shell pwd)"
	@echo "Git分支: $(shell git branch --show-current 2>/dev/null || echo '未知')"
	@echo "Git提交: $(shell git rev-parse --short HEAD 2>/dev/null || echo '未知')"
	@echo "虚拟环境: $(shell echo $$VIRTUAL_ENV || echo '未激活')"