#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
GraphRAG 测试运行器
==================

提供便捷的测试运行和管理功能：
1. 运行不同类型的测试
2. 生成测试报告
3. 测试覆盖率分析
4. 性能测试
5. 持续集成支持

使用方法:
    python tests/test_runner.py --help
    python tests/test_runner.py --unit
    python tests/test_runner.py --integration
    python tests/test_runner.py --all --coverage

作者: GraphRAG Team
创建时间: 2024
版本: 1.0.0
"""

import os
import sys
import argparse
import subprocess
import time
from pathlib import Path
from typing import List, Optional


class TestRunner:
    """测试运行器"""
    
    def __init__(self, project_root: Optional[Path] = None):
        """初始化测试运行器"""
        self.project_root = project_root or Path(__file__).parent.parent
        self.tests_dir = self.project_root / "tests"
        
        # 确保项目根目录在Python路径中
        if str(self.project_root) not in sys.path:
            sys.path.insert(0, str(self.project_root))
    
    def run_command(self, command: List[str], cwd: Optional[Path] = None) -> int:
        """运行命令并返回退出码"""
        cwd = cwd or self.project_root
        print(f"运行命令: {' '.join(command)}")
        print(f"工作目录: {cwd}")
        print("-" * 50)
        
        try:
            result = subprocess.run(
                command,
                cwd=cwd,
                check=False,
                text=True
            )
            return result.returncode
        except Exception as e:
            print(f"命令执行失败: {e}")
            return 1
    
    def run_unit_tests(self, coverage: bool = False, verbose: bool = False) -> int:
        """运行单元测试"""
        print("🧪 运行单元测试...")
        
        command = ["python", "-m", "pytest"]
        command.extend([str(self.tests_dir / "unit")])
        
        if verbose:
            command.append("-v")
        
        if coverage:
            command.extend([
                "--cov=app",
                "--cov-report=html:htmlcov",
                "--cov-report=term-missing"
            ])
        
        # 添加标记过滤
        command.extend(["-m", "not slow"])
        
        return self.run_command(command)
    
    def run_integration_tests(self, coverage: bool = False, verbose: bool = False) -> int:
        """运行集成测试"""
        print("🔗 运行集成测试...")
        
        command = ["python", "-m", "pytest"]
        command.extend([str(self.tests_dir / "integration")])
        
        if verbose:
            command.append("-v")
        
        if coverage:
            command.extend([
                "--cov=app",
                "--cov-append",
                "--cov-report=html:htmlcov",
                "--cov-report=term-missing"
            ])
        
        # 添加标记过滤
        command.extend(["-m", "integration"])
        
        return self.run_command(command)
    
    def run_performance_tests(self, verbose: bool = False) -> int:
        """运行性能测试"""
        print("⚡ 运行性能测试...")
        
        command = ["python", "-m", "pytest"]
        command.extend([str(self.tests_dir)])
        command.extend(["-m", "performance"])
        
        if verbose:
            command.append("-v")
        
        # 性能测试通常需要更长时间
        command.extend(["--timeout=300"])
        
        return self.run_command(command)
    
    def run_slow_tests(self, verbose: bool = False) -> int:
        """运行慢速测试"""
        print("🐌 运行慢速测试...")
        
        command = ["python", "-m", "pytest"]
        command.extend([str(self.tests_dir)])
        command.extend(["-m", "slow"])
        
        if verbose:
            command.append("-v")
        
        # 慢速测试需要更长超时时间
        command.extend(["--timeout=600"])
        
        return self.run_command(command)
    
    def run_all_tests(self, coverage: bool = False, verbose: bool = False) -> int:
        """运行所有测试"""
        print("🚀 运行所有测试...")
        
        command = ["python", "-m", "pytest"]
        command.extend([str(self.tests_dir)])
        
        if verbose:
            command.append("-v")
        
        if coverage:
            command.extend([
                "--cov=app",
                "--cov-report=html:htmlcov",
                "--cov-report=term-missing",
                "--cov-report=xml"
            ])
        
        return self.run_command(command)
    
    def run_specific_test(self, test_path: str, verbose: bool = False) -> int:
        """运行特定测试"""
        print(f"🎯 运行特定测试: {test_path}")
        
        command = ["python", "-m", "pytest"]
        command.append(test_path)
        
        if verbose:
            command.append("-v")
        
        return self.run_command(command)
    
    def run_tests_by_marker(self, marker: str, verbose: bool = False) -> int:
        """根据标记运行测试"""
        print(f"🏷️  运行标记为 '{marker}' 的测试...")
        
        command = ["python", "-m", "pytest"]
        command.extend([str(self.tests_dir)])
        command.extend(["-m", marker])
        
        if verbose:
            command.append("-v")
        
        return self.run_command(command)
    
    def generate_test_report(self) -> int:
        """生成测试报告"""
        print("📊 生成测试报告...")
        
        command = ["python", "-m", "pytest"]
        command.extend([str(self.tests_dir)])
        command.extend([
            "--html=test_report.html",
            "--self-contained-html",
            "--cov=app",
            "--cov-report=html:htmlcov"
        ])
        
        return self.run_command(command)
    
    def check_test_environment(self) -> bool:
        """检查测试环境"""
        print("🔍 检查测试环境...")
        
        # 检查必要的包
        required_packages = [
            "pytest",
            "pytest-cov",
            "pytest-html",
            "pytest-asyncio",
            "faker"
        ]
        
        missing_packages = []
        for package in required_packages:
            try:
                __import__(package.replace("-", "_"))
            except ImportError:
                missing_packages.append(package)
        
        if missing_packages:
            print(f"❌ 缺少必要的包: {', '.join(missing_packages)}")
            print("请运行: pip install " + " ".join(missing_packages))
            return False
        
        # 检查数据库连接
        try:
            from app.core.database import engine
            with engine.connect() as conn:
                conn.execute("SELECT 1")
            print("✅ 数据库连接正常")
        except Exception as e:
            print(f"❌ 数据库连接失败: {e}")
            return False
        
        print("✅ 测试环境检查通过")
        return True
    
    def clean_test_artifacts(self):
        """清理测试产生的文件"""
        print("🧹 清理测试文件...")
        
        artifacts = [
            self.project_root / "htmlcov",
            self.project_root / "test_report.html",
            self.project_root / "coverage.xml",
            self.project_root / ".coverage",
            self.project_root / ".pytest_cache"
        ]
        
        for artifact in artifacts:
            if artifact.exists():
                if artifact.is_dir():
                    import shutil
                    shutil.rmtree(artifact)
                else:
                    artifact.unlink()
                print(f"删除: {artifact}")
        
        print("✅ 清理完成")


def main():
    """主函数"""
    parser = argparse.ArgumentParser(
        description="GraphRAG 测试运行器",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例:
  %(prog)s --unit                    # 运行单元测试
  %(prog)s --integration             # 运行集成测试
  %(prog)s --all --coverage          # 运行所有测试并生成覆盖率报告
  %(prog)s --performance             # 运行性能测试
  %(prog)s --marker database         # 运行标记为database的测试
  %(prog)s --test tests/unit/test_models/test_base.py  # 运行特定测试
  %(prog)s --report                  # 生成HTML测试报告
  %(prog)s --check                   # 检查测试环境
  %(prog)s --clean                   # 清理测试文件
        """
    )
    
    # 测试类型选项
    test_group = parser.add_mutually_exclusive_group()
    test_group.add_argument("--unit", action="store_true", help="运行单元测试")
    test_group.add_argument("--integration", action="store_true", help="运行集成测试")
    test_group.add_argument("--performance", action="store_true", help="运行性能测试")
    test_group.add_argument("--slow", action="store_true", help="运行慢速测试")
    test_group.add_argument("--all", action="store_true", help="运行所有测试")
    test_group.add_argument("--test", type=str, help="运行特定测试文件或函数")
    test_group.add_argument("--marker", type=str, help="根据标记运行测试")
    test_group.add_argument("--report", action="store_true", help="生成测试报告")
    test_group.add_argument("--check", action="store_true", help="检查测试环境")
    test_group.add_argument("--clean", action="store_true", help="清理测试文件")
    
    # 选项
    parser.add_argument("--coverage", action="store_true", help="生成覆盖率报告")
    parser.add_argument("-v", "--verbose", action="store_true", help="详细输出")
    
    args = parser.parse_args()
    
    # 创建测试运行器
    runner = TestRunner()
    
    # 记录开始时间
    start_time = time.time()
    
    try:
        # 根据参数执行相应操作
        if args.check:
            success = runner.check_test_environment()
            return 0 if success else 1
        
        elif args.clean:
            runner.clean_test_artifacts()
            return 0
        
        elif args.unit:
            return runner.run_unit_tests(coverage=args.coverage, verbose=args.verbose)
        
        elif args.integration:
            return runner.run_integration_tests(coverage=args.coverage, verbose=args.verbose)
        
        elif args.performance:
            return runner.run_performance_tests(verbose=args.verbose)
        
        elif args.slow:
            return runner.run_slow_tests(verbose=args.verbose)
        
        elif args.all:
            return runner.run_all_tests(coverage=args.coverage, verbose=args.verbose)
        
        elif args.test:
            return runner.run_specific_test(args.test, verbose=args.verbose)
        
        elif args.marker:
            return runner.run_tests_by_marker(args.marker, verbose=args.verbose)
        
        elif args.report:
            return runner.generate_test_report()
        
        else:
            # 默认运行单元测试
            print("未指定测试类型，运行单元测试...")
            return runner.run_unit_tests(verbose=args.verbose)
    
    finally:
        # 显示执行时间
        end_time = time.time()
        duration = end_time - start_time
        print(f"\n⏱️  总执行时间: {duration:.2f} 秒")


if __name__ == "__main__":
    sys.exit(main())