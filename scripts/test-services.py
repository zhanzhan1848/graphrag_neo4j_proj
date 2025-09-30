#!/usr/bin/env python3
"""
GraphRAG 知识库系统 - 基础服务健康检查脚本 (Python 版本)

功能: 测试 PostgreSQL、Neo4j、Redis、MinIO、Weaviate、MinerU 等服务是否正常运行
作者: GraphRAG Team
使用方法: python scripts/test-services.py
"""

import os
import sys
import time
import json
import logging
import requests
import psycopg2
import redis
from neo4j import GraphDatabase
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass
from datetime import datetime

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
logger = logging.getLogger(__name__)

@dataclass
class ServiceConfig:
    """服务配置数据类"""
    name: str
    host: str
    port: int
    timeout: int = 10
    credentials: Optional[Dict] = None

@dataclass
class TestResult:
    """测试结果数据类"""
    service_name: str
    success: bool
    message: str
    response_time: float
    version: Optional[str] = None
    details: Optional[Dict] = None

class ServiceTester:
    """基础服务测试器"""
    
    def __init__(self):
        """初始化测试器，加载配置"""
        self.load_config()
        self.results: List[TestResult] = []
    
    def load_config(self):
        """从环境变量加载服务配置"""
        self.services = {
            'postgres': ServiceConfig(
                name='PostgreSQL',
                host=os.getenv('POSTGRES_HOST', 'localhost'),
                port=int(os.getenv('POSTGRES_PORT', 5432)),
                credentials={
                    'database': os.getenv('POSTGRES_DB', 'graphrag'),
                    'user': os.getenv('POSTGRES_USER', 'graphrag'),
                    'password': os.getenv('POSTGRES_PASSWORD', 'graphrag123')
                }
            ),
            'neo4j': ServiceConfig(
                name='Neo4j',
                host=os.getenv('NEO4J_HOST', 'localhost'),
                port=int(os.getenv('NEO4J_HTTP_PORT', 7474)),
                credentials={
                    'user': os.getenv('NEO4J_USER', 'neo4j'),
                    'password': os.getenv('NEO4J_PASSWORD', 'neo4j123'),
                    'bolt_port': int(os.getenv('NEO4J_BOLT_PORT', 7687))
                }
            ),
            'redis': ServiceConfig(
                name='Redis',
                host=os.getenv('REDIS_HOST', 'localhost'),
                port=int(os.getenv('REDIS_PORT', 6379)),
                credentials={
                    'password': os.getenv('REDIS_PASSWORD', 'redis123'),
                    'db': int(os.getenv('REDIS_DB', 0))
                }
            ),
            'minio': ServiceConfig(
                name='MinIO',
                host=os.getenv('MINIO_HOST', 'localhost'),
                port=int(os.getenv('MINIO_PORT', 9000)),
                credentials={
                    'access_key': os.getenv('MINIO_ROOT_USER', 'minioadmin'),
                    'secret_key': os.getenv('MINIO_ROOT_PASSWORD', 'minioadmin123')
                }
            ),
            'weaviate': ServiceConfig(
                name='Weaviate',
                host=os.getenv('WEAVIATE_HOST', 'localhost'),
                port=int(os.getenv('WEAVIATE_PORT', 8080))
            ),
            'mineru': ServiceConfig(
                name='MinerU',
                host=os.getenv('MINERU_HOST', 'localhost'),
                port=int(os.getenv('MINERU_PORT', 8501))
            )
        }
    
    def test_postgres(self) -> TestResult:
        """测试 PostgreSQL 连接"""
        config = self.services['postgres']
        start_time = time.time()
        
        try:
            # 建立连接
            conn = psycopg2.connect(
                host=config.host,
                port=config.port,
                database=config.credentials['database'],
                user=config.credentials['user'],
                password=config.credentials['password'],
                connect_timeout=config.timeout
            )
            
            # 执行测试查询
            cursor = conn.cursor()
            cursor.execute("SELECT version();")
            version_info = cursor.fetchone()[0]
            
            # 获取数据库统计信息
            cursor.execute("""
                SELECT 
                    count(*) as table_count
                FROM information_schema.tables 
                WHERE table_schema = 'public';
            """)
            table_count = cursor.fetchone()[0]
            
            cursor.close()
            conn.close()
            
            response_time = time.time() - start_time
            
            return TestResult(
                service_name=config.name,
                success=True,
                message="连接成功",
                response_time=response_time,
                version=version_info.split()[1],
                details={'table_count': table_count}
            )
            
        except Exception as e:
            response_time = time.time() - start_time
            return TestResult(
                service_name=config.name,
                success=False,
                message=f"连接失败: {str(e)}",
                response_time=response_time
            )
    
    def test_neo4j(self) -> TestResult:
        """测试 Neo4j 连接"""
        config = self.services['neo4j']
        start_time = time.time()
        
        try:
            # 测试 HTTP 接口
            http_url = f"http://{config.host}:{config.port}/db/data/"
            auth = (config.credentials['user'], config.credentials['password'])
            
            response = requests.get(http_url, auth=auth, timeout=config.timeout)
            response.raise_for_status()
            
            # 测试 Bolt 连接
            bolt_uri = f"bolt://{config.host}:{config.credentials['bolt_port']}"
            driver = GraphDatabase.driver(
                bolt_uri,
                auth=(config.credentials['user'], config.credentials['password'])
            )
            
            # 执行测试查询
            with driver.session() as session:
                result = session.run("CALL dbms.components() YIELD name, versions, edition")
                components = list(result)
                
                # 获取节点和关系统计
                stats_result = session.run("""
                    CALL apoc.meta.stats() YIELD nodeCount, relCount
                    RETURN nodeCount, relCount
                """)
                stats = stats_result.single()
                
            driver.close()
            response_time = time.time() - start_time
            
            version = components[0]['versions'][0] if components else "Unknown"
            
            return TestResult(
                service_name=config.name,
                success=True,
                message="连接成功",
                response_time=response_time,
                version=version,
                details={
                    'node_count': stats['nodeCount'] if stats else 0,
                    'relationship_count': stats['relCount'] if stats else 0
                }
            )
            
        except Exception as e:
            response_time = time.time() - start_time
            return TestResult(
                service_name=config.name,
                success=False,
                message=f"连接失败: {str(e)}",
                response_time=response_time
            )
    
    def test_redis(self) -> TestResult:
        """测试 Redis 连接"""
        config = self.services['redis']
        start_time = time.time()
        
        try:
            # 建立连接
            r = redis.Redis(
                host=config.host,
                port=config.port,
                password=config.credentials['password'],
                db=config.credentials['db'],
                socket_timeout=config.timeout,
                socket_connect_timeout=config.timeout
            )
            
            # 执行 PING 测试
            pong = r.ping()
            if not pong:
                raise Exception("PING 测试失败")
            
            # 获取服务器信息
            info = r.info()
            version = info.get('redis_version', 'Unknown')
            
            # 执行简单的读写测试
            test_key = 'graphrag:health_check'
            test_value = f'test_{int(time.time())}'
            
            r.set(test_key, test_value, ex=60)  # 60秒过期
            retrieved_value = r.get(test_key)
            
            if retrieved_value.decode() != test_value:
                raise Exception("读写测试失败")
            
            r.delete(test_key)  # 清理测试数据
            
            response_time = time.time() - start_time
            
            return TestResult(
                service_name=config.name,
                success=True,
                message="连接成功",
                response_time=response_time,
                version=version,
                details={
                    'used_memory': info.get('used_memory_human', 'Unknown'),
                    'connected_clients': info.get('connected_clients', 0)
                }
            )
            
        except Exception as e:
            response_time = time.time() - start_time
            return TestResult(
                service_name=config.name,
                success=False,
                message=f"连接失败: {str(e)}",
                response_time=response_time
            )
    
    def test_minio(self) -> TestResult:
        """测试 MinIO 连接"""
        config = self.services['minio']
        start_time = time.time()
        
        try:
            # 测试健康检查接口
            health_url = f"http://{config.host}:{config.port}/minio/health/live"
            response = requests.get(health_url, timeout=config.timeout)
            response.raise_for_status()
            
            response_time = time.time() - start_time
            
            return TestResult(
                service_name=config.name,
                success=True,
                message="连接成功",
                response_time=response_time,
                details={'status': 'healthy'}
            )
            
        except Exception as e:
            response_time = time.time() - start_time
            return TestResult(
                service_name=config.name,
                success=False,
                message=f"连接失败: {str(e)}",
                response_time=response_time
            )
    
    def test_weaviate(self) -> TestResult:
        """测试 Weaviate 连接"""
        config = self.services['weaviate']
        start_time = time.time()
        
        try:
            # 测试就绪检查接口
            ready_url = f"http://{config.host}:{config.port}/v1/.well-known/ready"
            response = requests.get(ready_url, timeout=config.timeout)
            response.raise_for_status()
            
            # 获取元数据信息
            meta_url = f"http://{config.host}:{config.port}/v1/meta"
            meta_response = requests.get(meta_url, timeout=config.timeout)
            meta_response.raise_for_status()
            
            meta_data = meta_response.json()
            version = meta_data.get('version', 'Unknown')
            
            response_time = time.time() - start_time
            
            return TestResult(
                service_name=config.name,
                success=True,
                message="连接成功",
                response_time=response_time,
                version=version,
                details={
                    'hostname': meta_data.get('hostname', 'Unknown'),
                    'modules': meta_data.get('modules', {})
                }
            )
            
        except Exception as e:
            response_time = time.time() - start_time
            return TestResult(
                service_name=config.name,
                success=False,
                message=f"连接失败: {str(e)}",
                response_time=response_time
            )
    
    def test_mineru(self) -> TestResult:
        """测试 MinerU 连接"""
        config = self.services['mineru']
        start_time = time.time()
        
        try:
            # 测试 Streamlit 健康检查接口
            health_url = f"http://{config.host}:{config.port}/_stcore/health"
            response = requests.get(health_url, timeout=config.timeout)
            response.raise_for_status()
            
            response_time = time.time() - start_time
            
            return TestResult(
                service_name=config.name,
                success=True,
                message="连接成功",
                response_time=response_time,
                details={'status': 'healthy'}
            )
            
        except Exception as e:
            response_time = time.time() - start_time
            return TestResult(
                service_name=config.name,
                success=False,
                message=f"连接失败: {str(e)}",
                response_time=response_time
            )
    
    def run_all_tests(self) -> List[TestResult]:
        """运行所有服务测试"""
        logger.info("开始执行基础服务健康检查...")
        
        test_methods = [
            self.test_postgres,
            self.test_neo4j,
            self.test_redis,
            self.test_minio,
            self.test_weaviate,
            self.test_mineru
        ]
        
        results = []
        for test_method in test_methods:
            try:
                result = test_method()
                results.append(result)
                
                if result.success:
                    logger.info(f"✅ {result.service_name}: {result.message} "
                              f"(响应时间: {result.response_time:.2f}s)")
                    if result.version:
                        logger.info(f"   版本: {result.version}")
                else:
                    logger.error(f"❌ {result.service_name}: {result.message}")
                    
            except Exception as e:
                logger.error(f"❌ 测试 {test_method.__name__} 时发生异常: {str(e)}")
                results.append(TestResult(
                    service_name="Unknown",
                    success=False,
                    message=f"测试异常: {str(e)}",
                    response_time=0.0
                ))
        
        self.results = results
        return results
    
    def generate_report(self) -> Dict:
        """生成测试报告"""
        if not self.results:
            return {}
        
        total_services = len(self.results)
        successful_services = sum(1 for r in self.results if r.success)
        failed_services = total_services - successful_services
        
        avg_response_time = sum(r.response_time for r in self.results) / total_services
        
        report = {
            'timestamp': datetime.now().isoformat(),
            'summary': {
                'total_services': total_services,
                'successful_services': successful_services,
                'failed_services': failed_services,
                'success_rate': (successful_services / total_services) * 100,
                'average_response_time': avg_response_time
            },
            'services': []
        }
        
        for result in self.results:
            service_info = {
                'name': result.service_name,
                'status': 'success' if result.success else 'failed',
                'message': result.message,
                'response_time': result.response_time
            }
            
            if result.version:
                service_info['version'] = result.version
            
            if result.details:
                service_info['details'] = result.details
            
            report['services'].append(service_info)
        
        return report
    
    def print_summary(self):
        """打印测试结果摘要"""
        if not self.results:
            logger.warning("没有测试结果可显示")
            return
        
        print("\n" + "="*60)
        print("GraphRAG 基础服务健康检查报告")
        print("="*60)
        
        total_services = len(self.results)
        successful_services = sum(1 for r in self.results if r.success)
        failed_services = total_services - successful_services
        
        print(f"总服务数: {total_services}")
        print(f"成功服务: {successful_services}")
        print(f"失败服务: {failed_services}")
        print(f"成功率: {(successful_services/total_services)*100:.1f}%")
        
        if successful_services == total_services:
            print("\n🎉 所有服务运行正常！")
        else:
            print(f"\n⚠️  有 {failed_services} 个服务存在问题，请检查日志")
        
        print("="*60)

def main():
    """主函数"""
    try:
        # 创建测试器实例
        tester = ServiceTester()
        
        # 运行所有测试
        results = tester.run_all_tests()
        
        # 打印摘要
        tester.print_summary()
        
        # 生成详细报告
        report = tester.generate_report()
        
        # 保存报告到文件
        report_file = f"health_check_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        with open(report_file, 'w', encoding='utf-8') as f:
            json.dump(report, f, ensure_ascii=False, indent=2)
        
        logger.info(f"详细报告已保存到: {report_file}")
        
        # 根据测试结果设置退出码
        failed_count = sum(1 for r in results if not r.success)
        sys.exit(0 if failed_count == 0 else 1)
        
    except KeyboardInterrupt:
        logger.info("测试被用户中断")
        sys.exit(1)
    except Exception as e:
        logger.error(f"执行测试时发生异常: {str(e)}")
        sys.exit(1)

if __name__ == "__main__":
    main()