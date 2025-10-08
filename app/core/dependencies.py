#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
GraphRAG 服务依赖注入模块
========================

本模块提供服务依赖注入和初始化功能，包括：
1. 服务实例管理 - 单例模式的服务实例管理
2. 依赖注入 - 自动解析和注入服务依赖
3. 服务生命周期 - 服务的启动、停止和健康检查
4. 配置管理 - 服务配置的加载和验证
5. 连接池管理 - 数据库连接池的管理
6. 缓存管理 - 服务级别的缓存管理
7. 监控集成 - 服务监控和指标收集

所有服务都通过依赖注入容器进行管理，确保服务的正确初始化和资源清理。

作者: GraphRAG Team
创建时间: 2024
版本: 1.0.0
"""

import asyncio
import logging
from typing import Dict, Any, Optional, Type, TypeVar, Callable, List
from contextlib import asynccontextmanager
from functools import lru_cache
import inspect
from datetime import datetime
import weakref

from app.core.config import settings
from app.core.logging import get_logger
# from app.utils.exceptions import ServiceInitializationError, DependencyError

logger = get_logger(__name__)

# 类型变量
T = TypeVar('T')

# 全局服务容器
_service_container: Dict[str, Any] = {}
_service_instances: Dict[Type, Any] = {}
_service_lifecycle_hooks: Dict[str, List[Callable]] = {
    "startup": [],
    "shutdown": [],
    "health_check": []
}


class ServiceContainer:
    """
    服务容器类
    
    负责管理所有服务实例的生命周期，包括创建、初始化、依赖注入和销毁。
    """
    
    def __init__(self):
        """初始化服务容器"""
        self._services: Dict[str, Any] = {}
        self._singletons: Dict[Type, Any] = {}
        self._factories: Dict[Type, Callable] = {}
        self._dependencies: Dict[Type, List[Type]] = {}
        self._initialized: bool = False
        self._startup_hooks: List[Callable] = []
        self._shutdown_hooks: List[Callable] = []
        self._health_checks: List[Callable] = []
        self.settings = settings
        
        logger.info("服务容器初始化完成")
    
    def register_singleton(self, service_type: Type[T], instance: T) -> None:
        """
        注册单例服务
        
        Args:
            service_type: 服务类型
            instance: 服务实例
        """
        service_name = service_type.__name__
        logger.info(f"注册单例服务: {service_name}")
        
        self._singletons[service_type] = instance
        self._services[service_name] = instance
    
    def register_factory(self, service_type: Type[T], factory: Callable[[], T]) -> None:
        """
        注册服务工厂
        
        Args:
            service_type: 服务类型
            factory: 服务工厂函数
        """
        service_name = service_type.__name__
        logger.info(f"注册服务工厂: {service_name}")
        
        self._factories[service_type] = factory
    
    def register_dependencies(self, service_type: Type, dependencies: List[Type]) -> None:
        """
        注册服务依赖
        
        Args:
            service_type: 服务类型
            dependencies: 依赖的服务类型列表
        """
        service_name = service_type.__name__
        logger.info(f"注册服务依赖: {service_name} -> {[dep.__name__ for dep in dependencies]}")
        
        self._dependencies[service_type] = dependencies
    
    def get_service(self, service_type: Type[T]) -> T:
        """
        获取服务实例
        
        Args:
            service_type: 服务类型
            
        Returns:
            T: 服务实例
            
        Raises:
            DependencyError: 当服务不存在或依赖无法解析时
        """
        # 检查单例
        if service_type in self._singletons:
            return self._singletons[service_type]
        
        # 检查工厂
        if service_type in self._factories:
            try:
                # 解析依赖
                dependencies = self._resolve_dependencies(service_type)
                
                # 创建实例
                factory = self._factories[service_type]
                if dependencies:
                    instance = factory(**dependencies)
                else:
                    instance = factory()
                
                # 如果是单例，缓存实例
                if hasattr(instance, '_singleton') and instance._singleton:
                    self._singletons[service_type] = instance
                
                return instance
                
            except Exception as e:
                logger.error(f"创建服务实例失败: {service_type.__name__}, 错误: {str(e)}")
                raise Exception(f"无法创建服务实例: {service_type.__name__}")
        
        raise Exception(f"服务未注册: {service_type.__name__}")
    
    def _resolve_dependencies(self, service_type: Type) -> Dict[str, Any]:
        """
        解析服务依赖
        
        Args:
            service_type: 服务类型
            
        Returns:
            Dict[str, Any]: 依赖实例字典
        """
        dependencies = {}
        
        if service_type in self._dependencies:
            for dep_type in self._dependencies[service_type]:
                dep_instance = self.get_service(dep_type)
                # 使用参数名作为键
                param_name = dep_type.__name__.lower().replace('service', '')
                dependencies[param_name] = dep_instance
        
        return dependencies
    
    async def initialize(self) -> None:
        """初始化所有服务"""
        if self._initialized:
            logger.warning("服务容器已经初始化")
            return
        
        logger.info("开始初始化服务容器")
        
        try:
            # 执行启动钩子
            for hook in self._startup_hooks:
                if asyncio.iscoroutinefunction(hook):
                    await hook()
                else:
                    hook()
            
            # 初始化所有单例服务（只初始化有initialize方法的服务）
            for service_type, instance in self._singletons.items():
                if hasattr(instance, 'initialize') and callable(instance.initialize):
                    logger.info(f"初始化服务: {service_type.__name__}")
                    if asyncio.iscoroutinefunction(instance.initialize):
                        await instance.initialize()
                    else:
                        instance.initialize()
                else:
                    logger.info(f"跳过服务初始化（无initialize方法）: {service_type.__name__}")
            
            self._initialized = True
            logger.info("服务容器初始化完成")
            
        except Exception as e:
            logger.error(f"服务容器初始化失败: {str(e)}")
            raise Exception(f"服务容器初始化失败: {str(e)}")
    
    async def shutdown(self) -> None:
        """关闭所有服务"""
        if not self._initialized:
            logger.warning("服务容器未初始化")
            return
        
        logger.info("开始关闭服务容器")
        
        try:
            # 关闭所有单例服务
            for service_type, instance in self._singletons.items():
                if hasattr(instance, 'shutdown') and callable(instance.shutdown):
                    logger.info(f"关闭服务: {service_type.__name__}")
                    try:
                        if asyncio.iscoroutinefunction(instance.shutdown):
                            await instance.shutdown()
                        else:
                            instance.shutdown()
                    except Exception as e:
                        logger.error(f"关闭服务失败: {service_type.__name__}, 错误: {str(e)}")
            
            # 执行关闭钩子
            for hook in self._shutdown_hooks:
                try:
                    if asyncio.iscoroutinefunction(hook):
                        await hook()
                    else:
                        hook()
                except Exception as e:
                    logger.error(f"执行关闭钩子失败: {str(e)}")
            
            self._initialized = False
            logger.info("服务容器关闭完成")
            
        except Exception as e:
            logger.error(f"服务容器关闭失败: {str(e)}")
    
    async def health_check(self) -> Dict[str, Any]:
        """
        执行健康检查
        
        Returns:
            Dict[str, Any]: 健康检查结果
        """
        logger.info("执行服务容器健康检查")
        
        health_status = {
            "status": "healthy",
            "services": {},
            "timestamp": datetime.utcnow().isoformat(),
            "container_info": {
                "initialized": self._initialized,
                "service_count": len(self._singletons),
                "factory_count": len(self._factories)
            }
        }
        
        # 检查所有服务
        unhealthy_services = []
        
        for service_type, instance in self._singletons.items():
            service_name = service_type.__name__
            
            try:
                if hasattr(instance, 'health_check') and callable(instance.health_check):
                    if asyncio.iscoroutinefunction(instance.health_check):
                        service_health = await instance.health_check()
                    else:
                        service_health = instance.health_check()
                    
                    # 处理不同类型的健康检查返回值
                    if isinstance(service_health, bool):
                        # 如果返回布尔值，转换为标准格式
                        service_health_dict = {
                            "status": "healthy" if service_health else "unhealthy",
                            "timestamp": datetime.utcnow().isoformat()
                        }
                    elif isinstance(service_health, dict):
                        # 如果返回字典，直接使用
                        service_health_dict = service_health
                    else:
                        # 其他类型，转换为字符串
                        service_health_dict = {
                            "status": "unknown",
                            "message": str(service_health),
                            "timestamp": datetime.utcnow().isoformat()
                        }
                    
                    health_status["services"][service_name] = service_health_dict
                    
                    if service_health_dict.get("status") != "healthy":
                        unhealthy_services.append(service_name)
                else:
                    health_status["services"][service_name] = {
                        "status": "unknown",
                        "message": "服务未实现健康检查"
                    }
                    
            except Exception as e:
                logger.error(f"服务健康检查失败: {service_name}, 错误: {str(e)}")
                health_status["services"][service_name] = {
                    "status": "unhealthy",
                    "error": str(e)
                }
                unhealthy_services.append(service_name)
        
        # 执行自定义健康检查钩子
        for hook in self._health_checks:
            try:
                if asyncio.iscoroutinefunction(hook):
                    hook_result = await hook()
                else:
                    hook_result = hook()
                
                if hook_result and isinstance(hook_result, dict):
                    health_status.update(hook_result)
                    
            except Exception as e:
                logger.error(f"执行健康检查钩子失败: {str(e)}")
        
        # 设置整体状态
        if unhealthy_services:
            health_status["status"] = "unhealthy"
            health_status["unhealthy_services"] = unhealthy_services
        
        return health_status
    
    def add_startup_hook(self, hook: Callable) -> None:
        """添加启动钩子"""
        self._startup_hooks.append(hook)
        logger.info(f"添加启动钩子: {hook.__name__}")
    
    def add_shutdown_hook(self, hook: Callable) -> None:
        """添加关闭钩子"""
        self._shutdown_hooks.append(hook)
        logger.info(f"添加关闭钩子: {hook.__name__}")
    
    def add_health_check_hook(self, hook: Callable) -> None:
        """添加健康检查钩子"""
        self._health_checks.append(hook)
        logger.info(f"添加健康检查钩子: {hook.__name__}")


# 全局服务容器实例
_container: Optional[ServiceContainer] = None


def get_container() -> ServiceContainer:
    """
    获取全局服务容器实例
    
    Returns:
        ServiceContainer: 服务容器实例
    """
    global _container
    if _container is None:
        _container = ServiceContainer()
    return _container


# 服务工厂函数

async def create_embedding_service():
    """创建嵌入服务"""
    from app.services.embedding_service import EmbeddingService
    
    logger.info("创建嵌入服务")
    service = EmbeddingService()
    # EmbeddingService没有initialize方法，直接返回
    return service


async def create_graph_service():
    """创建图服务"""
    from app.services.graph_service import GraphService
    
    logger.info("创建图服务")
    service = GraphService()
    # GraphService没有initialize方法，直接返回
    return service


async def create_document_service():
    """创建文档服务"""
    from app.services.document_service import DocumentService
    
    logger.info("创建文档服务")
    service = DocumentService()
    # DocumentService没有initialize方法，直接返回
    return service


async def create_text_service():
    """创建文本服务"""
    from app.services.text_service import TextService
    
    logger.info("创建文本服务")
    service = TextService()
    # TextService没有initialize方法，直接返回
    return service


async def create_entity_service():
    """创建实体服务"""
    from app.services.entity_service import EntityService
    
    logger.info("创建实体服务")
    service = EntityService()
    # EntityService没有initialize方法，直接返回
    return service


async def create_relation_service():
    """创建关系服务"""
    from app.services.relation_service import RelationService
    
    logger.info("创建关系服务")
    service = RelationService()
    # RelationService没有initialize方法，直接返回
    return service


async def create_graphrag_service():
    """创建GraphRAG服务"""
    from app.services.graphrag_service import GraphRAGService
    
    logger.info("创建GraphRAG服务")
    service = GraphRAGService()
    # GraphRAGService没有initialize方法，直接返回
    return service


async def create_azure_openai_service():
    """创建Azure OpenAI服务"""
    from app.services.azure_openai_service import AzureOpenAIService
    
    logger.info("创建Azure OpenAI服务")
    service = AzureOpenAIService()
    # AzureOpenAIService没有initialize方法，直接返回
    return service


async def create_file_storage_service():
    """创建文件存储服务"""
    from app.services.file_storage_service import FileStorageService
    
    logger.info("创建文件存储服务")
    service = FileStorageService()
    # FileStorageService没有initialize方法，直接返回
    return service


async def create_graph_query_service():
    """创建图查询服务"""
    from app.services.graph_query_service import GraphQueryService
    
    logger.info("创建图查询服务")
    service = GraphQueryService()
    # GraphQueryService没有initialize方法，直接返回
    return service


async def create_graph_index_service():
    """创建图索引服务"""
    from app.services.graph_index_service import GraphIndexService
    
    logger.info("创建图索引服务")
    service = GraphIndexService()
    # GraphIndexService没有initialize方法，直接返回
    return service


# 依赖注入装饰器

def inject(service_type: Type[T]) -> Callable[[], T]:
    """
    依赖注入装饰器
    
    Args:
        service_type: 服务类型
        
    Returns:
        Callable[[], T]: 服务获取函数
    """
    def get_service() -> T:
        container = get_container()
        return container.get_service(service_type)
    
    return get_service


# FastAPI 依赖函数

async def get_embedding_service():
    """获取嵌入服务依赖"""
    from app.services.embedding_service import EmbeddingService
    container = get_container()
    return container.get_service(EmbeddingService)


async def get_graph_service():
    """获取图服务依赖"""
    from app.services.graph_service import GraphService
    container = get_container()
    return container.get_service(GraphService)


async def get_document_service():
    """获取文档服务依赖"""
    from app.services.document_service import DocumentService
    container = get_container()
    return container.get_service(DocumentService)


async def get_text_service():
    """获取文本服务依赖"""
    from app.services.text_service import TextService
    container = get_container()
    return container.get_service(TextService)


async def get_entity_service():
    """获取实体服务依赖"""
    from app.services.entity_service import EntityService
    container = get_container()
    return container.get_service(EntityService)


async def get_relation_service():
    """获取关系服务依赖"""
    from app.services.relation_service import RelationService
    container = get_container()
    return container.get_service(RelationService)


async def get_graphrag_service():
    """获取GraphRAG服务依赖"""
    from app.services.graphrag_service import GraphRAGService
    container = get_container()
    return container.get_service(GraphRAGService)


async def get_azure_openai_service():
    """获取Azure OpenAI服务依赖"""
    from app.services.azure_openai_service import AzureOpenAIService
    container = get_container()
    return container.get_service(AzureOpenAIService)


async def get_file_storage_service():
    """获取文件存储服务依赖"""
    from app.services.file_storage_service import FileStorageService
    container = get_container()
    return container.get_service(FileStorageService)


async def get_graph_query_service():
    """获取图查询服务依赖"""
    from app.services.graph_query_service import GraphQueryService
    container = get_container()
    return container.get_service(GraphQueryService)


async def get_graph_index_service():
    """获取图索引服务依赖"""
    from app.services.graph_index_service import GraphIndexService
    container = get_container()
    return container.get_service(GraphIndexService)


# 服务初始化函数

async def initialize_services() -> None:
    """
    初始化所有服务
    
    按照依赖顺序初始化所有服务实例。
    """
    logger.info("开始初始化所有服务")
    
    try:
        container = get_container()
        
        # 导入所有服务类
        from app.services.embedding_service import EmbeddingService
        from app.services.graph_service import GraphService
        from app.services.document_service import DocumentService
        from app.services.text_service import TextService
        from app.services.entity_service import EntityService
        from app.services.relation_service import RelationService
        from app.services.graphrag_service import GraphRAGService
        from app.services.azure_openai_service import AzureOpenAIService
        from app.services.file_storage_service import FileStorageService
        from app.services.graph_query_service import GraphQueryService
        from app.services.graph_index_service import GraphIndexService
        
        # 注册服务工厂
        container.register_factory(EmbeddingService, create_embedding_service)
        container.register_factory(GraphService, create_graph_service)
        container.register_factory(DocumentService, create_document_service)
        container.register_factory(TextService, create_text_service)
        container.register_factory(EntityService, create_entity_service)
        container.register_factory(RelationService, create_relation_service)
        container.register_factory(GraphRAGService, create_graphrag_service)
        container.register_factory(AzureOpenAIService, create_azure_openai_service)
        container.register_factory(FileStorageService, create_file_storage_service)
        container.register_factory(GraphQueryService, create_graph_query_service)
        container.register_factory(GraphIndexService, create_graph_index_service)
        
        # 注册服务依赖关系
        container.register_dependencies(DocumentService, [FileStorageService, TextService])
        container.register_dependencies(EntityService, [GraphService, AzureOpenAIService])
        container.register_dependencies(RelationService, [GraphService, AzureOpenAIService])
        container.register_dependencies(GraphRAGService, [EmbeddingService, GraphService, AzureOpenAIService, DocumentService])
        container.register_dependencies(GraphQueryService, [GraphService])
        container.register_dependencies(GraphIndexService, [GraphService])
        
        # 创建核心服务实例（单例）
        embedding_service = await create_embedding_service()
        graph_service = await create_graph_service()
        azure_openai_service = await create_azure_openai_service()
        file_storage_service = await create_file_storage_service()
        
        # 注册单例服务
        container.register_singleton(EmbeddingService, embedding_service)
        container.register_singleton(GraphService, graph_service)
        container.register_singleton(AzureOpenAIService, azure_openai_service)
        container.register_singleton(FileStorageService, file_storage_service)
        
        # 初始化容器
        await container.initialize()
        
        logger.info("所有服务初始化完成")
        
    except Exception as e:
        logger.error(f"服务初始化失败: {str(e)}")
        raise Exception(f"服务初始化失败: {str(e)}")


async def shutdown_services() -> None:
    """
    关闭所有服务
    
    按照相反的依赖顺序关闭所有服务实例。
    """
    logger.info("开始关闭所有服务")
    
    try:
        container = get_container()
        await container.shutdown()
        
        logger.info("所有服务关闭完成")
        
    except Exception as e:
        logger.error(f"服务关闭失败: {str(e)}")


async def health_check_services() -> Dict[str, Any]:
    """
    执行所有服务的健康检查
    
    Returns:
        Dict[str, Any]: 健康检查结果
    """
    logger.info("执行所有服务健康检查")
    
    try:
        container = get_container()
        return await container.health_check()
        
    except Exception as e:
        logger.error(f"服务健康检查失败: {str(e)}")
        return {
            "status": "unhealthy",
            "error": str(e),
            "timestamp": datetime.utcnow().isoformat()
        }


# 上下文管理器

@asynccontextmanager
async def service_lifespan():
    """
    服务生命周期上下文管理器
    
    用于FastAPI应用的lifespan事件管理。
    """
    logger.info("启动服务生命周期管理")
    
    try:
        # 启动服务
        await initialize_services()
        yield
        
    except Exception as e:
        logger.error(f"服务启动失败: {str(e)}")
        raise
        
    finally:
        # 关闭服务
        try:
            await shutdown_services()
        except Exception as e:
            logger.error(f"服务关闭失败: {str(e)}")
        
        logger.info("服务生命周期管理结束")


# 监控和指标

class ServiceMetrics:
    """服务指标收集器"""
    
    def __init__(self):
        """初始化指标收集器"""
        self._metrics: Dict[str, Any] = {
            "service_calls": {},
            "service_errors": {},
            "service_response_times": {},
            "service_health_status": {}
        }
        self._start_time = datetime.utcnow()
    
    def record_service_call(self, service_name: str, method_name: str, response_time: float) -> None:
        """记录服务调用"""
        key = f"{service_name}.{method_name}"
        
        if key not in self._metrics["service_calls"]:
            self._metrics["service_calls"][key] = 0
        self._metrics["service_calls"][key] += 1
        
        if key not in self._metrics["service_response_times"]:
            self._metrics["service_response_times"][key] = []
        self._metrics["service_response_times"][key].append(response_time)
    
    def record_service_error(self, service_name: str, method_name: str, error: str) -> None:
        """记录服务错误"""
        key = f"{service_name}.{method_name}"
        
        if key not in self._metrics["service_errors"]:
            self._metrics["service_errors"][key] = []
        self._metrics["service_errors"][key].append({
            "error": error,
            "timestamp": datetime.utcnow().isoformat()
        })
    
    def update_service_health(self, service_name: str, status: str) -> None:
        """更新服务健康状态"""
        self._metrics["service_health_status"][service_name] = {
            "status": status,
            "timestamp": datetime.utcnow().isoformat()
        }
    
    def get_metrics(self) -> Dict[str, Any]:
        """获取所有指标"""
        uptime = (datetime.utcnow() - self._start_time).total_seconds()
        
        return {
            "uptime_seconds": uptime,
            "service_calls": self._metrics["service_calls"],
            "service_errors": self._metrics["service_errors"],
            "service_response_times": self._metrics["service_response_times"],
            "service_health_status": self._metrics["service_health_status"],
            "timestamp": datetime.utcnow().isoformat()
        }


# 全局指标收集器
_metrics: Optional[ServiceMetrics] = None


def get_metrics() -> ServiceMetrics:
    """
    获取全局指标收集器
    
    Returns:
        ServiceMetrics: 指标收集器实例
    """
    global _metrics
    if _metrics is None:
        _metrics = ServiceMetrics()
    return _metrics