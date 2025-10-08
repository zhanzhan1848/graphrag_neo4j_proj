"""
数据库连接和会话管理模块

该模块提供数据库连接、会话管理和依赖注入功能，
支持 PostgreSQL 数据库连接和 SQLAlchemy ORM 操作。
"""

import logging
from typing import Generator, Optional
from contextlib import contextmanager
from sqlalchemy import create_engine, event, pool
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker, Session
from sqlalchemy.pool import QueuePool
from sqlalchemy.exc import SQLAlchemyError, DisconnectionError

from app.core.config import settings
from app.utils.logger import get_logger

logger = get_logger(__name__)

# SQLAlchemy 基础模型类
Base = declarative_base()

# 数据库引擎实例
engine = None
SessionLocal = None


def create_database_engine():
    """
    创建数据库引擎
    
    配置数据库连接池、超时设置和连接参数
    
    Returns:
        Engine: SQLAlchemy 数据库引擎实例
    """
    global engine
    
    if engine is not None:
        return engine
    
    try:
        # 构建数据库连接 URL
        database_url = (
            f"postgresql://{settings.POSTGRES_USER}:{settings.POSTGRES_PASSWORD}"
            f"@{settings.POSTGRES_HOST}:{settings.POSTGRES_PORT}/{settings.POSTGRES_DB}"
        )
        
        # 创建数据库引擎
        engine = create_engine(
            database_url,
            poolclass=QueuePool,
            pool_size=settings.DATABASE_POOL_SIZE,
            max_overflow=settings.DATABASE_MAX_OVERFLOW,
            pool_timeout=settings.DATABASE_POOL_TIMEOUT,
            pool_recycle=settings.DATABASE_POOL_RECYCLE,
            pool_pre_ping=True,  # 连接前检查连接有效性
            echo=settings.DATABASE_ECHO,  # 是否打印 SQL 语句
            future=True,  # 使用 SQLAlchemy 2.0 风格
        )
        
        # 添加连接事件监听器
        @event.listens_for(engine, "connect")
        def set_sqlite_pragma(dbapi_connection, connection_record):
            """设置数据库连接参数"""
            if hasattr(dbapi_connection, 'execute'):
                # 设置 PostgreSQL 连接参数
                dbapi_connection.execute("SET timezone TO 'UTC'")
        
        @event.listens_for(engine, "checkout")
        def receive_checkout(dbapi_connection, connection_record, connection_proxy):
            """连接检出时的处理"""
            logger.debug("Database connection checked out")
        
        @event.listens_for(engine, "checkin")
        def receive_checkin(dbapi_connection, connection_record):
            """连接检入时的处理"""
            logger.debug("Database connection checked in")
        
        logger.info(f"Database engine created successfully: {settings.POSTGRES_HOST}:{settings.POSTGRES_PORT}")
        return engine
        
    except Exception as e:
        logger.error(f"Failed to create database engine: {e}")
        raise


def create_session_factory():
    """
    创建会话工厂
    
    Returns:
        sessionmaker: SQLAlchemy 会话工厂
    """
    global SessionLocal
    
    if SessionLocal is not None:
        return SessionLocal
    
    try:
        # 确保引擎已创建
        if engine is None:
            create_database_engine()
        
        # 创建会话工厂
        SessionLocal = sessionmaker(
            bind=engine,
            autocommit=False,
            autoflush=False,
            expire_on_commit=False,
        )
        
        logger.info("Database session factory created successfully")
        return SessionLocal
        
    except Exception as e:
        logger.error(f"Failed to create session factory: {e}")
        raise


def get_db() -> Generator[Session, None, None]:
    """
    获取数据库会话（依赖注入）
    
    用于 FastAPI 的依赖注入，自动管理数据库会话的生命周期。
    会话在请求结束时自动关闭，异常时自动回滚。
    
    Yields:
        Session: SQLAlchemy 数据库会话
        
    Raises:
        SQLAlchemyError: 数据库操作异常
    """
    # 确保会话工厂已创建
    if SessionLocal is None:
        create_session_factory()
    
    db = SessionLocal()
    try:
        logger.debug("Database session created")
        yield db
    except SQLAlchemyError as e:
        logger.error(f"Database session error: {e}")
        db.rollback()
        raise
    except Exception as e:
        logger.error(f"Unexpected error in database session: {e}")
        db.rollback()
        raise
    finally:
        db.close()
        logger.debug("Database session closed")


@contextmanager
def get_db_session() -> Generator[Session, None, None]:
    """
    获取数据库会话（上下文管理器）
    
    用于在服务层或其他非 FastAPI 环境中获取数据库会话。
    
    Yields:
        Session: SQLAlchemy 数据库会话
        
    Raises:
        SQLAlchemyError: 数据库操作异常
    """
    # 确保会话工厂已创建
    if SessionLocal is None:
        create_session_factory()
    
    db = SessionLocal()
    try:
        logger.debug("Database session created (context manager)")
        yield db
        db.commit()
    except SQLAlchemyError as e:
        logger.error(f"Database session error: {e}")
        db.rollback()
        raise
    except Exception as e:
        logger.error(f"Unexpected error in database session: {e}")
        db.rollback()
        raise
    finally:
        db.close()
        logger.debug("Database session closed (context manager)")


async def init_database():
    """
    初始化数据库
    
    创建数据库引擎和会话工厂，检查数据库连接。
    """
    try:
        logger.info("Initializing database...")
        
        # 创建引擎和会话工厂
        create_database_engine()
        create_session_factory()
        
        # 测试数据库连接
        with get_db_session() as db:
            db.execute("SELECT 1")
            logger.info("Database connection test successful")
        
        logger.info("Database initialized successfully")
        
    except Exception as e:
        logger.error(f"Failed to initialize database: {e}")
        raise


async def close_database():
    """
    关闭数据库连接
    
    清理数据库引擎和连接池。
    """
    global engine, SessionLocal
    
    try:
        logger.info("Closing database connections...")
        
        if engine is not None:
            engine.dispose()
            engine = None
            logger.info("Database engine disposed")
        
        SessionLocal = None
        logger.info("Database closed successfully")
        
    except Exception as e:
        logger.error(f"Error closing database: {e}")
        raise


def get_database_info() -> dict:
    """
    获取数据库信息
    
    Returns:
        dict: 数据库连接信息和状态
    """
    try:
        info = {
            "host": settings.POSTGRES_HOST,
            "port": settings.POSTGRES_PORT,
            "database": settings.POSTGRES_DB,
            "user": settings.POSTGRES_USER,
            "engine_created": engine is not None,
            "session_factory_created": SessionLocal is not None,
        }
        
        if engine is not None:
            pool = engine.pool
            info.update({
                "pool_size": pool.size(),
                "checked_in": pool.checkedin(),
                "checked_out": pool.checkedout(),
                "overflow": pool.overflow(),
            })
        
        return info
        
    except Exception as e:
        logger.error(f"Error getting database info: {e}")
        return {"error": str(e)}


# 数据库健康检查
async def check_database_health() -> dict:
    """
    检查数据库健康状态
    
    Returns:
        dict: 数据库健康状态信息
    """
    try:
        with get_db_session() as db:
            # 执行简单查询测试连接
            result = db.execute("SELECT version(), now()")
            row = result.fetchone()
            
            return {
                "status": "healthy",
                "version": row[0] if row else "unknown",
                "timestamp": row[1].isoformat() if row and row[1] else None,
                "connection_info": get_database_info()
            }
            
    except Exception as e:
        logger.error(f"Database health check failed: {e}")
        return {
            "status": "unhealthy",
            "error": str(e),
            "connection_info": get_database_info()
        }