# GraphRAG 贡献指南

感谢您对 GraphRAG 项目的关注！我们欢迎各种形式的贡献，包括但不限于代码贡献、文档改进、问题报告和功能建议。

## 🤝 如何贡献

我们欢迎各种形式的贡献！无论是：

- 代码开发
- 文档改进
- 问题反馈
- 功能建议
- 社区支持

> **当前开发重点**: 项目优先后端开发，暂不考虑测试、容灾、代码审查等流程

## 🚀 快速开始

### 1. 准备开发环境

```bash
# Fork 项目到你的 GitHub 账户
# 然后克隆你的 Fork

git clone https://github.com/YOUR_USERNAME/GraphRAG_NEO_IMG.git
cd GraphRAG_NEO_IMG

# 添加上游仓库
git remote add upstream https://github.com/your-org/GraphRAG_NEO_IMG.git

# 创建开发分支
git checkout -b feature/your-feature-name
```

### 2. 设置开发环境

```bash
# 启动基础服务（PostgreSQL, Neo4j, Redis 等）
docker-compose -f docker-compose.dev.yml up -d

# 安装 uv 包管理器（如果尚未安装）
curl -LsSf https://astral.sh/uv/install.sh | sh

# 设置后端开发环境（当前开发重点）
cd backend
uv venv                    # 创建虚拟环境
source .venv/bin/activate  # 激活虚拟环境
uv pip install -r requirements.txt  # 安装依赖

# 启动后端开发服务器
uvicorn src.main:app --reload --host 0.0.0.0 --port 8000
```

> **开发重点**: 当前阶段专注于后端 API 开发，前端开发暂缓

### 3. 开发验证

```bash
# 检查 API 服务状态
curl http://localhost:8000/health

# 查看 API 文档
# 浏览器访问: http://localhost:8000/docs
```

## 📋 开发流程

### Git 工作流

我们使用 [Git Flow](https://nvie.com/posts/a-successful-git-branching-model/) 工作流：

```mermaid
gitgraph
    commit id: "Initial"
    branch develop
    checkout develop
    commit id: "Dev setup"
    branch feature/new-feature
    checkout feature/new-feature
    commit id: "Feature work"
    commit id: "Feature complete"
    checkout develop
    merge feature/new-feature
    commit id: "Merge feature"
    checkout main
    merge develop
    commit id: "Release v1.1"
```

### 分支命名规范

| 分支类型 | 命名格式 | 示例 | 说明 |
|---------|---------|------|------|
| **功能分支** | `feature/description` | `feature/add-ocr-support` | 新功能开发 |
| **修复分支** | `fix/description` | `fix/memory-leak-issue` | Bug 修复 |
| **文档分支** | `docs/description` | `docs/update-api-guide` | 文档更新 |
| **重构分支** | `refactor/description` | `refactor/database-layer` | 代码重构 |
| **性能分支** | `perf/description` | `perf/optimize-query` | 性能优化 |

### 提交信息规范

我们使用 [Conventional Commits](https://www.conventionalcommits.org/) 规范：

```bash
<type>[optional scope]: <description>

[optional body]

[optional footer(s)]
```

#### 提交类型

| 类型 | 说明 | 示例 |
|------|------|------|
| `feat` | 新功能 | `feat(api): add document upload endpoint` |
| `fix` | Bug 修复 | `fix(db): resolve connection timeout issue` |
| `docs` | 文档更新 | `docs(readme): update installation guide` |
| `style` | 代码格式 | `style(backend): fix linting issues` |
| `refactor` | 代码重构 | `refactor(parser): simplify text extraction` |
| `perf` | 性能优化 | `perf(query): optimize graph traversal` |
| `test` | 测试相关 | `test(api): add integration tests` |
| `chore` | 构建/工具 | `chore(deps): update dependencies` |

#### 提交示例

```bash
# 好的提交信息
feat(knowledge): add entity relationship extraction

- Implement NER pipeline using spaCy
- Add relationship classification model
- Support custom entity types
- Include confidence scoring

Closes #123

# 不好的提交信息
fix bug
update code
add stuff
```

## 🔧 代码规范

### Python 代码规范

我们遵循 [PEP 8](https://pep8.org/) 和项目特定的规范：

```python
"""
文件级注释：说明模块的用途和主要功能
"""

from typing import List, Optional, Dict, Any
import logging

logger = logging.getLogger(__name__)


class DocumentProcessor:
    """
    文档处理器类
    
    负责处理各种格式的文档，包括解析、分块和元数据提取。
    
    Attributes:
        supported_formats: 支持的文档格式列表
        chunk_size: 文档分块大小
    """
    
    def __init__(self, chunk_size: int = 1000) -> None:
        """
        初始化文档处理器
        
        Args:
            chunk_size: 文档分块大小，默认1000字符
        """
        self.chunk_size = chunk_size
        self.supported_formats = ['.pdf', '.txt', '.md', '.html']
    
    def process_document(
        self, 
        file_path: str, 
        extract_metadata: bool = True
    ) -> Dict[str, Any]:
        """
        处理单个文档
        
        Args:
            file_path: 文档文件路径
            extract_metadata: 是否提取元数据
            
        Returns:
            包含处理结果的字典，包含文本内容、分块和元数据
            
        Raises:
            ValueError: 当文件格式不支持时
            FileNotFoundError: 当文件不存在时
        """
        if not self._is_supported_format(file_path):
            raise ValueError(f"Unsupported file format: {file_path}")
        
        # 实现处理逻辑
        logger.info(f"Processing document: {file_path}")
        
        return {
            'content': 'extracted_text',
            'chunks': ['chunk1', 'chunk2'],
            'metadata': {} if extract_metadata else None
        }
    
    def _is_supported_format(self, file_path: str) -> bool:
        """检查文件格式是否支持"""
        return any(file_path.endswith(fmt) for fmt in self.supported_formats)
```

### TypeScript 代码规范

```typescript
/**
 * 文档管理相关的类型定义和接口
 */

export interface Document {
  /** 文档唯一标识符 */
  id: string;
  /** 文档标题 */
  title: string;
  /** 文档内容 */
  content: string;
  /** 文档类型 */
  type: DocumentType;
  /** 创建时间 */
  createdAt: Date;
  /** 更新时间 */
  updatedAt: Date;
}

export enum DocumentType {
  PDF = 'pdf',
  TEXT = 'text',
  MARKDOWN = 'markdown',
  HTML = 'html'
}

/**
 * 文档服务类
 * 
 * 提供文档的 CRUD 操作和相关业务逻辑
 */
export class DocumentService {
  private readonly apiClient: ApiClient;

  constructor(apiClient: ApiClient) {
    this.apiClient = apiClient;
  }

  /**
   * 上传文档
   * 
   * @param file - 要上传的文件
   * @param options - 上传选项
   * @returns Promise<Document> 上传成功的文档信息
   */
  async uploadDocument(
    file: File, 
    options: UploadOptions = {}
  ): Promise<Document> {
    const formData = new FormData();
    formData.append('file', file);
    
    if (options.extractMetadata) {
      formData.append('extract_metadata', 'true');
    }

    try {
      const response = await this.apiClient.post('/documents/upload', formData);
      return response.data;
    } catch (error) {
      console.error('Failed to upload document:', error);
      throw new Error('Document upload failed');
    }
  }

  /**
   * 获取文档列表
   */
  async getDocuments(params: GetDocumentsParams = {}): Promise<Document[]> {
    const response = await this.apiClient.get('/documents', { params });
    return response.data;
  }
}
```

### 代码质量工具

```bash
# Python 代码检查
flake8 src/
black src/
isort src/
mypy src/

# TypeScript 代码检查
npm run lint
npm run format
npm run type-check

# 自动修复
npm run lint:fix
## 🔧 代码质量工具

### Python 代码格式化

```bash
# 使用 black 格式化代码
uv pip install black
black src/

# 使用 isort 整理导入
uv pip install isort
isort src/

# 使用 flake8 检查代码风格
uv pip install flake8
flake8 src/
```

## 📝 文档规范

### 文档结构

```markdown
# 标题

简短的描述说明文档的用途。

## 目录

- [概述](#概述)
- [安装](#安装)
- [使用方法](#使用方法)
- [API 参考](#api-参考)
- [示例](#示例)
- [常见问题](#常见问题)

## 概述

详细说明功能、特性和使用场景。

## 安装

提供清晰的安装步骤。

## 使用方法

包含代码示例和详细说明。

## API 参考

详细的 API 文档。

## 示例

实际的使用示例。

## 常见问题

常见问题和解决方案。
```

### 代码示例规范

```markdown
## 文档上传示例

### Python SDK

```python
from graphrag_client import GraphRAGClient

# 初始化客户端
client = GraphRAGClient(base_url="http://localhost:8000")

# 上传文档
with open("document.pdf", "rb") as f:
    result = client.upload_document(
        file=f,
        extract_metadata=True,
        auto_process=True
    )

print(f"Document uploaded: {result.id}")
```

### cURL

```bash
curl -X POST "http://localhost:8000/api/documents/upload" \
     -H "Content-Type: multipart/form-data" \
     -F "file=@document.pdf" \
     -F "extract_metadata=true"
```

### JavaScript

```javascript
const formData = new FormData();
formData.append('file', fileInput.files[0]);
formData.append('extract_metadata', 'true');

const response = await fetch('/api/documents/upload', {
  method: 'POST',
  body: formData
});

const result = await response.json();
console.log('Document uploaded:', result.id);
```
```

## 🐛 问题报告

### Bug 报告模板

```markdown
## Bug 描述
简要描述遇到的问题。

## 复现步骤
1. 执行操作 A
2. 执行操作 B
3. 观察到错误

## 预期行为
描述你期望发生的情况。

## 实际行为
描述实际发生的情况。

## 环境信息
- 操作系统: [例如 Ubuntu 20.04]
- Python 版本: [例如 3.9.7]
- Docker 版本: [例如 20.10.8]
- 浏览器: [例如 Chrome 95.0]

## 附加信息
- 错误日志
- 截图
- 相关配置文件
```

### 功能请求模板

```markdown
## 功能描述
简要描述建议的功能。

## 使用场景
描述什么情况下需要这个功能。

## 详细说明
详细描述功能的工作方式。

## 可能的实现方案
如果有想法，可以描述可能的实现方式。

## 优先级
- [ ] 高 - 核心功能
- [ ] 中 - 重要改进
- [ ] 低 - 便利功能
```

## 🏆 贡献者认可

### 贡献类型

我们认可以下类型的贡献：

| 贡献类型 | 说明 | 认可方式 |
|---------|------|---------|
| **代码** | 功能开发、Bug 修复 | GitHub 贡献者列表 |
| **文档** | 文档编写、翻译 | 文档署名 |
| **测试** | 测试用例、质量保证 | 测试贡献者列表 |
| **设计** | UI/UX 设计、图标 | 设计贡献者列表 |
| **社区** | 问题解答、社区管理 | 社区贡献者列表 |

### 贡献者权益

- **代码贡献者**: 获得项目 Contributor 权限
- **核心贡献者**: 获得项目 Maintainer 权限
- **文档贡献者**: 在相关文档中署名
- **所有贡献者**: 在 README 和发布说明中致谢

## 📞 联系我们

### 开发讨论

- **GitHub Discussions**: [项目讨论区](https://github.com/your-org/GraphRAG_NEO_IMG/discussions)
- **开发者邮件列表**: dev@graphrag.com
- **技术交流群**: [加入 Slack](https://graphrag.slack.com)

### 问题反馈

- **Bug 报告**: [GitHub Issues](https://github.com/your-org/GraphRAG_NEO_IMG/issues)
- **功能请求**: [GitHub Issues](https://github.com/your-org/GraphRAG_NEO_IMG/issues)
- **安全问题**: security@graphrag.com

### 项目维护者

- **项目负责人**: [@maintainer1](https://github.com/maintainer1)
- **技术负责人**: [@maintainer2](https://github.com/maintainer2)
- **文档负责人**: [@maintainer3](https://github.com/maintainer3)

---

## 📄 许可证

通过贡献代码，您同意您的贡献将在与项目相同的 [MIT 许可证](LICENSE) 下获得许可。

---

**感谢您的贡献！** 🙏

每一个贡献都让 GraphRAG 变得更好，我们期待与您一起构建更强大的知识图谱系统！