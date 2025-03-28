# AI Agent Retrieval System

一个基于语义相似度的智能体检索系统，能够根据用户需求智能匹配最合适的 AI Agent。

## 项目特点

- 基于 OpenAI API 和 ChromaDB 实现的向量检索系统
- 支持多领域 AI Agent 的动态生成和管理
- 提供友好的 Gradio Web 界面
- 支持用户反馈和新 Agent 类型的动态添加
- 支持向量检索结果缓存，提升查询性能
- 实现 AI 回答的流式输出，提供更好的用户体验
-

## 核心功能

1. **智能检索匹配**

   - 基于语义相似度的 Agent 匹配
   - 支持多维度的检索结果排序
   - 智能推荐最适合的 Agent 类型

2. **动态 Agent 生成**

   - 支持多个专业领域的 Agent 自动生成
   - 使用 GPT-4 生成专业的系统提示词
   - 向量化存储 Agent 信息

3. **交互式用户界面**

   - 实时查询和响应
   - 支持用户反馈机制
   - 允许添加新的 Agent 类型

4. **性能优化**
   - 检索结果缓存机制
     - 支持可配置的缓存过期时间
     - 基于查询参数的缓存键生成
     - 自动清理过期缓存
   - 流式响应输出
     - AI 回答实时流式显示
     - 减少用户等待时间
     - 提供更自然的交互体验

## 技术架构

- **OpenAI API**: 用于生成 Agent 描述和文本嵌入
- **ChromaDB**: 向量数据库，用于存储和检索 Agent 信息
- **Gradio**: Web 界面框架
- **Python**: 核心开发语言
- **缓存系统**: 内置内存缓存，提升检索性能
- **流式处理**: 支持 OpenAI API 的流式响应处理

## 项目结构

```
.
├── main.py # 主程序入口
├── init_data.py # 数据初始化脚本
├── retrieval.py # 检索系统核心逻辑
├── config/
│ └── settings.py # 配置文件
├── chroma_db/ # 向量数据库存储
└── .env # 环境变量配置
```

## 配置说明

主要配置项在 config/settings.py 中：

- OPENAI_API_KEY: OpenAI API 密钥
- EMBEDDING_MODEL: 使用的嵌入模型
- COLLECTION_NAME: ChromaDB 集合名称
- PERSIST_DIR: 数据持久化目录
- CACHE_TTL: 缓存过期时间(秒)，默认 3600 秒

## 安装说明

1. 克隆项目

```bash
git clone [repository-url]
```

2. 安装依赖

```bash
pip install -r requirements.txt
```

3. 环境配置
   创建 `.env` 文件并添加：OPENAI_API_KEY=your_api_key

## 使用方法

1. 初始化系统

```bash
python init_data.py
```

2. 启动应用

```bash
python main.py
```

3. 访问 Web 界面 打开浏览器访问显示的本地地址（通常是 http://localhost:7860）

## 性能特性

### 缓存系统

- 支持检索结果的自动缓存
- 可配置缓存过期时间
- 缓存命中可显著提升查询速度
- 自动处理缓存失效和更新

### 流式输出

- AI 响应实时显示，无需等待完整回答
- 提供更好的交互体验
- 支持长文本生成场景

测试缓存性能：

```bash
python test_cache.py
```

## 注意事项

- 首次运行需要执行 `init_data.py` 初始化数据
- 确保 OpenAI API 密钥配置正确
- ChromaDB 数据文件夹已添加到 .gitignore

## License

MIT License

## Contributing

Contact: https://t.me/magic_Cxsea

欢迎提交 Issue 和 Pull Request。
