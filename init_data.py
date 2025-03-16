import chromadb
from config.settings import Config
from openai import OpenAI
import json
import logging
from chromadb.config import Settings

logger = logging.getLogger("aiagent_log")

# 代码优化：不要没一个类调用一次openai，能否只调用一次，然后生成多个类Prompt
# 存储数据 时 这里的documents 是什么意思
# C:\Users\MagicKD\.cache\chroma\onnx_models\all-MiniLM-L6-v2\onnx.tar.gz:
# 这个路径表明 ChromaDB 在首次使用时会自动下载并缓存所需的嵌入模型,以便后续使用。all-MiniLM-L6-v2 是一个常用的句子转换模型,用于将文本转换为向量形式。
# 能否调用openai生成Embedding，然后调用chromadb存储数据

def generate_agent_info(categories: list, client: OpenAI) -> dict:
    """
    使用OpenAI生成指定类别的agent信息
    """
    system_massage =f"""你是一个AI助手生成器。请为以下所有类别的智能体生成一个新的AI助手信息，每个AI助手信息有对应合适的agent ID和system prompt。
    类别列表: {categories}
    返回格式应该是JSON数组，每个类别对应一个对象，包含:
    - id: 类别缩写_编号 (例如: finance_001)
    - system_prompt: 详细的系统提示词，描述agent的专业领域和能力。用中文描述。
    - category: agent所属的类别
    请严格按照以下 JSON 格式返回数据：
    {{"agents": [
        {{"id": "agent1 id",
        "system_prompt": "agent1 description",
        "category": "agent1 category"}},
        {{"id": "agent2 id",
        "system_prompt": "agent2 description",
        "category": "agent2 category"}}
    ]}}"""

    response = client.chat.completions.create(
            model="gpt-4o",
            messages=[
                {"role": "system", "content": "你是一个专业的智能体设计专家。"},
                {"role": "user", "content": system_massage}
            ],
            temperature=0.7,
            response_format={ "type": "json_object" }
        )
    # 解析返回的JSON
    response_data = json.loads(response.choices[0].message.content)
    agent_info = response_data['agents']
    # print(agent_info)
    return agent_info

def get_embedding(agent_info: list, client: OpenAI) -> list:
    """
    使用OpenAI生成文本的嵌入向量
    """
    documents = [agent["system_prompt"] for agent in agent_info]

    # for doc in documents:
    #     response = client.embeddings.create(
    #         model="text-embedding-ada-002",
    #         input=doc
    #     )
    #     embeddings.append(response.data[0].embedding)

    # 批量处理所有文档，而不是逐个处理 
    response = client.embeddings.create(
        model="text-embedding-ada-002",
        input=documents
    )

    embeddings = [data.embedding for data in response.data]
    return embeddings

class AgentInitializationError(Exception):
    """自定义异常类，用于处理agent初始化过程中的错误"""
    pass

def initialize_agents(categories: list[str], client: OpenAI):
    """
    根据给定的类别列表初始化agents
    """

    # 为每个类别生成agent信息
    agent_info = generate_agent_info(categories, client)
    
    # 连接ChromaDB
    logger.info("连接ChromaDB...")
    try:
        db_client = chromadb.PersistentClient(path=Config.PERSIST_DIR)
    except Exception as e:
        print(f"Creating new client with custom settings: {e}")
        db_client = chromadb.PersistentClient(path=Config.PERSIST_DIR, 
                                        settings=Settings(
                    anonymized_telemetry=False,
                    allow_reset=True
                ))
    try:
        collection = db_client.get_collection(Config.COLLECTION_NAME)
    except Exception as e:
        print(f"Creating new collection: {Config.COLLECTION_NAME}")
        collection = db_client.create_collection(
            name=Config.COLLECTION_NAME,
            metadata={"description": "AI Agents information"}
        )
    print("ChromaDB连接成功！")

    # 存储数据
    try:
        logger.info("准备数据...")
        documents = [agent["system_prompt"] for agent in agent_info]
        metadatas = [{"category": agent["category"]} for agent in agent_info]
        ids = [agent["id"] for agent in agent_info]
        # 将system_prompt转成Embedding
        embeddings = get_embedding(agent_info, client)

        logger.info("正在存储agent数据...")
        collection.add(
            documents=documents, 
            # 通过embeddings参数指定每条文本数据的向量
            embeddings=embeddings,
            metadatas=metadatas,
            ids=ids
        )
        logger.info(f"成功初始化 {len(agent_info)} 个agents")
        return agent_info
    except Exception as e:
        logger.error(f"Agent初始化过程发生错误: {str(e)}")
        raise AgentInitializationError(f"Agent初始化失败: {str(e)}")

# 使用示例
if __name__ == "__main__":
    # categories = ["finance", "law"]
    categories = ["finance", "law", "medical", "technology", "education", "career", "fashion", "travel", \
                  "politics", "entertainment", "mental health"]
    # agents = initialize_agents(categories)
    # for agent in agents:
    #     print(f"\nCategory: {agent['category']}")
    #     print(f"ID: {agent['id']}")
    #     print(f"System Prompt: {agent['system_prompt'][:100]}...")
    client = OpenAI(api_key=Config.OPENAI_API_KEY)
    # agent_info = generate_agent_info(categories, client)
    # embeddings = get_embedding(client, agent_info)
    agent_info = initialize_agents(categories, client)