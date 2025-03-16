from openai import OpenAI
import chromadb
from config.settings import Config
from typing import List, Dict, Optional
import numpy as np
from functools import lru_cache
import hashlib
import json
import time

class VectorRetriever:
    def __init__(self, cache_ttl: int = 3600):
        self.client = OpenAI(api_key=Config.OPENAI_API_KEY)
        self.chroma_client = chromadb.PersistentClient(path=Config.PERSIST_DIR)
        self.collection = self.chroma_client.get_collection(Config.COLLECTION_NAME)
        self.cache_ttl = cache_ttl  # 缓存过期时间（秒）
        self.cache = {}  # 简单的内存缓存

    def _generate_cache_key(self, query: str, top_k: int, filters: Optional[Dict], min_score: float) -> str:
        """生成缓存键"""
        cache_dict = {
            'query': query,
            'top_k': top_k,
            'filters': filters,
            'min_score': min_score
        }
        cache_str = json.dumps(cache_dict, sort_keys=True)
        return hashlib.md5(cache_str.encode()).hexdigest()

    def _get_from_cache(self, cache_key: str) -> Optional[List[Dict]]:
        """从缓存中获取结果"""
        if cache_key in self.cache:
            cached_result, timestamp = self.cache[cache_key]
            if time.time() - timestamp < self.cache_ttl:
                return cached_result
            else:
                # 删除过期缓存
                del self.cache[cache_key]
        return None
    
    def _save_to_cache(self, cache_key: str, results: List[Dict]):
        """保存结果到缓存"""
        self.cache[cache_key] = (results, time.time())

    # 在VectorRetriever类中添加新方法
    def recommend_agent(self, user_prompt: str) -> Dict:
        """
        根据用户的Prompt推荐合适的agent类型
        Args:
            user_prompt: 用户的输入prompt
        Returns:
            Dict: 包含推荐的agent类型和描述的字典
        """
        system_prompt = """你是一个AI助手专家，你的任务是根据用户的需求推荐最合适的agent类型（从行业领域角度划分的类型）。
        分析用户的需求，并推荐一个或多个最适合的agent类型。在描述的时候尽可能减少AI词汇的属性，因为在数据库中每个agent描述都自带AI字眼。
        返回格式应包含：
        1. 推荐的agent类型。
        2. 从行业角度描述该专业领域和能力。
        """

        user_message = f"基于以下用户需求，请推荐合适的AI agent类型：{user_prompt}"

        response = self.client.chat.completions.create(
            model="gpt-4o",
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_message}
            ],
            temperature=0.7,
            max_tokens=500
        )

        recommendation = response.choices[0].message.content

        # 将返回结果构造成字典格式
        return {
            "user_prompt": user_prompt,
            "agent_recommendation": recommendation
        }

    @lru_cache(maxsize=1000)
    def get_embedding(self, text: str) -> list:
        response = self.client.embeddings.create(
            input=text,
            model="text-embedding-ada-002"
        )
        return response.data[0].embedding

    def search(self,
               query: str,
               top_k: int = 3,
               filters: Optional[Dict] = None,
               min_score: float = 0.4) -> List[Dict]:
        """
        混合检索方法 -> 带缓存的混合检索方法
        Args:
            query: 查询文本
            top_k: 返回结果数量
            filters: 元数据过滤条件, 如 {"category": "finance"}
            min_score: 最小相似度阈值
        """

        # 生成缓存键
        cache_key = self._generate_cache_key(query, top_k, filters, min_score)
        
        # 尝试从缓存获取结果
        cached_results = self._get_from_cache(cache_key)
        if cached_results is not None:
            return cached_results
        
        # 如果缓存未命中，执行检索, 生成查询向量
        query_embedding = self.get_embedding(query)
        # 构建查询参数
        query_params = {
            "query_embeddings": [query_embedding],
            "n_results": top_k * 2
        }
        
        # 只有在有过滤条件时才添加 where 参数
        if filters:
            query_params["where"] = filters
            
        results = self.collection.query(
            **query_params
        )
        # 整理检索结果
        search_results = []
        for i in range(len(results["ids"][0])):
            # 计算归一化相似度分数 (1 - distance)
            similarity_score = 1 - results["distances"][0][i]
            # 应用相似度阈值过滤
            if similarity_score < min_score:
                continue
            # 获取元数据
            metadata = results["metadatas"][0][i] if "metadatas" in results else {}
            result = {
            "id": results["ids"][0][i],
                "score": similarity_score,
                "document": results["documents"][0][i],
                "metadata": metadata
            }
           
            search_results.append(result)
        # 结果排序（综合考虑相似度分数和业务规则）
        search_results = self._rank_results(search_results)
        final_results = search_results[:top_k]
        # 保存到缓存
        self._save_to_cache(cache_key, final_results)
        return final_results
    
    def enhanced_search(self,
                        user_prompt: str,
                        top_k: int = 3,
                        filters: Optional[Dict] = None,
                        min_score: float = 0.4) -> Dict:
            """
            增强版检索方法，结合agent推荐和向量检索
            Args:
                user_prompt: 用户的原始问题
                top_k: 返回结果数量
                filters: 元数据过滤条件
                min_score: 最小相似度阈值
            Returns:
                Dict: 包含agent推荐和检索结果的字典
            """
            # 1. 首先获取agent推荐
            agent_recommendation = self.recommend_agent(user_prompt)
            
            # 2. 构建增强查询文本
            enhanced_query = f"""
            用户需求: {user_prompt}
            推荐Agent类型和描述: {agent_recommendation['agent_recommendation']}
            """
            
            # 3. 使用增强查询进行向量检索
            search_results = self.search(
                query=enhanced_query,
                top_k=top_k,
                filters=filters,
                min_score=min_score
            )
            
            # 4. 返回完整结果
            return {
                "original_prompt": user_prompt,
                "agent_recommendation": agent_recommendation['agent_recommendation'],
                "search_results": search_results
            }
    
    def _rank_results(self, results: List[Dict]) -> List[Dict]:
        """
        对检索结果进行排序
        排序规则：
        1. 优先级 priority (如果存在)
        2. 相似度分数 score
        """
        # def get_sort_key(item):
        #     priority = item.get("metadata", {})
        #     score = item["score"]
        #     return (priority, score)
        # return sorted(results, key=get_sort_key, reverse=True)
        return sorted(results, key=lambda x: x["score"], reverse=True)

if __name__ == "__main__":
    """测试基本检索功能"""
    retriever = VectorRetriever()
    
    # 测试增强版检索功能
    test_prompt = "我要找工作，帮我写简历，要找金融科技工作。"
    results = retriever.enhanced_search(test_prompt, top_k=3)
    
    # 打印完整结果
    print("\n=== 增强检索结果 ===")
    print(f"原始问题: {results['original_prompt']}")
    print("\n推荐Agent:")
    print(results['agent_recommendation'])
    print("\n相关文档检索结果:")
    for i, res in enumerate(results['search_results'], 1):
        print(f"\n结果 {i}:")
        print(f"ID: {res['id']}")
        print(f"相似度: {res['score']:.4f}")
        print(f"内容: {res['document'][:80]}...")
        print("="*50)