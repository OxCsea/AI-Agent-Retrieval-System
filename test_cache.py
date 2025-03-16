import time
from retrieval import VectorRetriever

def test_cache_effectiveness():
    # 初始化检索器，设置较短的缓存时间以便测试
    retriever = VectorRetriever(cache_ttl=5)  # 5秒的缓存时间
    
    test_query = "我需要一个金融分析师帮我分析股市"
    
    # 第一次查询（无缓存）
    print("第一次查询（无缓存）:")
    start_time = time.time()
    first_results = retriever.search(test_query, top_k=3)
    first_query_time = time.time() - start_time
    print(f"查询时间: {first_query_time:.2f} 秒")
    
    # 立即进行第二次相同查询（应该使用缓存）
    print("\n第二次查询（应使用缓存）:")
    start_time = time.time()
    second_results = retriever.search(test_query, top_k=3)
    second_query_time = time.time() - start_time
    print(f"查询时间: {second_query_time:.2f} 秒")
    
    # 验证两次结果是否相同
    print("\n结果验证:")
    results_match = first_results == second_results
    print(f"结果是否完全相同: {results_match}")
    
    # 验证缓存加速效果
    speedup = (first_query_time - second_query_time)/first_query_time 
    print(f"缓存加速比: {speedup:.2f}x")
    
    # 等待缓存过期
    print("\n等待缓存过期（5秒）...")
    time.sleep(6)
    
    # 缓存过期后的查询（应该重新检索）
    print("\n缓存过期后的查询:")
    start_time = time.time()
    third_results = retriever.search(test_query, top_k=3)
    third_query_time = time.time() - start_time
    print(f"查询时间: {third_query_time:.2f} 秒")
    
    # 打印缓存状态
    print("\n当前缓存状态:")
    print(f"缓存条目数量: {len(retriever.cache)}")

def test_different_queries():
    retriever = VectorRetriever(cache_ttl=300)  # 5分钟缓存
    
    # 测试不同的查询
    queries = [
        "我需要一个金融分析师",
        "帮我写一篇营销文案",
        "我需要法律咨询"
    ]
    
    print("测试不同查询的缓存效果:")
    for query in queries:
        # 第一次查询
        start_time = time.time()
        retriever.search(query, top_k=3)
        first_time = time.time() - start_time
        
        # 重复查询
        start_time = time.time()
        retriever.search(query, top_k=3)
        second_time = time.time() - start_time
        
        print(f"\n查询: {query}")
        print(f"首次查询时间: {first_time:.2f} 秒")
        print(f"缓存查询时间: {second_time:.2f} 秒")
        print(f"加速比: {((first_time-second_time)/first_time):.2f}x")

if __name__ == "__main__":
    print("=== 测试缓存有效性 ===")
    test_cache_effectiveness()
    
    print("\n=== 测试不同查询的缓存效果 ===")
    test_different_queries()
