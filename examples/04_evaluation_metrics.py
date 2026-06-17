"""
示例4: 检索评估指标
演示 nDCG, MRR, Precision@k, Recall@k, MAP 等指标计算
"""

import sys
sys.path.insert(0, '../src')

from vectorsearch import RetrievalEvaluator

def main():
    print("=" * 60)
    print("示例4: 检索系统评估指标")
    print("=" * 60)
    
    # 1. 创建评估器
    evaluator = RetrievalEvaluator()
    
    # 2. 模拟检索结果评估数据
    # 模拟3个查询的检索结果和相关性标注
    test_queries = [
        {
            "query": "向量数据库",
            "retrieved": ["doc1", "doc2", "doc3", "doc4", "doc5"],
            "relevant": {"doc1", "doc3"},  # 相关文档集合
            "relevances": [3, 1, 2, 0, 0]  # 相关性分级 (0-3)
        },
        {
            "query": "混合检索",
            "retrieved": ["doc2", "doc4", "doc5", "doc1", "doc3"],
            "relevant": {"doc2", "doc5"},
            "relevances": [3, 0, 2, 1, 0]
        },
        {
            "query": "RAG技术",
            "retrieved": ["doc3", "doc1", "doc2", "doc5", "doc4"],
            "relevant": {"doc3", "doc4"},
            "relevances": [3, 1, 0, 0, 2]
        }
    ]
    
    print("\n[1] 评估数据集:")
    for q in test_queries:
        print(f"   查询: '{q['query']}'")
        print(f"     检索结果: {q['retrieved']}")
        print(f"     相关文档: {q['relevant']}")
        print()
    
    # 3. 执行评估
    print("[2] 执行评估...")
    result = evaluator.evaluate(
        test_queries,
        k_values=[1, 3, 5]
    )
    
    # 4. 打印评估报告
    print("\n[3] 评估报告:")
    evaluator.print_report(result)
    
    # 5. 指标说明
    print("\n[4] 指标说明:")
    print("   • nDCG: 归一化折损累计增益，衡量排序质量 [0-1]")
    print("   • MRR: 平均倒数排名，第一个相关文档的位置倒数")
    print("   • MAP: 平均准确率，所有查询的AP平均值")
    print("   • Precision@k: 前k个结果中的相关比例")
    print("   • Recall@k: 前k个结果中召回的相关文档比例")
    
    print("\n" + "=" * 60)
    print("评估演示完成!")
    print("这些指标是搜索/推荐系统的工业界标准评估方法")
    print("=" * 60)

if __name__ == "__main__":
    main()
