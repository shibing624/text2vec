# -*- coding: utf-8 -*-
"""
@author:XuMing(xuming624@qq.com)
@description: 测试GPU显存释放问题
"""
import sys
import gc
import torch

sys.path.append('..')
from text2vec import SentenceModel


def print_gpu_memory(prefix=""):
    """打印当前GPU显存使用情况"""
    if torch.cuda.is_available():
        allocated = torch.cuda.memory_allocated() / 1024**2  # MB
        reserved = torch.cuda.memory_reserved() / 1024**2  # MB
        print(f"{prefix}GPU显存 - 已分配: {allocated:.2f} MB, 已保留: {reserved:.2f} MB")
        return allocated
    else:
        print(f"{prefix}CUDA不可用，无法测试GPU显存")
        return 0


def test_memory_leak():
    """测试显存泄漏问题"""
    print("=" * 60)
    print("测试GPU显存释放问题")
    print("=" * 60)
    print()
    
    if not torch.cuda.is_available():
        print("警告: CUDA不可用，此测试需要GPU环境")
        print("将继续运行但无法测试显存释放")
        print()
    
    # 初始化模型
    print("1. 初始化模型...")
    m = SentenceModel("shibing624/text2vec-base-chinese")
    initial_memory = print_gpu_memory("   ")
    print()
    
    # 测试不同数量的句子
    test_cases = [10, 50, 100, 200, 500]
    
    print("2. 测试不同数量的句子编码...")
    print("-" * 60)
    
    memory_records = []
    
    for num_sentences in test_cases:
        print(f"\n测试 {num_sentences} 个句子:")
        
        # 生成测试句子
        corpus = [f'这是第{i}个测试句子，用于验证GPU显存释放是否正常' for i in range(num_sentences)]
        
        # 编码前显存
        before_memory = print_gpu_memory("   编码前 - ")
        
        # 编码
        embeddings = m.encode(corpus, batch_size=32, show_progress_bar=False)
        
        # 编码后显存
        after_memory = print_gpu_memory("   编码后 - ")
        
        print(f"   结果shape: {embeddings.shape}")
        memory_increase = after_memory - before_memory
        print(f"   显存增长: {memory_increase:.2f} MB")
        
        # 记录编码后的显存（相对于初始显存）
        memory_from_initial = after_memory - initial_memory
        memory_records.append((num_sentences, memory_from_initial))
        
        # 清理Python变量
        del embeddings
        del corpus
        
        # 强制垃圾回收和清空缓存
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        
        # 清理后显存
        cleaned_memory = print_gpu_memory("   清理后 - ")
        final_increase = cleaned_memory - initial_memory
        print(f"   相对初始显存增长: {final_increase:.2f} MB")
    
    print()
    print("=" * 60)
    print("测试结果汇总:")
    print("-" * 60)
    print(f"{'句子数量':<10} {'编码后显存增长(MB)':<20}")
    print("-" * 60)
    for num_sentences, memory in memory_records:
        print(f"{num_sentences:<10} {memory:<20.2f}")
    print("-" * 60)
    print()
    
    # 分析结果
    print("结果分析:")
    if len(memory_records) >= 2:
        # 检查显存是否线性增长
        memory_diffs = [memory_records[i][1] - memory_records[i-1][1] 
                       for i in range(1, len(memory_records))]
        avg_diff = sum(memory_diffs) / len(memory_diffs)
        max_diff = max(memory_diffs)
        
        if max_diff > 50:  # 如果显存增长超过50MB
            print(f"  ⚠️  检测到显存持续增长! 平均增长: {avg_diff:.2f} MB, 最大增长: {max_diff:.2f} MB")
            print("  可能存在显存泄漏问题")
        else:
            print(f"  ✓ 显存释放正常! 平均增长: {avg_diff:.2f} MB (正常波动)")
            print("  显存能够正常释放，没有明显泄漏")
    
    print()
    print("预期结果:")
    print("  - 编码后的显存应该主要用于模型权重(固定)")
    print("  - 随着句子数量增加，显存增长应该很小")
    print("  - 清理后显存应该回到接近初始状态")
    print("=" * 60)


def test_batch_processing():
    """测试批处理过程中的显存变化"""
    print()
    print("=" * 60)
    print("测试批处理过程中的显存变化")
    print("=" * 60)
    print()
    
    if not torch.cuda.is_available():
        print("CUDA不可用，跳过批处理测试")
        return
    
    print("初始化模型...")
    m = SentenceModel("shibing624/text2vec-base-chinese")
    initial_memory = print_gpu_memory("   ")
    print()
    
    # 生成大量句子
    num_sentences = 1000
    corpus = [f'句子{i}：这是用于测试批处理显存释放的测试句子' for i in range(num_sentences)]
    
    print(f"编码 {num_sentences} 个句子 (batch_size=32, 显示进度)...")
    print()
    
    # 编码并显示进度
    embeddings = m.encode(corpus, batch_size=32, show_progress_bar=True)
    
    after_memory = print_gpu_memory("\n编码完成 - ")
    print(f"结果shape: {embeddings.shape}")
    print(f"相对初始显存增长: {after_memory - initial_memory:.2f} MB")
    
    # 清理
    del embeddings
    del corpus
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    
    final_memory = print_gpu_memory("清理后 - ")
    print(f"最终显存相对初始: {final_memory - initial_memory:.2f} MB")
    print()
    print("=" * 60)


if __name__ == '__main__':
    # 运行显存泄漏测试
    test_memory_leak()
    
    # 运行批处理测试
    test_batch_processing()
    
    print()
    print("测试完成!")
    print()
    print("提示:")
    print("  - 如果没有GPU环境，脚本仍会运行但无法测试显存")
    print("  - 建议在GPU环境下运行以验证显存释放效果")
    print("  - 修复后，显存应该只随模型权重增长，不随句子数量线性增长")
