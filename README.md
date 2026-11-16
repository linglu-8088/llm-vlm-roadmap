# LLM & VLM Learning Roadmap

## 项目介绍

本项目用于记录从计算机视觉（CV）转向大语言模型（LLM）和视觉语言模型（VLM）的学习过程。

**学习周期**：12 个月。

**目标方向**：Multimodal / VLM 算法工程师。

## 技术栈

- Python
- PyTorch
- Hugging Face Transformers
- Hugging Face Datasets
- CUDA

## Week 01：语言模型基础

**学习内容**：

- Token 与 Embedding
- Cross Entropy 与 Perplexity
- Tokenizer 分词实验
- 文本数据处理
- 小型语言模型推理

## 环境配置

创建 Python 3.11 虚拟环境并安装依赖：

```bash
python -m pip install -r requirements.txt
```

GPU 环境需要根据实际 CUDA 平台选择 PyTorch 安装版本。

## 学习记录

每周更新学习代码、实验结果、遇到的问题和总结。

1. Token 是模型处理文本的离散符号单元。
2. Tokenizer 将文本转换成 Token ID，支持反向解码。
3. 不同模型可能使用不同词表和分词算法，相同文本的 Token 数可能不同。

### Day 02：语言模型基础

#### 1. Embedding

Embedding 将离散 Token ID 转换为连续向量。

例如，输入 3 个 Token ID，Embedding 维度设置为 8，则输出形状为 `[3, 8]`。

#### 2. Logits 与 Softmax

Logits 是模型输出的原始预测分数，可以为负数。

Softmax 将 Logits 转换为概率分布，所有候选 Token 的概率之和为 1。

#### 3. Cross Entropy

交叉熵用于衡量模型预测与真实标签的差距。

正确 Token 的预测概率越高，交叉熵越低。

#### 4. Perplexity

困惑度通常由平均交叉熵计算：

`PPL = exp(平均 Cross Entropy)`

PPL 越低，通常代表模型在相同评估设置下对测试文本的预测越好。

#### 5. 反向传播

- `optimizer.zero_grad()`：清除旧梯度
- `loss.backward()`：计算参数梯度
- `optimizer.step()`：更新参数

#### 6. TinyLM 实验

**模型架构**：Embedding → Linear → Logits

**训练数据**：`<BOS> 我喜欢 深度 学习 <EOS>`

**实验结果**：

- 初始 Loss：1.9691
- 初始 PPL：7.1642
- Step 20 Loss：0.0002
- 最终 Loss：接近 0
- 最终 PPL：接近 1
- 生成结果：`<BOS> 我喜欢 深度 学习 <EOS>`

#### 7. 实验结论

TinyLM 能够学习简单的 Token 转移关系，但没有 Attention 机制，无法根据更长的上下文调整预测。

下一步需要学习真实文本的 Tokenization、文本数据处理和 Transformer 上下文建模机制。

### Day 03 补充实验：BPE 内部切分分析

**测试文本**：`数据处理模块正在分析测试结果。`

#### 实验结果

- Qwen2.5：8 Tokens
- DistilGPT2：35 Tokens
- Token 数量比值：4.375

#### Qwen2.5 切分

`数据 | 处理 | 模块 | 正在 | 分析 | 测试 | 结果 | 。`

#### DistilGPT2 切分

一个中文字符可能被拆分成多个 Token，部分 Token 仅表示 UTF-8 字符的一部分字节。

因此，单独解码部分 Token 时会得到 Unicode 替代字符 `�`。

#### 结论

1. 不同 Tokenizer 的分词粒度可能存在显著差异。
2. 中文词表覆盖和 BPE 合并规则影响 Token 数量。
3. 单个 Token 不一定对应完整字符。
4. Byte-level Tokenizer 可以通过完整 Token 序列恢复原始文本。
5. Token 数量会影响模型上下文窗口占用和计算开销，但不能直接用于判断模型的整体能力。
