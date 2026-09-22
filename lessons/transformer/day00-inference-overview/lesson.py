from __future__ import annotations


def section(title: str) -> None:
    print(f"\n## {title}\n")


def code_block(value: str) -> None:
    print("```text")
    print(value)
    print("```")


def main() -> None:
    section("1. 本课只看一次推理")
    print("本课固定研究一个问题：LLM 收到 `我喜欢打` 后，怎样继续生成 `网球`？")
    print("这里讨论的是**推理**：模型参数已经训练完成，本次过程不会修改参数。")
    print("\n**输入：** `我喜欢打`")
    print("\n**最终可见输出：** `我喜欢打网球`")
    print("\n模型一次只预测一个 Token。使用本课程的 tokenizer 时，`网球` 是两个 Token，")
    print("所以完整生成需要两轮：先生成 `网`，再生成 `球`。")

    section("2. 先看完整流程，不展开公式")
    print("```mermaid")
    print("flowchart TD")
    print('    A["输入文本<br/>我喜欢打"]')
    print('    B["Tokenizer<br/>我｜喜欢｜打"]')
    print('    C["Embedding + 位置信息<br/>每个 Token 变成初始向量"]')
    print('    D["多层 Transformer<br/>最后位置获得上下文表示"]')
    print('    E["LM Head<br/>得到词表中每个 Token 的分数"]')
    print('    F["Softmax<br/>得到下一个 Token 的概率"]')
    print('    G["选择：网"]')
    print('    H["把 网 追加到输入<br/>再次执行同一流程"]')
    print('    I["选择：球"]')
    print("    A --> B --> C --> D --> E --> F --> G --> H --> I")
    print("```")

    section("3. 第一轮：从“我喜欢打”预测“网”")
    print("### 3.1 文本变成模型可以计算的输入\n")
    code_block(
        "文本：我喜欢打\n"
        "Token：[我] [喜欢] [打]\n"
        "Token ID：[7522, 69681, 15552]"
    )
    print("Token ID 只是词表索引。Embedding 层把每个 ID 查成向量，再加入位置信息。")

    print("\n### 3.2 Transformer 改写每个位置的表示\n")
    print("Self-Attention 让最后一个 Token `打` 读取前面的 `我`、`喜欢` 和自己。")
    print("经过 Attention、FFN、残差连接和多层堆叠后，最后位置得到一个上下文向量。")
    print("这个向量表达的不是孤立的“打”，而是当前上下文中的“我喜欢打”。")
    print("\n> Attention 的输出仍然是向量，不是文字，也不是“网球”的概率。")

    print("\n### 3.3 最后一个向量变成下一个 Token 的概率\n")
    print("LM Head 把最后位置的向量映射成整个词表的分数（logits），Softmax 再转成概率。")
    print("下面的数字只用于展示输出形式，不代表真实模型结果：")
    print("\n| 候选 Token | 示例概率 |")
    print("| --- | ---: |")
    print("| `网` | 42% |")
    print("| `球` | 18% |")
    print("| `游戏` | 12% |")
    print("| 其他 Token | 28% |")
    print("\n解码策略从概率分布中选出 `网`，再把它追加到输入末尾。")

    section("4. 第二轮：从“我喜欢打网”预测“球”")
    code_block(
        "新输入：我喜欢打网\n"
        "Token：[我] [喜欢] [打] [网]\n"
        "再次经过：Embedding -> Transformer -> LM Head -> Softmax\n"
        "选出下一个 Token：[球]"
    )
    print("把 `球` 追加后，用户最终看到 `我喜欢打网球`。")

    section("5. 每个组件接收什么、输出什么")
    print("| 组件 | 输入 | 输出 |")
    print("| --- | --- | --- |")
    print("| Tokenizer | 文本 | Token ID 序列 |")
    print("| Embedding | Token ID + 位置 | 初始向量序列 |")
    print("| Transformer 层 | 向量序列 | 融合上下文后的向量序列 |")
    print("| LM Head | 最后位置的向量 | 词表 logits |")
    print("| Softmax | logits | 下一个 Token 的概率分布 |")
    print("| 解码策略 | 概率分布 | 选中的下一个 Token |")
    print("| 生成循环 | 新 Token | 追加 Token 后再次预测 |")

    section("6. 训练暂时不进入这条主线")
    print("训练会使用完整文本 `我喜欢打网球`，让模型学习：看到 `我喜欢打` 时，")
    print("正确的下一个 Token 是 `网`；看到 `我喜欢打网` 时，正确答案是 `球`。")
    print("Loss、反向传播和参数更新将在推理主线走通后单独讲解。")

    section("7. 本课结论")
    code_block(
        "输入文本\n"
        "-> Token ID\n"
        "-> 初始向量\n"
        "-> 上下文向量\n"
        "-> 词表概率\n"
        "-> 选出一个 Token\n"
        "-> 追加后继续下一轮"
    )
    print("后续每一课只放大其中一个步骤，但始终回到这条推理链定位。")


if __name__ == "__main__":
    main()

