import torch
from modeling_llama import LlamaForCausalLM, LlamaConfig

def main():
    # 使用默认配置（非常小的模型，方便测试）
    config = LlamaConfig(
        vocab_size=32000,    # 自定义一个 vocab
        hidden_size=512,
        intermediate_size=1376,
        num_hidden_layers=2,
        num_attention_heads=8,
        max_position_embeddings=128,
        rms_norm_eps=1e-6,
    )

    # 初始化模型
    model = LlamaForCausalLM(config)

    # 随机生成 input_ids
    batch_size = 1
    seq_length = 10
    input_ids = torch.randint(0, config.vocab_size, (batch_size, seq_length))

    # 前向推理
    outputs = model(input_ids)

    # 查看 logits
    print(f"Input IDs: {input_ids}")
    print(f"Logits shape: {outputs.logits.shape}")  # (batch_size, seq_length, vocab_size)

if __name__ == "__main__":
    main()
