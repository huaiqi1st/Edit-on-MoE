from transformers import AutoModelForCausalLM, AutoTokenizer

model_name = "mistralai/Mixtral-8x7B-v0.1"
save_path = "../models/mixtral-8x7b"

# 下载模型
model = AutoModelForCausalLM.from_pretrained(model_name)
model.save_pretrained(save_path)

# 下载分词器
tokenizer = AutoTokenizer.from_pretrained(model_name)
tokenizer.save_pretrained(save_path)

print(f"模型已下载到 {save_path}")