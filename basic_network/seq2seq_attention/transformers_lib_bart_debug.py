"""
# 使用transformers库调试Bart模型流程
"""
import os
from transformers import BartTokenizer, BartForConditionalGeneration


if __name__ == "__main__":
    tokenizer = BartTokenizer.from_pretrained('facebook/bart-large')
    TXT = "My friends are <mask> but they eat too many carbs."
    model = BartForConditionalGeneration.from_pretrained('facebook/bart-large')
    input_ids = tokenizer.batch_encode_plus([TXT], return_tensors='pt')['input_ids']
    logits = model(input_ids)[0]
    masked_index = (input_ids[0] == tokenizer.mask_token_id).nonzero().item()
    probs = logits[0, masked_index].softmax(dim=0)
    values, predictions = probs.topk(5)
    tokenizer.decode(predictions).split()
