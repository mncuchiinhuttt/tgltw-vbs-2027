---
name: VLM Prompting & Local Execution
description: Guidelines for constructing structured prompts (JSON outputs, YES/NO responses) and managing local Vision-Language Models offline.
---

# VLM Prompting & Local Execution Skill

This skill provides guidelines for working with Vision-Language Models (VLM) such as Qwen3-VL and OpenAI GPT-series models in the pipeline.

## Prompt Engineering Patterns

### 1. JSON Extraction Prompt
When requesting structured attributes from a frame, ensure the prompt explicitly defines the JSON structure and demands only raw JSON output without conversational fluff:

```
Analyze this frame and extract in JSON:
{
  "objects": ["xe máy", "ô tô", "người"],
  "text_on_screen": ["51-B1 234.56"],
  "colors_dominant": ["đỏ", "trắng"],
  "count_people": 3,
  "scene_type": "đường phố ban ngày"
}
Only output JSON, no explanation or markdown code blocks.
```

Always clean the output before parsing with `json.loads()`:
```python
raw_output = raw_output.strip()
if raw_output.startswith("```json"):
    raw_output = raw_output[7:]
if raw_output.endswith("```"):
    raw_output = raw_output[:-3]
data = json.loads(raw_output.strip())
```

### 2. Yes/No Scoring Verification Prompt
For Type 2 VQA candidate verification, use binary prompts specifying confidence scores:

```
Question: {question_text}
Answer YES or NO, then give confidence 0-1.
Format: {"answer": "YES/NO", "confidence": 0.9, "reason": "..."}
```

Calculate scores:
- If answer is `"YES"`: score = `confidence`
- If answer is `"NO"`: score = `1.0 - confidence`

---

## Local Offline Execution (Qwen3-VL)

### 1. Memory Optimization
Qwen models (especially 7B and larger) require significant GPU memory. 
- Use `torch.bfloat16` or `torch.float16` to reduce VRAM requirements.
- Set `device_map="auto"` in `from_pretrained` to dynamically offload parts of the model to CPU if necessary.
- In batch operations, process sequentially rather than in large parallel batches to avoid Out of Memory (OOM) failures.

### 2. Chat Templates
Always use `processor.apply_chat_template` to format messages for instruct models:
```python
messages = [
    {
        "role": "user",
        "content": [
            {"type": "image", "image": pil_image},
            {"type": "text", "text": prompt}
        ]
    }
]
formatted_text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
```
