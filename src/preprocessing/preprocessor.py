import os
import re
import json
import torch
from qwen_vl_utils import process_vision_info

# =====================================================================
# MARKDOWN PROMPT TEMPLATES
# =====================================================================

VLM_IMAGE_ANALYSIS_PROMPT = """
# ROLE
You are an objective visual observer for a fact-checking system.

# TASK
Describe EXACTLY what you see in the provided image. Focus on physical details.

# CONSTRAINTS
- Focus on: people, clothing, actions, visible text, and surroundings.
- Language: Vietnamese.
- Format: A simple bulleted list. 
- NO continuous paragraphs.
- NO interpretations or assumptions.

# OUTPUT
(Danh sách gạch đầu dòng các sự kiện thấy trong ảnh):
"""

LLM_FACT_EXTRACTION_PROMPT = """
# ROLE
You are a logic analyst.

# TASK
Break down the claim into verifiable atomic facts.

# RULES
- Each fact must be a single, complete sentence.
- Language: Vietnamese.
- Focus on: subjects, actions, locations, and time.

# EXAMPLE
**Claim**: "At 10 PM yesterday, John stole a phone at the park and ran away."
**Thinking**: The claim has time (10 PM), subject (John), action 1 (stole phone), location (park), action 2 (ran away).
**Facts**:
1. Sự việc diễn ra vào lúc 10 giờ tối ngày hôm qua.
2. Đối tượng tên là John.
3. John đã thực hiện hành vi trộm điện thoại.
4. Sự việc xảy ra tại khu vực công viên.

# YOUR TASK
**Claim**: "{user_claim}"
**Thinking**:
"""

LLM_JSON_RAG_PROMPT = """
# ROLE
You are a data architect for a RAG (Retrieval-Augmented Generation) system.

# INPUT DATA
- **Claim**: "{user_claim}"
- **Image Context**: "{image_context}"
- **Text Facts**: "{atomic_facts}"

# TASK
Create a Preprocessing JSON object based on the input.

# JSON FIELD INSTRUCTIONS
1. `image_facts`: Extract bullet points from 'Image Context'.
2. `text_facts`: Extract sentences from 'Text Facts'.
3. `normalized_facts`: Merged array of image and text facts.
4. `alignment`: Start with ["Khớp hoàn toàn"], ["Không khớp"], or ["Khớp một phần"], then explain in 1 sentence.
5. `rag_queries`: 3 search keywords (2-5 words each) based ONLY on the Claim.
6. `hyde_doc`: 2 hypothetical documents. Doc 1: News style. Doc 2: Internal report style. (Based ONLY on Claim).

# CONSTRAINTS
- Return ONLY a valid JSON object.
- NO explanations outside the JSON.
- Language: Vietnamese.

# OUTPUT FORMAT
```json
{{
  "claim": "{user_claim}",
  "image_provided": {img_provided_str},
  "image_facts": [],
  "text_facts": [],
  "normalized_facts": [],
  "alignment": [""],
  "rag_queries": [],
  "hyde_doc": []
}}
"""

# =====================================================================
# MULTIMODAL PREPROCESSOR CLASS
# =====================================================================

class MultimodalPreprocessor:
    def __init__(self, model, processor):
        self.model = model
        self.processor = processor

    def _run_inference(self, messages, max_tokens=2048):
        text = self.processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        
        # Qwen-VL xử lý cả có ảnh lẫn không có ảnh qua hàm này
        image_inputs, video_inputs = process_vision_info(messages)
        
        inputs = self.processor(
            text=[text],
            images=image_inputs,
            videos=video_inputs,
            padding=True,
            return_tensors="pt",
        ).to(self.model.device)

        with torch.no_grad():
            generated_ids = self.model.generate(
                **inputs, 
                max_new_tokens=max_tokens, 
                temperature=0.3, 
                do_sample=True, 
                repetition_penalty=1.1
            )
        
        generated_ids_trimmed = [
            out_ids[len(in_ids):] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
        ]
        output_text = self.processor.batch_decode(
            generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
        )
        return output_text[0]

    def process_claim(self, claim, image_path=None):
        # 1. Image Analysis (Nếu có ảnh)
        has_image = False
        image_context = "No image provided."
        
        if image_path and os.path.exists(image_path):
            has_image = True
            msg_vlm = [{"role": "user", "content": [
                {"type": "image", "image": f"file://{image_path}"}, 
                {"type": "text", "text": VLM_IMAGE_ANALYSIS_PROMPT} # Thay bằng VLM_IMAGE_ANALYSIS_PROMPT của bạn
            ]}]
            image_context = self._run_inference(msg_vlm)

        # 2. Fact Extraction (Thuần Text - Vẫn dùng Qwen)
        formatted_fact_prompt = LLM_FACT_EXTRACTION_PROMPT.format(user_claim=claim) # Thay bằng LLM_FACT_EXTRACTION_PROMPT của bạn
        msg_facts = [{"role": "user", "content": [{"type": "text", "text": formatted_fact_prompt}]}]
        facts_raw = self._run_inference(msg_facts)
        atomic_facts = facts_raw.split("Facts:")[-1].strip() if "Facts:" in facts_raw else facts_raw

        # 3. JSON Generation (Thuần Text - Vẫn dùng Qwen)
        img_str = "true" if has_image else "false"
        formatted_json_prompt = LLM_JSON_RAG_PROMPT.format(
            user_claim=claim,
            image_context=image_context,
            atomic_facts=atomic_facts,
            img_provided_str=img_str
        )
        msg_json = [{"role": "user", "content": [{"type": "text", "text": formatted_json_prompt}]}]
        json_raw = self._run_inference(msg_json)

        json_match = re.search(r'\{.*\}', json_raw, re.DOTALL)
        if json_match:
            try:
                return json.loads(json_match.group(0))
            except:
                return {"error": "JSON Decode Failed", "raw": json_raw}
        return {"error": "No JSON found", "raw": json_raw}