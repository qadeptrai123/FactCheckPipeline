import os
import json
import torch
from transformers import (
    Qwen2VLForConditionalGeneration, 
    Qwen2_5_VLForConditionalGeneration,
    AutoProcessor, 
    AutoModelForCausalLM, 
    AutoTokenizer,
    BitsAndBytesConfig
)
# Import class từ file preprocessor.py bạn vừa tạo
from src.preprocessing.preprocessor import MultimodalPreprocessor


model_dir = "models" 
corpus_dir = "FinalDataset"
vlm_model_dir = os.path.join(model_dir, "qwen-vl-instruct")
quant_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_use_double_quant=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.float16 # Giữ nguyên compute dtype để đảm bảo độ chính xác khi tính toán
)


def main():
    print(os.path.join(model_dir, "qwen-vl-instruct"))
    print("="*50)
    print(" KHỞI TẠO HỆ THỐNG FACT-CHECKING PREPROCESSOR ")
    print("="*50)
    
    # ---------------------------------------------------------
    # BƯỚC 1: LOAD VLM (Xử lý ảnh) - Mặc định dùng Qwen2-VL
    # ---------------------------------------------------------
    print("\n[1/2] Đang tải VLM (Qwen2.5-VL-7B-Instruct)...")
    vlm_model_id = "Qwen/Qwen2.5-VL-7B-Instruct"
    
    vlm_model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        vlm_model_dir, # Sử dụng thư mục local đã tải sẵn
        device_map="auto", # Tự động chia VRAM
        torch_dtype=torch.float16, # Giữ nguyên dtype để đảm bảo độ chính xác khi tính toán
        # quantization_config=quant_config
    )
    vlm_processor = AutoProcessor.from_pretrained(vlm_model_dir)
    # ---------------------------------------------------------
    # BƯỚC 3: GẮN VÀO PIPELINE VÀ CHẠY TEST
    # ---------------------------------------------------------
    print("\n[2/2] Đang khởi tạo Pipeline...")
    pipeline = MultimodalPreprocessor(
        model=vlm_model,
        processor=vlm_processor,
    )

    # Dữ liệu Test
    test_claim = "Công an tỉnh Thái Bình đã khởi tố 10 đối tượng trong đường dây lừa đảo hỗ trợ vay vốn online."
    test_image_path = os.path.join(corpus_dir, "media/post_1_cmt_img_0.jpg") # Thay bằng đường dẫn ảnh thật của bạn
    
    # Tạo một file ảnh rỗng tạm thời để code không báo lỗi nếu bạn chưa có ảnh
    if not os.path.exists(test_image_path):
        from PIL import Image
        img = Image.new('RGB', (100, 100), color = 'red')
        img.save(test_image_path)
        print(f"[*] Đã tạo ảnh tạm {test_image_path} để test.")

    print("\n" + "-"*50)
    print(f"BẮT ĐẦU XỬ LÝ CLAIM:\n'{test_claim}'")
    print("-"*50)
    
    # Chạy Pipeline
    result = pipeline.process_claim(claim=test_claim, image_path=test_image_path)
    
    # In kết quả
    print("\n" + "="*20 + " KẾT QUẢ JSON CUỐI CÙNG " + "="*20)
    print(json.dumps(result, ensure_ascii=False, indent=4))

if __name__ == "__main__":
    # Đảm bảo bạn đã cài đặt thư viện cần thiết trước khi chạy:
    # pip install transformers accelerate qwen-vl-utils pillow
    main()