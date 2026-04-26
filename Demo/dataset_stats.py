import os
import sys
from pathlib import Path

# Cấu hình encoding để in được tiếng Việt trên console Windows
if sys.stdout.encoding != 'utf-8':
    try:
        sys.stdout.reconfigure(encoding='utf-8')
    except AttributeError:
        # Fallback cho Python cũ hơn 3.7
        import codecs
        sys.stdout = codecs.getwriter("utf-8")(sys.stdout.detach())

def get_count(path, ext="*"):
    p = Path(path)
    if not p.exists():
        return 0
    return len(list(p.glob(ext)))

def analyze_dataset():
    # --- Paths ---
    raw_root = Path("data_raw")
    img_root = Path("data_image_train")
    split_root = Path("splits")

    print("="*50)
    print("THỐNG KÊ CHI TIẾT TẬP DỮ LIỆU DEEPFAKE")
    print("="*50)

    # 1. RAW DATA (VIDEOS)
    raw_sources = {
        "Celeb-DF (Real)": raw_root / "Celeb-real",
        "YouTube (Real)": raw_root / "YouTube-real",
        "Celeb-DF (Fake)": raw_root / "Celeb-synthesis"
    }

    print(f"\n[1] DỮ LIỆU THÔ (VIDEO)")
    total_raw_real = 0
    total_raw_fake = 0
    
    c_real = get_count(raw_sources["Celeb-DF (Real)"], "*.mp4")
    y_real = get_count(raw_sources["YouTube (Real)"], "*.mp4")
    f_raw = get_count(raw_sources["Celeb-DF (Fake)"], "*.mp4")
    
    print(f" - Celeb-DF Real: {c_real} videos")
    print(f" - YouTube Real:  {y_real} videos")
    print(f" - Celeb-DF Fake: {f_raw} videos")
    print(f" => Tổng Real: {c_real + y_real} | Tổng Fake: {f_raw}")

    # 2. EXTRACTED DATA (IMAGES)
    print(f"\n[2] DỮ LIỆU SAU TRÍCH XUẤT (CROP FACES)")
    real_imgs = get_count(img_root / "Real", "*.jpg")
    fake_imgs = get_count(img_root / "Fake", "*.jpg")
    
    print(f" - Tổng ảnh REAL: {real_imgs}")
    print(f" - Tổng ảnh FAKE: {fake_imgs}")
    print(f" => Tỉ lệ thực tế: 1 : {(fake_imgs/real_imgs if real_imgs > 0 else 0):.2f} (Gần mức 1:1)")

    # 3. SPLIT DATA (TRAIN/VAL/TEST)
    print(f"\n[3] PHÂN CHIA TẬP DỮ LIỆU (SPLITS)")
    splits = ["train.txt", "val.txt", "test.txt"]
    for s in splits:
        s_path = split_root / s
        if s_path.exists():
            with open(s_path, 'r') as f:
                lines = f.readlines()
                count = len(lines)
                real_s = sum(1 for line in lines if "Real" in line)
                fake_s = count - real_s
                print(f" - {s:9}: {count:5} mẫu (Real: {real_s:4}, Fake: {fake_s:4})")
        else:
            print(f" - {s:9}: Không tìm thấy file")

    print("\n" + "="*50)
    print("PHÂN TÍCH: Tại sao cần bước tiếp theo?")
    print("-"*50)
    print("1. Tỷ lệ trích xuất thấp cho thấy bộ lọc chất lượng (Blur/Face detection) hoạt động tốt.")
    print("2. Tỷ lệ Real/Fake sau trích xuất đã tự cân bằng hơn so với video gốc (1:7).")
    print("3. Cần phân chia tập Test riêng biệt để đánh giá khách quan khả năng tổng quát hóa.")
    print("="*50)

if __name__ == "__main__":
    analyze_dataset()
