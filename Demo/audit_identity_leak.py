import os

def extract_ids(file_path):
    ids = set()
    with open(file_path, "r") as f:
        for line in f:
            path = line.split(" ")[0]
            filename = os.path.basename(path)
            
            if "FF_" in filename:
                # Common patterns: 
                # FF_real_294_f0.jpg -> ID 294
                # FF_fake_FaceSwap_684_717_f1.jpg -> IDs 684, 717
                parts = filename.split("_")
                for p in parts:
                    if p.isdigit():
                        ids.add(f"FF_{p}")
            
            elif "Celeb-" in filename:
                # Pattern: Celeb-real_id3_0008.mp4_f6.jpg -> ID id3
                parts = filename.split("_")
                for p in parts:
                    if p.startswith("id") and p[2:].isdigit():
                        ids.add(f"Celeb_{p}")
    return ids

def audit():
    train_file = "splits/train.txt"
    test_file = "splits/test.txt"
    
    if not os.path.exists(train_file) or not os.path.exists(test_file):
        print("Error: Split files not found.")
        return
        
    train_ids = extract_ids(train_file)
    test_ids = extract_ids(test_file)
    
    intersection = train_ids.intersection(test_ids)
    
    print("="*50)
    print("IDENTITY LEAKAGE AUDIT REPORT")
    print("="*50)
    print(f"Total Unique IDs in TRAIN: {len(train_ids)}")
    print(f"Total Unique IDs in TEST:  {len(test_ids)}")
    print(f"Number of Leaked IDs:      {len(intersection)}")
    
    if len(test_ids) > 0:
        leak_percent = (len(intersection) / len(test_ids)) * 100
        print(f"Leakage Percentage:        {leak_percent:.2f}%")
        
    if len(intersection) > 0:
        print("\nLeaked IDs found:")
        print(list(intersection)[:10], "..." if len(intersection) > 10 else "")
        print("\nWARNING: Identity leakage detected! The model might be recognizing faces instead of fake artifacts.")
    else:
        print("\nSUCCESS: No identity leakage detected. The model is learning generalized deepfake features.")
    print("="*50)

if __name__ == "__main__":
    audit()
