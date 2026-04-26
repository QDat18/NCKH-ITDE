import os
import random
from collections import defaultdict

class UnionFind:
    def __init__(self):
        self.parent = {}
    
    def find(self, i):
        if i not in self.parent:
            self.parent[i] = i
        if self.parent[i] == i:
            return i
        self.parent[i] = self.find(self.parent[i])
        return self.parent[i]
    
    def union(self, i, j):
        root_i = self.find(i)
        root_j = self.find(j)
        if root_i != root_j:
            self.parent[root_i] = root_j

def extract_all_ids(filename):
    found = []
    if "FF_" in filename:
        parts = filename.split("_")
        for p in parts:
            if p.isdigit() and len(p) >= 2: # Ignore small numbers like f0, f1
                found.append(f"FF_{p}")
    elif "Celeb-" in filename:
        parts = filename.split("_")
        for p in parts:
            if p.startswith("id") and p[2:].isdigit():
                found.append(f"Celeb_{p}")
    return list(set(found))

def repartition_v2():
    base_dir = "d:/KyV_HocVienNganHang/NCKH/Final/Demo"
    data_dir = os.path.join(base_dir, "data_image_train")
    
    uf = UnionFind()
    file_to_ids = {}
    all_samples = []
    
    # 1. First pass: Identify all ID connections
    for label_dir, label in [("Real", 0), ("Fake", 1)]:
        full_path = os.path.join(data_dir, label_dir)
        if not os.path.exists(full_path): continue
        
        print(f"Pass 1: Analyzing {label_dir}...")
        for filename in os.listdir(full_path):
            if filename.lower().endswith(('.jpg', '.jpeg', '.png')):
                ids = extract_all_ids(filename)
                if ids:
                    img_path = os.path.join(full_path, filename)
                    all_samples.append((img_path, label, ids))
                    # Link all IDs in this file together
                    for i in range(len(ids) - 1):
                        uf.union(ids[i], ids[i+1])
                    # If it's just one ID, ensure it's in UF
                    if len(ids) == 1:
                        uf.find(ids[0])

    # 2. Second pass: Group samples by their component root ID
    root_to_samples = defaultdict(list)
    for path, label, ids in all_samples:
        # Every file belongs to the group of its first ID's root
        root = uf.find(ids[0])
        root_to_samples[root].append((path, label))
    
    all_roots = list(root_to_samples.keys())
    random.seed(42)
    random.shuffle(all_roots)
    
    print(f"Total ID Clusters found: {len(all_roots)}")
    
    # 3. Split the Clusters (80/10/10)
    train_idx = int(0.8 * len(all_roots))
    val_idx = int(0.9 * len(all_roots))
    
    train_roots = all_roots[:train_idx]
    val_roots = all_roots[train_idx:val_idx]
    test_roots = all_roots[val_idx:]
    
    splits = {
        "splits/train.txt": train_roots,
        "splits/val.txt": val_roots,
        "splits/test.txt": test_roots
    }
    
    # 4. Write output
    for file_path, roots in splits.items():
        count = 0
        with open(file_path, "w", encoding="utf-8") as f:
            for r in roots:
                for path, label in root_to_samples[r]:
                    f.write(f"{path} {label}\n")
                    count += 1
        print(f"Wrote {count} samples to {file_path}")

    print("\n" + "="*50)
    print("V2 RE-PARTITIONING (CLUSTER-BASED) COMPLETE")
    print("="*50)
    print("This ensures that if a fake video uses Person A and Person B,")
    print("both individuals are quarantined in the same split.")
    print("="*50)

if __name__ == "__main__":
    repartition_v2()
