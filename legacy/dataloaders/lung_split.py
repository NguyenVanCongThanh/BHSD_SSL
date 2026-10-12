"""
Archived legacy lung_split.py from original PICK repository.
"""
import os

def create_splits(img_root, out_train='train.txt', out_test='test.txt'):
    img_path = sorted(os.listdir(img_root))
    train_list = [item[:8] for item in img_path[:48]]
    test_list = [item[:8] for item in img_path[48:]]
    with open(out_test, 'w') as f:
        for name in test_list:
            f.write(f"{name}\n")
    with open(out_train, 'w') as f:
        for name in train_list:
            f.write(f"{name}\n")
