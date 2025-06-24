import os
import shutil
import random

# Paths
train_dir = os.path.join('data', 'train')
test_dir = os.path.join('data', 'test')

# Ensure test_dir exists
os.makedirs(test_dir, exist_ok=True)

# For each number folder in train
for number in os.listdir(train_dir):
    number_path = os.path.join(train_dir, number)
    if not os.path.isdir(number_path):
        continue
    # List all image files
    images = [f for f in os.listdir(number_path) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
    if not images:
        continue
    # Shuffle and select 20%
    random.shuffle(images)
    n_move = max(1, int(len(images) * 0.2))
    images_to_move = images[:n_move]
    # Create corresponding test folder
    test_number_path = os.path.join(test_dir, number)
    os.makedirs(test_number_path, exist_ok=True)
    # Move files
    for img in images_to_move:
        src = os.path.join(number_path, img)
        dst = os.path.join(test_number_path, img)
        shutil.move(src, dst)
print('Done moving 20% of images from each number folder to test folders.')
