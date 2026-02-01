import os

images_dir = "images"       # folder containing images
labels_dir = "labels"       # folder containing YOLO txt files

image_exts = (".jpg", ".jpeg", ".png")

count_deleted = 0
count_kept = 0

for img in os.listdir(images_dir):
    if not img.lower().endswith(image_exts):
        continue

    base = os.path.splitext(img)[0]
    label_path = os.path.join(labels_dir, base + ".txt")
    image_path = os.path.join(images_dir, img)

    if os.path.exists(label_path):
        os.remove(image_path)
        count_deleted += 1
    else:
        count_kept += 1

print("Images deleted:", count_deleted)
print("Images kept:", count_kept)
