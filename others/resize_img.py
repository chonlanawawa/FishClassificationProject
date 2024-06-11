import os
from PIL import Image

def resize_images_in_directory(root_dir, target_size=(224, 224)):
    for subdir, dirs, files in os.walk(root_dir):
        for file in files:
            if file.endswith(('jpg')):
                file_path = os.path.join(subdir, file)
                try:
                    with Image.open(file_path) as img:
                        img_resized = img.resize(target_size)
                        img_resized.save(file_path)
                        print(f'Resized and saved: {file_path}')
                except Exception as e:
                    print(f'Error processing {file_path}: {e}')

if __name__ == "__main__":
    root_directory = 'dataset/train'
    resize_images_in_directory(root_directory)
