import os
from pathlib import Path
from PIL import Image

dataset_dir = "dataset"
converted_count = 0
removed_count = 0

print("🔍 Начинаю поиск и конвертацию файлов .jfif...")

for root, dirs, files in os.walk(dataset_dir):
    for file in files:
        file_path = Path(root) / file

        # Удаляем скрытые системные файлы Mac (DS_Store)
        if file.startswith('.'):
            os.remove(file_path)
            removed_count += 1
            continue

        # Находим файлы jfif
        if file_path.suffix.lower() == '.jfif':
            print(f"🔄 Конвертирую: {file_path.name}")
            try:
                # Открываем картинку через Pillow
                img = Image.open(file_path)
                # Переводим в RGB (обязательно для JPEG)
                img = img.convert('RGB')

                # Создаем новое имя с расширением .jpg
                new_path = file_path.with_suffix('.jpg')
                img.save(new_path, "JPEG")

                # Удаляем старый проблемный файл
                os.remove(file_path)
                converted_count += 1
            except Exception as e:
                print(f"❌ Файл поврежден, удаляю {file_path.name}: {e}")
                os.remove(file_path)
                removed_count += 1

print(f"✅ Готово! Конвертировано в JPG: {converted_count}. Удалено мусора: {removed_count}.")