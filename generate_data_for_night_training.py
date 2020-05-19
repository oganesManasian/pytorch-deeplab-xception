import shutil
import os
from tqdm import tqdm

IMAGE_PATH = "data/cityscapes/leftImg8bit/train"
LABELS_PATH = "data/cityscapes/gtFine/train"

IMAGE_PATH_TARGET = "data/cityscapes/leftImg8bit/train_combined"
LABELS_PATH_TARGET = "data/cityscapes/gtFine/train_combined"

TRANSLATED_IMAGE_PATH = "translated"
TRANSLATED_IMAGE_SUFFIX = "_fake_B"

IMAGE_SUFFIX = "_leftImg8bit.png"
LABEL_SUFFIX = "_gtFine_labelIds.png"

if os.path.exists(IMAGE_PATH_TARGET):
    shutil.rmtree(IMAGE_PATH_TARGET)

if os.path.exists(LABELS_PATH_TARGET):
    shutil.rmtree(LABELS_PATH_TARGET)

os.mkdir(IMAGE_PATH_TARGET)
os.mkdir(LABELS_PATH_TARGET)

print("Copying train day data")
cities = os.listdir(IMAGE_PATH)
print(cities)

# for city in cities:
#     os.mkdir(f"{IMAGE_PATH_TARGET}/{city}")
#     os.mkdir(f"{LABELS_PATH_TARGET}/{city}")
#
#
# for city in tqdm(cities):
#     images = os.listdir(f"{IMAGE_PATH}/{city}")
#     labels = os.listdir(f"{LABELS_PATH}/{city}")
#
#     for file in images:
#         shutil.copy(f"{IMAGE_PATH}/{city}/{file}", f"{IMAGE_PATH_TARGET}/{city}/{file}")
#
#     for file in labels:
#         shutil.copy(f"{LABELS_PATH}/{city}/{file}", f"{LABELS_PATH_TARGET}/{city}/{file}")

print("Building train night data")
translated_cities = cities[:6]
print(translated_cities)

for city in translated_cities:
    os.mkdir(f"{IMAGE_PATH_TARGET}/{city}_night")
    os.mkdir(f"{LABELS_PATH_TARGET}/{city}_night")

# Copy night images and corresponding labels
images_to_sort = os.listdir(TRANSLATED_IMAGE_PATH)
for file in tqdm(images_to_sort):
    for city in translated_cities:
        if file.startswith(city):
            new_image_name = file.replace(TRANSLATED_IMAGE_SUFFIX, "")
            shutil.copy(f"{TRANSLATED_IMAGE_PATH}/{file}", f"{IMAGE_PATH_TARGET}/{city}_night/{new_image_name}")
            label_original_name = file.replace(TRANSLATED_IMAGE_SUFFIX, "").replace(IMAGE_SUFFIX, LABEL_SUFFIX)
            # label_new_name = label_original_name.split(".")[0] + TRANSLATED_IMAGE_SUFFIX + ".png"
            shutil.copy(f"{LABELS_PATH}/{city}/{label_original_name}", f"{LABELS_PATH_TARGET}/{city}_night/{label_original_name}")
            break
