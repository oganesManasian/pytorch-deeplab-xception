import shutil
import os

DATA_PATH = "data/cityscapes/leftImg8bit/test_night2day_raw"
TARGET_PATH = "data/cityscapes/leftImg8bit/test_night2day/night2day"

filenames = os.listdir(DATA_PATH)

for filename in filenames:
    if "_fake_B" in filename:
        new_filename = filename.replace("_fake_B", "")
        shutil.copy(os.path.join(DATA_PATH, filename),
                    os.path.join(TARGET_PATH, new_filename))
