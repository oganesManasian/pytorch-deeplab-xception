# from skimage.io import imread
import os
from PIL import Image
from tqdm import tqdm

PATH = "data/cityscapes/leftImg8bit/train_combined/"

folders = os.listdir(PATH)

beaten_files = []
no_channels = []

for folder in folders:
    files = os.listdir(os.path.join(PATH, folder))
    print(folder, len(files))
    # for file in tqdm(files):
    #     path = os.path.join(PATH, folder, file)
    #     print(path)
    #     image = Image.open(path).convert('RGB')
    #     print(image)
    #     # try:
    #     #     path = os.path.join(PATH, folder, file)
    #     #     image = Image.open(path).convert('RGB')
    #     #     print(image.size())
    #     #     if image.size()[1] < 3:
    #     #         no_channels.append(path)
    #     #     # image = cv2.imread(os.path.join(PATH, file), 1)
    #     #     # image = imread(os.path.join(PATH, file))
    #     # except:
    #     #     beaten_files.append(path)

print("beaten_files", len(beaten_files))
if len(beaten_files) > 0:
    print(beaten_files[0])

print("no_channels", len(no_channels))
if len(no_channels) > 0:
    print(no_channels[0])
