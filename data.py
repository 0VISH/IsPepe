#This script convert .webp files from synthetic-pepe/ to .png and places them under train/, test/, and val/ under data/

import os
from PIL import Image

#        train test val
counts = [304, 100, 100]
paths = ["data/train/pepe/", "data/test/pepe/", "data/val/pepe/"]

targetSize = (224, 224)
count = 0
saveCount = 0
off = 0
savePath = paths[0]
if not os.path.isdir(savePath): os.makedirs(savePath)

for path in os.listdir("synthetic-pepe/"):
    if path.endswith((".webp", ".jpg", ".jpeg")):
        img = Image.open("synthetic-pepe/"+path)
        img = img.resize(targetSize, Image.Resampling.LANCZOS)
        pngImage = img.save(savePath+f"pepe{count}.png", "PNG")
        count += 1
        if count >= counts[off]:
            saveCount += count
            count = 0
            off += 1
            savePath = paths[off]
            if not os.path.isdir(savePath): os.makedirs(savePath)

print(f"Processed {saveCount+count} pepe imagess")

#Place 504 random images under notPepe/ More the number of frogs(diff colour, angle, etc..), the better

count = 0
saveCount = 0
off = 0
paths = ["data/train/notPepe/", "data/test/notPepe/", "data/val/notPepe/"]
savePath = paths[0]
if not os.path.isdir(savePath): os.makedirs(savePath)

for path in os.listdir("notPepe/"):
    if path.endswith((".webp", ".jpg", ".jpeg", ".png")):
        img = Image.open("notPepe/"+path)
        img = img.resize(targetSize, Image.Resampling.LANCZOS)
        pngImage = img.save(savePath+f"notPepe{count}.png", "PNG")
        count += 1
        if count >= counts[off]:
            saveCount += count
            count = 0
            off += 1
            savePath = paths[off]
            if not os.path.isdir(savePath): os.makedirs(savePath)

print(f"Processed {saveCount+count} non pepe images")