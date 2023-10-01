import os
import cv2
import sys


for file in sorted(os.listdir("textures")):
    new_path = os.path.join("new_textures", file)
    old_path = os.path.join("textures", file)
    img = cv2.imread(old_path)
    print(new_path.replace(".jpg", ".png"))
    cv2.imwrite(new_path.replace(".jpg", ".png"), img)
