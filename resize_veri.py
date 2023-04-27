import cv2
import glob
import os

os.mkdir('data/VeRi/train_resize')
os.mkdir('data/VeRi/test_resize')
os.mkdir('data/VeRi/query_resize')

for img in glob.glob("data/VeRi/image_train/*.jpg"):
    image = cv2.imread(img)
    h = image
    imgResized = cv2.resize(image, (64, 128))
    cv2.imwrite(os.path.join('data/VeRi/train_resize', img.split(os.sep)[-1]), imgResized)

for img in glob.glob("data/VeRi/image_test/*.jpg"):
    image = cv2.imread(img)
    h = image
    imgResized = cv2.resize(image, (64, 128))
    cv2.imwrite(os.path.join('data/VeRi/test_resize', img.split(os.sep)[-1]), imgResized)

for img in glob.glob("data/VeRi/image_query/*.jpg"):
    image = cv2.imread(img)
    h = image
    imgResized = cv2.resize(image, (64, 128))
    cv2.imwrite(os.path.join('data/VeRi/query_resize', img.split(os.sep)[-1]), imgResized)