from PIL import Image
import pytesseract
import re
import cv2

pytesseract.pytesseract.tesseract_cmd = r'C:/Program Files/Tesseract-OCR/tesseract.exe'


img = cv2.imread('./001_project/data/test_cap.bmp')
img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
print(pytesseract.image_to_string(img_rgb))

# text = pytesseract.image_to_string(img)
# print(text) # 2020-10-06 11:22:28

