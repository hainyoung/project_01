import pyautogui
from PIL import ImageGrab

while True:
    screen = ImageGrab.grab()
    print(screen.getpixel(pyautogui.position()))