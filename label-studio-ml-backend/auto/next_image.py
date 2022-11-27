import pyautogui
import time
from tqdm import tqdm

print(pyautogui.KEYBOARD_KEYS)


def wait(t):
    for _ in tqdm(range(t)):
        time.sleep(1)
        
# for i in range(100):
#     wait(1)
#     # pyautogui.hotkey('tab')
#     pyautogui.hotkey("shift", "down")
#     # pyautogui.hotkey("shift", "tab")

# raise



# wait(5)

# # pyautogui.hotkey("shift", "up")
# pyautogui.keyDown("shift")
# wait(1)
# pyautogui.keyDown("down")
# wait(1)
# pyautogui.keyDown("down")
# wait(1)
# pyautogui.press("enter")

# raise

while True:
    wait(10)
    # pyautogui.moveTo(393,1085)
    # pyautogui.click()
    # pyautogui.press("down")
    # pyautogui.press("down")
    
    pyautogui.hotkey("shiftleft", "down")
    # pyautogui.keyDown("shift")
    # pyautogui.keyDown("down")
    wait(1)
    pyautogui.press("enter")
    pyautogui.press("down")
    pyautogui.press("down")
    pyautogui.scroll(-300)

wait(5)

pyautogui.hotkey("shift", "down")
# pyautogui.keyDown("shift")
# pyautogui.keyDown("down")
wait(1)
pyautogui.press("enter")