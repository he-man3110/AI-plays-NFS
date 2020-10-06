import os
import time
import cv2
import numpy as np
from controls.capture_screen import getProcessedImg 
from controls.simulate_keyboard import PressKey,ReleaseKey
from controls.getkeys import key_check


file_name = 'Training_data_straight.npy'
SAMPLES = 10000

if os.path.isfile(file_name):
    print("File exists, loading previous data")
else:
    print("File does not exist, starting fresh")
training_data = []

def keys_to_output(keys):
    # output = {UP, LEFT, RIGHT, DOWN}
    #UP = '\x00' followed by 'H'
    #DOWN = '\x00' followed by 'P'
    #RIGHT = '\x00' followed by 'M'
    #LEFT = '\x00' followed by 'K'
    output = [0,0,0,0]
    if 'W' in keys:
        output[0] = 1
    elif 'A' in keys:
        output[1] = 1
    elif 'D' in keys:
        output[2] = 1
    elif 'S' in keys:
        output[3] = 1
    return output





def main():
    last_time = time.time()
    cv2.namedWindow("Computer vision", cv2.WINDOW_AUTOSIZE)
    while True:
        #cur_time = time.time()    
        screencap = getProcessedImg()
        output = keys_to_output(key_check())
        training_data.append([screencap, output])
        #cv2.imshow("Computer vision", screencap)
        
        #print(f"Loop took {last_time}")
        #last_time = cur_time - last_time
        
        if len(training_data) % SAMPLES == 0:
            print(len(training_data))
            np.save(file_name, training_data)
            cur_time = time.time()
            print(f"entire opreation took {cur_time - last_time}s to finish")
            cv2.destroyAllWindows()
            break
        cv2.waitKey(5)

        #if cv2.waitKey(5) ==  ord('q'):
        #    cv2.destroyAllWindows()
        #    break

if __name__ == "__main__":
    for i in range(10,0,-1):
        os.system("cls")
        print(i)
        time.sleep(0.5)
    main()