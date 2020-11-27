import numpy as np
import pandas as pd
from collections import Counter
from random import shuffle
import cv2

train_data = np.load('Training_data_straight.npy')
#shuffle(train_data)
fwds = []
lfts = []
rhts = []


def balance():
    for data in train_data:
            img = data[0]
            choice = data[1]
            if choice == [1,0,0,0]:
                choice = [1,0,0]
                fwds.append([img,choice])
            elif choice == [0,1,0,0]:
                choice = [0,1,0]
                lfts.append([img,choice])
            elif choice == [0,0,1,0]:
                choice = [0,0,1]
                rhts.append([img,choice])
            else :
                print("No matches")
                continue
    fwds = fwds[:len(lfts)][:len(rhts)]
    lfts = lfts[:len(fwds)]
    rhts = rhts[:len(rhts)]


    final_data = fwds + lfts + rhts
    shuffle(final_data)
    print(len(final_data))
    np.save("Balanced_training_data.npy", final_data)

    def data_info():
        print("samples : ", len(train_data))
        df = pd.DataFrame(train_data)
        print(df.head())
        print(Counter(df[1].apply(str)))





def visiualize():
    cv2.namedWindow("Recorded", cv2.WINDOW_AUTOSIZE)
    
    for data in train_data:
        img = data[0]
        choice = data[1]
        cv2.imshow("Recorded",img)
        print(img.shape)
        #print(choice)
        if cv2.waitKey(10) == ord('q'):
            cv2.destroyAllWindows()
            break

if __name__ == "__main__":
    visiualize()