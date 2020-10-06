from PIL import ImageGrab
import cv2
import numpy as np 


#bbox dimension
BBOX_W = 1024 
BBOX_H = 768
#[[0,690],[580,340],[700,340],[1280,660]]
#[[10,500],[10,300],[300,200],[500,200],[800,300],[800,500]]
#modified : [[0,690],[80,340],[700,340],[1280,690],[0,690]]
#new [[10,30],[778,30],[778,1054],[10,1054]]
ROI_VERTICES = np.array([[10,30],[1034+100,30],[1034+100,798],[10,798]])
ROI_TRAINING_VERTICES = np.array([[0,450],[400,200],[500,200],[900,450]])
ROI_TRAINING_VERTICES_OPTIMIZED = np.array([[0,450],[400,200],[500,200],[900,450],[900,500],[700,500],[500,250],[400,250],[200,500],[0,500]])



def draw_lines(img, lines):
    try:
        for line in lines:
            coords = line[0]
            cv2.line(img, (coords[0], coords[1]),(coords[2], coords[3]), [255,255,255],1)
    except :
        pass

def getProcessedImg():
    temp = ImageGrab.grab(bbox=(10,30,1280,750))  #was 900, 500
    temp = np.array(temp)
    #temp = temp[10:798,30:1054]
    original_img = cv2.cvtColor(temp, cv2.COLOR_BGR2GRAY)
    
    original_img = cv2.resize(original_img,(300,200))
    
    #original_img = cv2.Canny(original_img,threshold1=300,threshold2=500)
    #original_img = cv2.GaussianBlur(original_img, (5,5),0)
    #return original_img
    #original_img = roi(original_img, [ROI_TRAINING_VERTICES_OPTIMIZED])
    #lines = cv2.HoughLinesP(original_img, 1, np.pi/180, 180, minLineLength=10, maxLineGap=5)
    #draw_lines(original_img, lines)

    return original_img

def roi(original_img, vertices):
    mask = np.zeros_like(original_img)
    cv2.fillPoly(mask,vertices, 255)
    return cv2.bitwise_and(original_img, mask)
