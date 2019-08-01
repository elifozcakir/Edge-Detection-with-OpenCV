import numpy as np
import cv2


cap = cv2.VideoCapture(0)

while(1): #True
    # Capture frame by frame
    ret, frame = cap.read()
    if frame is None:
        continue

    _, frame = cap.read()
    # Our operations on the frame come here
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    edges = cv2.Canny(gray.copy(),threshold1=50, threshold2=150,apertureSize = 3)
    #Laplace
    laplacian = cv2.Laplacian(gray.copy(),cv2.CV_8U,13)
   
    #Adaptive Thresholding
    grayim = cv2.medianBlur(gray,5)
    th2 = cv2.adaptiveThreshold(grayim,255,cv2.ADAPTIVE_THRESH_MEAN_C,\
                cv2.THRESH_BINARY,11,2)
    th3 = cv2.adaptiveThreshold(grayim,255,cv2.ADAPTIVE_THRESH_GAUSSIAN_C,\
                cv2.THRESH_BINARY,11,2)

    #Showing Images
    
    cv2.imshow('Gray',gray)
    cv2.imshow('Edges',edges)
    cv2.imshow('Laplacian',laplacian)
    cv2.imshow('Adaptive Mean Thresholding',th2)
    cv2.imshow('Adaptive Guassian Thresholding',th3)
    
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# When everything done, release the capture
cap.release()
cv2.destroyAllWindows()
