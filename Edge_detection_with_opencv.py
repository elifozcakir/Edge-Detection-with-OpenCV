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


    #sobel
    laplacian = cv2.Laplacian(gray.copy(),cv2.CV_8U,13)
    #kernel = np.ones((3,3),np.uint8)
    #laplacian = cv2.dilate(laplacian,kernel,iterations = 1)

    im_th1 = cv2.adaptiveThreshold(gray.copy(), 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, \
                                   cv2.THRESH_BINARY, 5, 2)
    
    # Display the resulting frame
    #edges = cv2.Canny(gray.copy(),threshold1=50, threshold2=150,apertureSize = 3)
    tempImg = cv2.medianBlur(gray, 5)
    im_th1 = cv2.adaptiveThreshold(tempImg, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, \
                                cv2.THRESH_BINARY, 5, 2)

    blur = cv2.GaussianBlur(im_th1, (5, 5), 0)
    _, im_th2 = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    
    
    #numpy_horizontal = np.hstack((gray, edges, laplacian, im_th2))

    #cv2.imshow('Numpy Horizontal', numpy_horizontal)
    cv2.imshow('Gray',gray)
    cv2.imshow('Edges',edges)
    cv2.imshow('Laplacian',laplacian)
    cv2.imshow('Adaptive',im_th2)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# When everything done, release the capture
cap.release()
cv2.destroyAllWindows()
