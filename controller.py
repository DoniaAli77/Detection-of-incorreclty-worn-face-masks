import alg
import  cv2
from matplotlib import pyplot as plt
def display(img, frameName="Mask"):
    h, w = img.shape[0:2]
    neww = 1200
    newh = int(neww*(h/w))
    img = cv2.resize(img, (neww, newh))
    cv2.imshow(frameName, img)
# def onlinecamera():
#     import requests
#     import cv2
#     import numpy as np
#     import imutils
# # Replace the below URL with your own. Make sure to add "/shot.jpg" at last.
#     url = "http://192.168.1.2:8080/shot.jpg"
#     i=0
#     Casee=""


# # While loop to continuously fetching data from the Url
#     while True:
#         img_resp = requests.get(url)
#         print(img_resp)
#         img_arr = np.array(bytearray(img_resp.content), dtype=np.uint8)
#         img = cv2.imdecode(img_arr, -1)
#         img = imutils.resize(img, width=1000, height=1800)
#         if i % 5 == 0:
#            faces =alg.detectionML(img)
#            if(isinstance(faces, str)):
#               if(faces=="no face detected"):
#                   print("no face detected") 
#            else:       
#               images=alg.alg(faces,img)
#               Casee=images[0]
#               color=images[1]
#               x, y, w, h = images[2]
#         #    pre1.append(images[1])
#         i+=1
#         if(Casee != ""):
#             cv2.rectangle(img,(int(x),int(y)),(int(x+w),int(y+h)),color,2) 
#             cv2.putText(img, Casee, (x,y-5),cv2.FONT_HERSHEY_SIMPLEX, 1.3, color, 3)
#         print("here",i)
#         cv2.imshow("Android_cam", img)

#     # Press Esc key to exit
#         if cv2.waitKey(1) == 27:
#             break

#     cv2.destroyAllWindows()   
# # if __name__ == '__main__':     
def video():
    video_capture =cv2.VideoCapture(0)
    print(video_capture)
    pre1=[]
    i=0
    Casee=""

    while (video_capture.isOpened()):
        
        ret, image = video_capture.read()
        if not ret:
           break
        # plt.imshow(image)
        # plt.show()
        if i%5 == 0:
           faces =alg.detectionML(image)
           if(isinstance(faces, str)):
              if(faces=="no face detected"):
                  print("no face detected") 
           else: 
                     
              images=alg.alg(faces,image)
              Casee=images[0]
              color=images[1]
              x, y, w, h = images[2]
        #    pre1.append(images[1])
        i+=1
        if(Casee != ""):
            cv2.rectangle(image,(int(x),int(y)),(int(x+w),int(y+h)),color,2) 
            cv2.putText(image, Casee, (x,y-5),cv2.FONT_HERSHEY_SIMPLEX, 1.3, color, 3)
        print("here",i)
        display(image)
        # cv2.namedWindow('image',WINDOW_NORMAL)
        # cv2.imshow("Faces found", image)
        if cv2.waitKey(1) & 0xFF == ord('q'):
           break
    video_capture.release()  
    cv2.destroyAllWindows()  

video()    