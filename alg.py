from mtcnn import MTCNN
import numpy as np
import pandas as pd
import cv2 
from pathlib import Path
import glob
import os
from matplotlib import pyplot as plt
import skimage
from skimage import feature
from scipy.spatial import distance
import sys
import dlib
from os.path import dirname, join
# from imutils import face_utils
# import imutils
import pickle

mouth_cascade=cv2.CascadeClassifier(cv2.data.haarcascades+'haarcascade_smile.xml')
nose_cascade=cv2.CascadeClassifier(cv2.data.haarcascades+'haarcascade_mcs_nose.xml')


MaskVsNoMask_Filename = "C:/Users/Eng.Donia/Desktop/blabla/Untitled Folder/MaskVsNoMask65kBalanced.pkl" 
with open(MaskVsNoMask_Filename, 'rb') as file:  
    MaskVsNoMask_model = pickle.load(file)
print(MaskVsNoMask_model) 

CorrectVsIncorrect_Filename = "C:/Users/Eng.Donia/Desktop/blabla/Untitled Folder/CorrectVsIncorrect-BalancedFaces63Kpart2.pkl" 
with open(CorrectVsIncorrect_Filename, 'rb') as file:  
    CorrectVsIncorrect_model = pickle.load(file)  
print(CorrectVsIncorrect_model)

NoMaskVsIncorrectVsCorrectFaces_Filename = "C:/Users/Eng.Donia/Desktop/blabla/Untitled Folder/NoMaskVsIncorrectVsCorrectFacesBalanced63ktotalfinal.pkl" 
with open(NoMaskVsIncorrectVsCorrectFaces_Filename, 'rb') as file:  
    NoMaskVsIncorrectVsCorrectFaces_model = pickle.load(file)
print(NoMaskVsIncorrectVsCorrectFaces_model)

CorrectNoseVsIncorrectNose_Filename = "C:/Users/Eng.Donia/Desktop/blabla/Untitled Folder/CorrectNoseVsIncorrectNose-BalancedFaces5k.pkl" 
with open(CorrectNoseVsIncorrectNose_Filename, 'rb') as file:  
    CorrectNoseVsIncorrectNose_model = pickle.load(file)
print(CorrectNoseVsIncorrectNose_model)

detector = MTCNN()
def detectionML(imgg):
    try:       
#         imgg = cv2.imread(os.path.abspath(str(imgg)))
#         print(imgg)
        gray=cv2.cvtColor(imgg, cv2.COLOR_BGR2GRAY)
        img = cv2.cvtColor(imgg, cv2.COLOR_BGR2RGB) 
#         plt.imshow(img)
#         plt.show()
        detections = detector.detect_faces(img)
       
        roi_gray_arrays=[]

        if(len(detections)==0):
            return "no face detected"
        for detection in detections:
            roi_gray_array=[]
            score = detection["confidence"]
            if score > 0.90 or score==0.90:
                    x, y, w, h = detection["box"]
                    temp=img[int(y):int(y+h), int(x):int(x+w)]
                    temp2=imgg[int(y):int(y+h), int(x):int(x+w)]
                    if(len(temp)!=0 ):
                        if(len(temp[0])!=0):
                                detected_face=temp
                                roi_color=detected_face
                                roi_bgr=temp2
                                roi_gray =gray[int(y):int(y+h), int(x):int(x+w)]
                                roi_gray_mouth = gray[int(int(y)+(int(h/2))):int(y+h), int(x):int(x+w)]
# #                         print(int(int(y)+(int(h/2))),int(y+h), int(x),int(x+w))

                                # cv2.rectangle(imgg,(int(x),int(y)),(int(x+w),int(y+h)),(0,255,0),2) 
#                                 plt.imshow(cv2.cvtColor(imgg, cv2.COLOR_BGR2RGB) ,cmap='gray')
# #                               plt.show()
#                                 print(roi_bgr)
                                roi_gray_array.append(roi_bgr)
#                                 print(roi_bgr.shape)
#                                 print(roi_color.shape)
                                roi_gray_array.append(detection)
                                roi_gray_arrays.append(roi_gray_array)

                    
        if(len(roi_gray_arrays)>0):
            return roi_gray_arrays
        else:
            return "no face detected"
    except Exception as e:
                    return str(e)

def newsize128(img):
    try:
#         img= cv2.imread( os.path.abspath(str(img)))
        newimg=cv2.resize(img, (128,128),interpolation = cv2.INTER_AREA) 
        return newimg
    except Exception as e:
        return str(e)
def FindPoint(x1, y1, x2,
              y2, x, y) :
    if (x > x1 and x < x2 and
        y < y1 and y > y2) :
        return True
    else :
        return False
def mask (img):
#     images=detectionML(img)
#     print(images)
#     if(isinstance(images, str)):
#         if(images=="no face detected"):
#             return "no face detected" 
    predictions=[]    
#     for i in range (0,len( images)) :
#         imgg=images[i]
#         imgg = cv2.imread(os.path.abspath(str(img)))
#         plt.imshow(imgg)
#         plt.show
    imgg=newsize128(img)
    img1=np.array([cv2.cvtColor(imgg, cv2.COLOR_BGR2GRAY).reshape(-1)])
    prediction=NoMaskVsIncorrectVsCorrectFaces_model.predict(img1)
#         print("prediction: "+str(prediction[0]))
    predictions.append( prediction[0])

    return predictions
def mask1 (img):
#     images=detectionML(img)
#     print(images)
#     if(isinstance(images, str)):
#         if(images=="no face detected"):
#             return "no face detected" 
        predictions=[]    
#     for i in range (0,len( images)) :
#         imgg=images[i]
#         imgg = cv2.imread(os.path.abspath(str(img)))
#         plt.imshow(imgg)
#         plt.show
#         print(imgg)
        imgg=newsize128(img)
#         plt.imshow(imgg)
#         plt.show
        img1= np.array([cv2.cvtColor(imgg, cv2.COLOR_BGR2GRAY).reshape(-1)])
        prediction=MaskVsNoMask_model.predict(img1)
        if(prediction[0]==1):
                prediction1=CorrectVsIncorrect_model.predict(img1)
#                 print("prediction case mask "+str(prediction1[0]))        
                predictions.append(prediction1[0])
#                 return "prediction case mask "+str(prediction1[0])

#                 if(prediction1[0]==1):
#                     prediction2=CorrectNoseVsIncorrectNose_model.predict(img1)
#                     return "prediction case mask on nose: "+str(prediction2[0])
#                 else:
                    
        else:  
#                 print("prediction: "+str(prediction[0]))
                predictions.append( prediction[0])

        return predictions
def Nose (img):
#     images=detectionML(img)
#     print(images)
    
        predictions=[]    
#         imgg = cv2.imread(os.path.abspath(str(img)))
#         plt.imshow(imgg)
#         plt.show
        imgg=newsize128(img)
        img1= np.array([cv2.cvtColor(imgg, cv2.COLOR_BGR2GRAY).reshape(-1)])
        prediction=CorrectNoseVsIncorrectNose_model.predict(img1)
#         print("prediction: "+str(prediction[0]))
        predictions.append( prediction[0])

        return predictions
# detector = MTCNN()
# predictor = dlib.shape_predictor(pre)
def detectionNoseMouth(imgg,detection):
    try: 
#         imgg = cv2.imread(os.path.abspath(str(imgg)))
        x, y, w, h = detection['box']
        gray=cv2.cvtColor(imgg, cv2.COLOR_BGR2GRAY)
        img = cv2.cvtColor(imgg, cv2.COLOR_BGR2RGB)
        mouthlength=0
        noselength=0
        inose=0
        imouth=0
        roi_color=img
        roi_gray =gray
        roi_gray_mouth = img[len(img)//2:len(img)][:]
#                         roi_color_mouth = img[len(img)//2:len(img)][:]
#                         cv2.rectangle(imgg,(int(x),int(y)),(int(x+w),int(y+h)),(200,55,0),1) 
        mouth = mouth_cascade.detectMultiScale3(roi_gray_mouth,1.3,29,outputRejectLevels=True)
        xml="mouth"
        if(len(mouth[0])==0):
            mouth_cascade2=cv2.CascadeClassifier(cv2.data.haarcascades+'Mouth.xml')
            mouth = mouth_cascade2.detectMultiScale3(roi_gray_mouth,1.3,30,maxSize=(1000,1000),outputRejectLevels=True)
            xml="smile"
        nose = nose_cascade.detectMultiScale3(roi_gray,1.3,2,minSize=(35,35),maxSize=(200,200),outputRejectLevels=True)
        mouthlength=len(mouth[0])
        noselength=len(nose[0])
#                         print(mouth,xml)
#                         print(nose)
        if(noselength>0 ):
            noseconfidence=nose[2][inose]
        if(mouthlength>0 ):
            mouthconfidence=mouth[2][imouth]
        for (ex,ey,ew,eh) in mouth[0]:     
#             cv2.rectangle(imgg,(ex+x,ey+(int(h/2))+y),(ex+x+ew,ey+y+(int(h/2))+eh),(150,0,255),1) 
            if(mouth[2][imouth]>mouthconfidence):
                    mouthconfidence=mouth[2][imouth]
                    mouthlength-=1
            elif(mouthconfidence==max(mouth[2]) and mouth[2][imouth]<mouthconfidence):
                    mouthlength-=1
            imouth+=1
                                
        if(mouthlength>0 ):
            if(mouthconfidence<1):
                    mouthlength-=1
#                         print(imouth)
        i=0
        if(mouthlength>0):
            for confidence in mouth[2]: 
                    if( confidence == mouthconfidence):
                            break
                    else:
                            i+=1
            if((mouth[0][i][1]+(int(h/2))+y+eh)<detection["keypoints"]['nose'][1]) :
                     mouthlength-=1
                                        
                                        
        for (ex,ey,ew,eh) in nose[0]:
            flag=FindPoint(ex+x,ey+y+eh,ex+x+ew,ey+y,detection["keypoints"]['nose'][0],detection["keypoints"]['nose'][1])

            if(flag):
                cv2.rectangle(imgg,(ex+x,ey+y),(ex+x+ew,ey+y+eh),(0,0,255),1)

                if(nose[2][inose]>noseconfidence):
                        noseconfidence=nose[2][inose]
                        noselength-=1
                elif(noseconfidence==max(nose[2]) and nose[2][inose]<noseconfidence):
                        noselength-=1
            else:
                noselength-=1
                if(noselength>0 and inose<len(nose[0])-1):
                    inose+=1
                    if(nose[2][inose]>noseconfidence):
                        noseconfidence=nose[2][inose]
                    inose-=1                                    
            inose+=1           
        mask=-1
        case=""
        if(noselength>0):
            if(mouthlength>0):
                mask=0
                case="no mask"
            else:
                mask=1
                case="mask on mouth or chin"   
        else:
            mask=1
            case="mask"                                   
        r="Case: "+case+" Mask: "+str(mask)
#                         plt.scatter(detection["keypoints"]['mouth_right'][0],detection["keypoints"]['mouth_right'][1])
#                         plt.scatter(detection["keypoints"]['mouth_left'][0],detection["keypoints"]['mouth_left'][1])
#                         plt.scatter(detection["keypoints"]['nose'][0],detection["keypoints"]['nose'][1])
#                         plt.imshow(imgg)
#                         plt.show() 
#                         plt.imshow(roi_gray_mouth)
#                         plt.show() 
#                         print(r)
        return r
                    
    except Exception as e:
                    return str(e)

def alg (faces,imagee):
#    face detection
#     plt.imshow(img)
#     plt.show()
    # faces =detectionML(img)
    # if(isinstance(faces, str)):
    #     if(faces=="no face detected"):
    #         return "no face detected" 
    predictions =[]  
    cases=[]
    for i in range (0,len( faces)) :
        facee=faces[i][0]
        # plt.imshow(cv2.cvtColor(facee,cv2.COLOR_BGR2RGB))
        # plt.show()
#         print(face.shape)
        faceg=cv2.cvtColor(facee, cv2.COLOR_BGR2GRAY)
        face= cv2.equalizeHist(faceg)
        face=cv2.cvtColor(face, cv2.COLOR_GRAY2RGB)
        print(face.shape)
        # plt.imshow(face)
        # plt.show()
        detectionbox=faces[i][1]
        # plt.imshow(cv2.cvtColor(face,cv2.COLOR_BGR2GRAY),cmap='gray')
        # plt.show()
        # imgresized=newsize128(face)
        # plt.imshow(cv2.cvtColor(imgresized,cv2.COLOR_BGR2GRAY),cmap='gray')
        # plt.show()

#   mask
        case1=0
        case0=0
        case_1=0
        predictionN=detectionNoseMouth(face,detectionbox)
        if(predictionN=="Case: mask Mask: 1"):
            predictionN=Nose(face)[0]
            if(predictionN==1):
                case1+=1
            else :
                case_1+=1.5
        elif(predictionN=="Case: mask on mouth or chin Mask: 1"):
                case_1+=1
        else:
               case0+=1
            
        predictionMl=mask1(face)[0]#pipline
        if(predictionMl==0):
            case0+=1
        elif(predictionMl==1):
            case1+=1
        else:
            case_1+=1
        predictionML=mask(face)[0] #kollo
        if(predictionML==0):
            case0+=1
        elif(predictionML==1):
            case1+=1
        else:
            case_1+=1
        case=''
        Casee=""
        print("case1: ",case1)
        print("case0: ",case0)
        print("case_1",case_1)
        if(case0>case1):
            if(case0>case_1):
                case='0'
                color = (0, 0, 255)
                Casee="No Mask"
            else:
                color = (0, 140, 255)
                case='-1'
                Casee="Incorrectly Mask"
        else:
            if(case1>case_1):
                color= (0,255,0)	
                case='1'
                Casee="Correctly Mask"
            else:
                color = (0, 140,255)
                case='-1'
                Casee="Incorrectly Mask"

        prediction=[]
        prediction.append(predictionN)
        prediction.append("prdiction from pipline: "+str(predictionMl))#pipline
        prediction.append("prediction from kolo: "+str(predictionML))#kollo
        predictions.append(prediction)
        print(predictions)
        cases.append(case)
        # color = (0, 0, 255)
        # x, y, w, h = detectionbox['box']
        # cv2.rectangle(imagee,(int(x),int(y)),(int(x+w),int(y+h)),color,2) 
        # cv2.putText(imagee, Casee, (x,y-5),cv2.FONT_HERSHEY_SIMPLEX, 1.3, color, 3)
    return [Casee,color,detectionbox['box']]
# if __name__ == '__main__':
#     video_capture =cv2.VideoCapture(r"D:/bach/try/esraanomask1.mp4")
#     print(video_capture)
#     pre1=[]
#     i=0
#     while (video_capture.isOpened()):
#         ret, image = video_capture.read()
#         if not ret:
#            break
#         images=alg(image)
#         i+=1
#         print("here",i)
#         print(images[0])
#         pre1.append(images[1])
#         cv2.imshow("Faces found", image)
#         if cv2.waitKey(1) & 0xFF == ord('q'):
#            break
#         video_capture.release()  
#         cv2.destroyAllWindows()  