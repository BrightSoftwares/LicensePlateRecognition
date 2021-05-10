from darkflow.net.build import TFNet
import tensorflow as tf
from tensorflow.keras import layers, models
import numpy as np
import cv2
import imutils
import requests
import pytesseract
pytesseract.pytesseract.tesseract_cmd = r'/usr/local/bin/tesseract'


options = {"pbLoad": "yolo-plate.pb", "metaLoad": "yolo-plate.meta", "gpu": 0.9}
yoloPlate = TFNet(options)

options = {"pbLoad": "yolo-character.pb", "metaLoad": "yolo-character.meta", "gpu":0.9}
yoloCharacter = TFNet(options)

post_to_api = 1
api_url = "http://localhost:8009/plates/"


def firstCrop(img, predictions):
    predictions.sort(key=lambda x: x.get('confidence'))
    xtop = predictions[-1].get('topleft').get('x')
    ytop = predictions[-1].get('topleft').get('y')
    xbottom = predictions[-1].get('bottomright').get('x')
    ybottom = predictions[-1].get('bottomright').get('y')
    firstCrop = img[ytop:ybottom, xtop:xbottom]
    cv2.rectangle(img,(xtop,ytop),(xbottom,ybottom),(0,255,0),3)
    return firstCrop
    
def secondCrop(img):
    gray=cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
    ret,thresh = cv2.threshold(gray,127,255,0)
    contours,_ = cv2.findContours(thresh,cv2.RETR_LIST,cv2.CHAIN_APPROX_SIMPLE)
    areas = [cv2.contourArea(c) for c in contours]
    if(len(areas)!=0):
        max_index = np.argmax(areas)
        cnt=contours[max_index]
        x,y,w,h = cv2.boundingRect(cnt)
        cv2.rectangle(img,(x,y),(x+w,y+h),(0,255,0),2)
        secondCrop = img[y:y+h,x:x+w]
    else: 
        secondCrop = img
    return secondCrop

def auto_canny(image, sigma=0.33):
    # compute the median of the single channel pixel intensities
    v = np.median(image)
 
    # apply automatic Canny edge detection using the computed median
    lower = int(max(0, (1.0 - sigma) * v))
    upper = int(min(255, (1.0 + sigma) * v))
    edged = cv2.Canny(image, lower, upper)
 
    # return the edged image
    return edged

def opencvReadPlate(img):
    charList=[]
    gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
    thresh_inv = cv2.adaptiveThreshold(gray,255,cv2.ADAPTIVE_THRESH_MEAN_C,cv2.THRESH_BINARY_INV,39,1)
    edges = auto_canny(thresh_inv)
    ctrs, _ = cv2.findContours(edges.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    sorted_ctrs = sorted(ctrs, key=lambda ctr: cv2.boundingRect(ctr)[0])
    img_area = img.shape[0]*img.shape[1]

    for i, ctr in enumerate(sorted_ctrs):
        x, y, w, h = cv2.boundingRect(ctr)
        roi_area = w*h
        non_max_sup = roi_area/img_area

        if((non_max_sup >= 0.015) and (non_max_sup < 0.09)):
            if ((h>1.2*w) and (3*w>=h)):
                char = img[y:y+h,x:x+w]
                charList.append(cnnCharRecognition(char))
                cv2.rectangle(img,(x,y),( x + w, y + h ),(90,0,255),2)
    cv2.imshow('OpenCV character segmentation',img)
    licensePlate="".join(charList)
    return licensePlate

def cnnCharRecognition(img):
    dictionary = {0:'0', 1:'1', 2 :'2', 3:'3', 4:'4', 5:'5', 6:'6', 7:'7', 8:'8', 9:'9', 10:'A',
    11:'B', 12:'C', 13:'D', 14:'E', 15:'F', 16:'G', 17:'H', 18:'I', 19:'J', 20:'K',
    21:'L', 22:'M', 23:'N', 24:'P', 25:'Q', 26:'R', 27:'S', 28:'T', 29:'U',
    30:'V', 31:'W', 32:'X', 33:'Y', 34:'Z'}

    blackAndWhiteChar=cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    blackAndWhiteChar = cv2.resize(blackAndWhiteChar,(75,100))
    image = blackAndWhiteChar.reshape((1, 100,75, 1))
    image = image / 255.0
    characterRecognition = tf.keras.models.load_model('character_recognition.h5')
    new_predictions = characterRecognition.predict(image)
    char = np.argmax(new_predictions)
    return dictionary[char]

def yoloCharDetection(predictions,img):
    charList = []
    positions = []
    for i in predictions:
        if i.get("confidence")>0.10:
            xtop = i.get('topleft').get('x')
            positions.append(xtop)
            ytop = i.get('topleft').get('y')
            xbottom = i.get('bottomright').get('x')
            ybottom = i.get('bottomright').get('y')
            char = img[ytop:ybottom, xtop:xbottom]
            cv2.rectangle(img,(xtop,ytop),( xbottom, ybottom ),(255,0,0),2)
            charList.append(cnnCharRecognition(char))

    cv2.imshow('Yolo character segmentation',img)
    sortedList = [x for _,x in sorted(zip(positions,charList))]
    licensePlate="".join(sortedList)
    return licensePlate


def post_plate_to_api(plate, accuracy, owner, detector=None):
    payload = {'license_number': plate, 'owner': owner, 'detector': detector, 'accuracy': accuracy}

    r = requests.post(api_url, data=payload)

    print("License plate posted. Response is {}".format(r.text))


def yolo_extract_plate_images(frame):
    # Detect the plates
    predictions = yoloPlate.return_predict(frame)
    print("Predictions OpenCV", predictions)
    
    firstCropImg = firstCrop(frame, predictions)
    cv2.imshow('First crop plate',firstCropImg)

    return firstCropImg

def cv_extract_plate_images(img):
    gray=cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
    ret,thresh = cv2.threshold(gray,127,255,0)
    contours,_ = cv2.findContours(thresh,cv2.RETR_LIST,cv2.CHAIN_APPROX_SIMPLE)
    areas = [cv2.contourArea(c) for c in contours]
    if(len(areas)!=0):
        max_index = np.argmax(areas)
        cnt=contours[max_index]
        x,y,w,h = cv2.boundingRect(cnt)
        cv2.rectangle(img,(x,y),(x+w,y+h),(0,255,0),2)
        secondCrop = img[y:y+h,x:x+w]
    # else: 
    #     secondCrop = img
    return secondCrop

def tesseract_readplate(img):
    text = pytesseract.image_to_string(img, config='--oem 3 --psm 6 -c tessedit_char_whitelist=ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789')
    print("programming_fever's License Plate Recognition\n")
    print("Detected license plate Number is:",text)
    # Cropped = cv2.resize(Cropped,(400,200))
    # cv2.imshow('Cropped',Cropped)

cap = cv2.VideoCapture('20210502_161520.mp4')
counter=0

while(cap.isOpened()):
    ret, frame = cap.read()
    h, w, l = frame.shape
    # frame = imutils.rotate(frame, 270)


    if counter%6 == 0:
        licensePlate = []
        try:
            # yoloCropImg = yolo_extract_plate_images(frame)
            # cv2.imshow('yoloCropImg',yoloCropImg)

            # cvCropImg = cv_extract_plate_images(frame)
            # cv2.imshow('cvCropImg',cvCropImg)
            
            firstCropImg = yolo_extract_plate_images(frame)
            # firstCropImg = cv_extract_plate_images(frame)
            print("First crop done")
            cv2.imshow('First crop plate',firstCropImg)

            secondCropImg = secondCrop(firstCropImg)
            print("Second crop done")
            cv2.imshow('Second crop plate',secondCropImg)

            # print("Open CV Read plate")
            # cv_detected_plate = opencvReadPlate(secondCropImg)
            # licensePlate.append(cv_detected_plate)
            # # plate = licensePlate[0]
            # print("OpenCV+CNN : " + cv_detected_plate)
            # accuracy = 10

            # Reading the plate with tesseract
            tesseract_readplate(secondCropImg)

            # Post plate to api
            # post_plate_to_api(cv_detected_plate, accuracy, 2, "OpenCV+CNN")

            # print("Yolo Read plate")
            # predictions = yoloCharacter.return_predict(secondCropImg)
            # print("Predictions Yolo: ", predictions)

            # print("Detect plate with yolo on the second detection")
            # secondCropImgCopy = secondCropImg.copy()
            # yolo_detected_plate = yoloCharDetection(predictions,secondCropImgCopy)
            # licensePlate.append(yolo_detected_plate)
            # # plate = licensePlate[1]
            # print("Yolo+CNN plate : " + yolo_detected_plate)
            # accuracy = 10

            # Post plate to api
            # post_plate_to_api(yolo_detected_plate, accuracy, 2, "Yolo+CNN")

        except Exception as e:
            print("error", str(e))

    counter+=1
    cv2.imshow('Video',frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()

