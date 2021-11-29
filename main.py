import cv2 #uvoz biblioteke OpenCV-a
import time #uvoz biblioteke potreban za prikaz vremena
COLORS=[(0,255,255),(255,255,0),(255,255,0),(0,255,0),(255,0,0)] #definicija boja
class_names=[] #deklaracija liste
with open("coco.names","r") as f: #otvaranje definisanih imena modela
    class_names=[cname.strip() for cname in f.readlines()] #citanje imena modela
cap=cv2.VideoCapture("http://192.168.1.203:4747/video") #ulazni video, staviti mp4 fajl u slučaju detekcije objekata na već postojćem video klipu
net=cv2.dnn.readNet("yolov4-tiny.weights","yolov4-tiny.cfg") #učitavanje modela i konfiguracionih fajlova
model=cv2.dnn_DetectionModel(net) #definisanje modela
model.setInputParams(size=(416,416),scale=1/255) #definicija ulaznih parametara modela
while True:
    _,frame=cap.read() #citanje frejmova
    start=time.time()
    classes, scores, boxes=model.detect(frame,0.2,0.2) #detekcija
    end=time.time()
    for(classid,score,box) in zip(classes,scores,boxes): #kreiranje okvira oko detektovanog objetka
        color=COLORS[int(classid)%len(COLORS)] #definicija boja
        label=f"{class_names[classid[0]]}:{(score*100).round(2)}%" #tekst okvira
        cv2.rectangle(frame,box,color,2) #crtanje okvira
        cv2.putText(frame,label,(box[0],box[1]-10),cv2.FONT_HERSHEY_SIMPLEX,0.5,color,2) #stavljanje teksta iznad okvira
    fps_label=f"FPS: {round((1.0/(end-start)),2)}" #definicija teksta za merenje frejmova u sekundi
    cv2.putText(frame, fps_label, (5, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0),3)  # stavljanje teksta broja frejmova u sekundi preko videa /crni tekst
    cv2.putText(frame,fps_label,(5,30),cv2.FONT_HERSHEY_SIMPLEX,0.6,(0,255,255),2) #stavljanje teksta broja frejmova u sekundi preko videa /zuti tekst
    cv2.imshow("Detektovanje objekata YOLO",frame) #prikaz slike
    if cv2.waitKey(1)==27:
        break
cap.release()
cv2.destroyAllWindows()

