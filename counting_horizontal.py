import cv2
import pandas as pd
import numpy as np
from ultralytics import YOLO
from tracker import*
import cvzone

url = ""

model=YOLO('yolov8s.pt')

def RGB(event, x, y, flags, param):
    if(event==cv2.EVENT_MOUSEMOVE):
        point = [x,y]
        print(point)

cv2.namedWindow('RGB')
cv2.setMouseCallback('RGB', RGB)
cap=cv2.VideoCapture('test_video/clip_7.mp4')

# fourcc = cv2.VideoWriter_fourcc(*'avc1')
# output = cv2.VideoWriter('output.avi',fourcc, 5, (640,480))

output = cv2.VideoWriter('output/out_clip7.avi',cv2.VideoWriter_fourcc(*'MPEG'),30,(1020,500))

file = open('coco.names', 'r')
data = file.read()
class_list = data.split('\n')

count=0
personin={}
tracker=Tracker()
counter1=[]

personout={}
counter2=[]
cx1=300
cx2=450
offset=40

while True:
    ret, frame = cap.read()
    if not ret:
        break
    #frame=stream_read()

    count+=1
    if (count%1 != 0):
        continue

    frame=cv2.resize(frame, (1020,500))

    results=model.predict(frame)
    #print(results)

    a=results[0].boxes.data
    px=pd.DataFrame(a).astype("float")
    #print(px)

    list=[]

    for index,row in px.iterrows():
        #print(rows)

        x1=int(row[0])
        y1=int(row[1])
        x2=int(row[2])
        y2=int(row[3])
        d=int(row[5])

        c=class_list[d]
        if 'person' in c:
            list.append([x1,y1,x2,y2])
    
    bbox_id=tracker.update(list)
    for bbox in bbox_id:
        x3,y3,x4,y4,id=bbox
        cx=int(x3+x4)//2
        cy=int(y3+y4)//2
        cv2.circle(frame,(cx,cy),4,(255,0,255),-1)
        
        cv2.rectangle(frame, (x3,y3),(x4,y4),(255,0,0),2)
        cvzone.putTextRect(frame,f'{id}', (x3,y3), 1,2)
        
        ## for down going
        if (cx1<(cx+offset) and (cx1>cx-offset)):
            print("first if")
            cv2.rectangle(frame, (x3,y3),(x4,y4),(0,0,255),2)
            cvzone.putTextRect(frame,f'{id}', (x3,y3), 1,2)
            personin[id]=(cx,cy)

        if (id in personin):
            if (cx2<(cx+offset) and (cx2>cx-offset)):
                print("first counting")
                cv2.rectangle(frame, (x3,y3),(x4,y4),(0,255,255),2)
                cvzone.putTextRect(frame,f'{id}', (x3,y3), 1,2)
                if counter1.count(id)==0:
                    print(len(counter1))
                    counter1.append(id)
        
        ## for up going
        if (cx2<(cx+offset) and (cx2>cx-offset)):
            print("second if")
            cv2.rectangle(frame, (x3,y3),(x4,y4),(0,255,0),2)
            cvzone.putTextRect(frame,f'{id}', (x3,y3), 1,2)
            personout[id]=(cx,cy)

        if (id in personout):
            if (cx1<(cx+offset) and (cx1>cx-offset)):
                print("second counting")
                cv2.rectangle(frame, (x3,y3),(x4,y4),(0,255,255),2)
                cvzone.putTextRect(frame,f'{id}', (x3,y3), 1,2)
                if counter2.count(id)==0:
                    print(len(counter1))
                    counter2.append(id)

    cv2.line(frame,(cx1, 3), (cx1, 1018),(0,255,0),2)
    cv2.line(frame,(cx2, 5), (cx2, 1019),(0,255,255),2)

    #print(persondown)
    #print(counter1)
    #print(len(counter1)) #lenght means we can get the counnt who is going down
    downcount=len(counter1)
    upcount=len(counter2)
    
    cvzone.putTextRect(frame, f'In: {downcount}', (50,60), 2,2)
    cvzone.putTextRect(frame, f'Out: {upcount}', (50,160), 2,2)
    output.write(frame)
    cv2.imshow('RGB', frame)
    if cv2.waitKey(1) & 0xff==27:
        break

cap.release()
cv2.destroyAllWindows()



