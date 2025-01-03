import cv2, pandas, time
from ultralytics import YOLO
from datetime import datetime

## Loading yolo model
model = YOLO(r'D:\Project file\Object Detection Using Yolo\Object Detection using Yolo Model\model.pt')
print(model.names)
first_frame = None
status_list = [None,None]
times = []
df=pandas.DataFrame(columns=["Start","End"])

## here 0th index for own device camera capture
webcamera = cv2.VideoCapture(0)

## Check if the camera opened successfully
if not webcamera.isOpened():
    print("Falled to open the Webcam Please Verify and try angain......")
    exit()

while True:
    ## Capturing frame in cam
    success, frame = webcamera.read()
    if not success:
        print("Failed to capture frame from camera. Exiting...")
        break

    results = model.predict(frame, conf=0.8)
    frame_with_detections = results[0].plot()

    cv2.imshow("Laptop Camera - YOLO Object Detection", frame_with_detections)

    if cv2.waitKey(1) == ord('q'):
        break

    check, frame = webcamera.read()
    status = 0
    gray = cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)
    gray = cv2.GaussianBlur(gray,(21,21),0)

    if first_frame is None:
        first_frame=gray
        continue

    delta_frame=cv2.absdiff(first_frame,gray)
    thresh_frame=cv2.threshold(delta_frame, 30, 255, cv2.THRESH_BINARY)[1]
    thresh_frame=cv2.dilate(thresh_frame, None, iterations=2)

    (cnts,_)=cv2.findContours(thresh_frame.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    for contour in cnts:
        if cv2.contourArea(contour) < 10000:
            continue
        status=1

        (x, y, w, h)=cv2.boundingRect(contour)
        cv2.rectangle(frame, (x, y), (x+w, y+h), (0,255,0), 3)
    status_list.append(status)

    status_list=status_list[-2:]


    if status_list[-1]==1 and status_list[-2]==0:
        times.append(datetime.now())
    if status_list[-1]==0 and status_list[-2]==1:
        times.append(datetime.now())


    # cv2.imshow("Gray Frame",gray)
    # cv2.imshow("Delta Frame",delta_frame)
    # cv2.imshow("Threshold Frame",thresh_frame)
    cv2.imshow("Color Frame",frame)

    key=cv2.waitKey(1)

    if key == ord('q'):
        if status == 1:
            times.append(datetime.now())
        break

print(status_list)
print(times)

for i in range(0, len(times), 2):
    df = df.append({"Start": times[i],"End": times[i+1]}, ignore_index=True)

df.to_csv("Times.csv")

webcamera.release()
cv2.destroyAllWindows()


## Documentation
'''
Project Name: Object Detection using Yolo
Created By: Santosh Kumar
Creating Date: 02.01.2025
'''