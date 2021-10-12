from django.http import HttpResponse
from django.shortcuts import render,redirect
from .models import *
from django.core.mail import EmailMessage
from django.views.decorators import gzip
from django.http import StreamingHttpResponse
import cv2
import threading
from django.templatetags.static import static

import mediapipe as mp
import pickle
import pandas as pd
import numpy as np

import time
from datetime import datetime, date

# from django.core.mail import send_mail
import smtplib
import ssl
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart

port = 465
smtp_server = "smtp.gmail.com"
sender_email = "pnpitmshackathonemail@gmail.com"
receiver_email = "jarodaustria@gmail.com"
password = "adminpassword2021"

mp_drawing = mp.solutions.drawing_utils # Drawing helpers
mp_holistic = mp.solutions.holistic # Mediapipe Solutions

with open('webcam/model/body_language.pkl', 'rb') as f:
    model = pickle.load(f)

def index(request):
    return render(request, 'webcam/dashboard.html')

def pending_crimes(request):
    crimes = Crime.objects.all()
    total_crimes = len(crimes)
    crimes = Crime.objects.filter(validated=False)
    total_unvalidated_crimes = len(crimes)
    context = {
        'crimes': crimes,
        'total_crimes':total_crimes,
        'total_unvalidated_crimes': total_unvalidated_crimes,
    }
    return render(request, 'webcam/pending_crimes.html', context)
def crimes(request):
    crimes = Crime.objects.all()
    total_crimes = len(crimes)
    crimes = Crime.objects.filter(validated=True)
    total_validated_crimes = len(crimes)
    context = {
        'crimes': crimes,
        'total_crimes':total_crimes,
        'total_validated_crimes': total_validated_crimes,
    }
    return render(request, 'webcam/crimes.html', context)

def update_crime_true(request,pk):
    crime = Crime.objects.get(pk=pk)
    crime.correct = True
    crime.validated = True
    crime.save()
    crimes = Crime.objects.all()
    
    context = {
        'crimes': crimes,
    }
    return redirect('/crimes')
def update_crime_false(request,pk):
    crime = Crime.objects.get(pk=pk)
    crime.correct = False
    crime.validated = True
    crime.save()
    crimes = Crime.objects.all()
    
    context = {
        'crimes': crimes,
    }
    return redirect('/crimes')




@gzip.gzip_page
def Home(request):
    try:
        cam = VideoCamera()
        return StreamingHttpResponse(gen(cam), content_type="multipart/x-mixed-replace;boundary=frame")
    except:
        pass
    return render(request, 'webcam/dashboard.html')

#to capture video class
class VideoCamera(object):
    def __init__(self):
        self.video = cv2.VideoCapture(0)
        (self.grabbed, self.frame) = self.video.read()
        threading.Thread(target=self.update, args=()).start()

    def __del__(self):
        self.video.release()

    def get_frame(self):
        image = self.frame
        _, jpeg = cv2.imencode('.jpg', image)
        return jpeg.tobytes()
    
    def send_mail(self):
        print("sending email...")
        message = MIMEMultipart("alternative")
        message["Subject"] = "Possible Crime!"
        message["From"] = sender_email
        message["To"] = receiver_email
        text = """\
            Date: """+str(date.today())+"""
            Time: """+str(datetime.now().strftime("%H:%M:%S"))+"""

            There's a possible crime in this location! Check immediately.
            """

        part1 = MIMEText(text, "plain")
        # part2 = MIMEText(html, "html")

        message.attach(part1)
        # message.attach(part2)

        context = ssl.create_default_context()
        with smtplib.SMTP_SSL(smtp_server, port, context=context) as server:
            server.login(sender_email, password)
            server.sendmail(sender_email, receiver_email,
                            message.as_string())
        print("email sent!")

    def update(self):
        with mp_holistic.Holistic(min_detection_confidence=0.5, min_tracking_confidence=0.5) as holistic:
        # while True:
            start = time.time()
            while self.video.isOpened():
                (self.grabbed, self.frame) = self.video.read()
                # Recolor Feed
                image = cv2.cvtColor(self.frame, cv2.COLOR_BGR2RGB)
                image.flags.writeable = False  

                results = holistic.process(image)
                # print(results.face_landmarks)
                
                # face_landmarks, pose_landmarks, left_hand_landmarks, right_hand_landmarks
                
                # Recolor image back to BGR for rendering
                self.frame.flags.writeable = True   
                # self.frame = cv2.cvtColor(self.frame, cv2.COLOR_RGB2BGR)
                

                # 2. Right hand
                mp_drawing.draw_landmarks(self.frame, results.right_hand_landmarks, mp_holistic.HAND_CONNECTIONS, 
                                        mp_drawing.DrawingSpec(color=(80,22,10), thickness=2, circle_radius=4),
                                        mp_drawing.DrawingSpec(color=(80,44,121), thickness=2, circle_radius=2)
                                        )

                # 3. Left Hand
                mp_drawing.draw_landmarks(self.frame, results.left_hand_landmarks, mp_holistic.HAND_CONNECTIONS, 
                                        mp_drawing.DrawingSpec(color=(121,22,76), thickness=2, circle_radius=4),
                                        mp_drawing.DrawingSpec(color=(121,44,250), thickness=2, circle_radius=2)
                                        )

                # 4. Pose Detections
                mp_drawing.draw_landmarks(self.frame, results.pose_landmarks, mp_holistic.POSE_CONNECTIONS, 
                                        mp_drawing.DrawingSpec(color=(245,117,66), thickness=2, circle_radius=4),
                                        mp_drawing.DrawingSpec(color=(245,66,230), thickness=2, circle_radius=2)
                                        )
                try:
                    
                    # Extract Pose landmarks
                    pose = results.pose_landmarks.landmark
                    pose_row = list(np.array([[landmark.x, landmark.y, landmark.z, landmark.visibility] for landmark in pose]).flatten())
                    # Extract Face landmarks
        #             face = results.face_landmarks.landmark
        #             face_row = list(np.array([[landmark.x, landmark.y, landmark.z, landmark.visibility] for landmark in face]).flatten())
                    
                    # Concate rows
                    row = pose_row#+face_row
                    
        #             # Append class name 
        #             row.insert(0, class_name)
                    
        #             # Export to CSV
        #             with open('bodyposture.csv', mode='a', newline='') as f:
        #                 csv_writer = csv.writer(f, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)
        #                 csv_writer.writerow(row) 

                    # Make Detections
                    X = pd.DataFrame([row])
                    body_language_class = model.predict(X)[0]
                    body_language_prob = model.predict_proba(X)[0]

                    end = time.time()
                    interval = end-start
                    if body_language_class == "Punching" and interval > 5:
                        start = time.time()
                        end = time.time()
                        stump = time.time()

                        c = Crime(
                            classification = body_language_class,
                            # correct = False,
                            image = '/webcam/crimes/'+str(stump)+'.jpg'
                        )
                        c.save()
                        cv2.imwrite('webcam/static/webcam/crimes/'+str(stump)+'.jpg', self.frame)
                        
                        #send mail
                        threading.Thread(target=self.send_mail, args=()).start()
                        
                    
                        print(body_language_class, body_language_prob, interval)
                    
                    # Grab ear coords
                    coords = tuple(np.multiply(
                                    np.array(
                                        (results.pose_landmarks.landmark[mp_holistic.PoseLandmark.LEFT_EAR].x, 
                                        results.pose_landmarks.landmark[mp_holistic.PoseLandmark.LEFT_EAR].y))
                                , [640,480]).astype(int))
                    
                    cv2.rectangle(self.frame, 
                                (coords[0], coords[1]+5), 
                                (coords[0]+len(body_language_class)*20, coords[1]-30), 
                                (245, 117, 16), -1)
                    cv2.putText(self.frame, body_language_class, coords, 
                                cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2, cv2.LINE_AA)
                    
                    # Get status box
                    cv2.rectangle(self.frame, (0,0), (250, 60), (245, 117, 16), -1)
                    
                    # Display Class
                    cv2.putText(self.frame, 'CLASS'
                                , (95,12), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1, cv2.LINE_AA)
                    cv2.putText(self.frame, body_language_class.split(' ')[0]
                                , (90,40), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2, cv2.LINE_AA)
                    
                    # Display Probability
                    cv2.putText(self.frame, 'PROB'
                                , (15,12), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1, cv2.LINE_AA)
                    cv2.putText(self.frame, str(round(body_language_prob[np.argmax(body_language_prob)],2))
                                , (10,40), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2, cv2.LINE_AA)
                    
                except Exception as e: 
                    print(e)
                    

                # cv2.imshow('Raw Webcam Feed', self.frame)

                if cv2.waitKey(10) & 0xFF == ord('q'):
                    break

def gen(camera):
    while True:
        frame = camera.get_frame()
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n\r\n')