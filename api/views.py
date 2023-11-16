from django.shortcuts import render
from rest_framework import status
from rest_framework.response import Response
from rest_framework.views import APIView
from .serializers import VerificationSerializer
import cv2
import face_recognition
import pickle
import os
import numpy as np
import json
from datetime import datetime
from PIL import Image
import base64

class VerificationView(APIView):
    serializer_class = VerificationSerializer
            
    def match_student(self, frame_encoding,roll):
        encodings_file = f"encodings\\{roll}.pkl"
        with open(encodings_file, 'rb') as f:
            student_data = pickle.load(f)
        for roll_number, known_encoding in student_data.items():
            matches = face_recognition.compare_faces([known_encoding], frame_encoding)
            if any(matches):
                return roll_number
        return None

    def recognize_face(self, frame,roll):
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        face_locations = face_recognition.face_locations(rgb_frame)
        face_encodings = face_recognition.face_encodings(rgb_frame, face_locations)

        for encoding in face_encodings:
            roll_number = self.match_student(encoding,roll)
            if roll_number:
                return roll_number

        return None

    def verify(self, roll, frame):
        detect_roll = self.recognize_face(frame,roll)
        roll = str(roll)
        detect_roll = str(detect_roll)
        if detect_roll == roll:
            return True
        else:
            return False

    def post(self, request, format=None):
        serializer = self.serializer_class(data=request.data)
        if serializer.is_valid():
            json_file = "failed_result.json"
            roll = serializer.validated_data['roll']
            if not os.path.exists(f"encodings\\{roll}.pkl"):
                current_datetime = datetime.now()
                date_time_string = current_datetime.strftime("%Y-%m-%d %H:%M:%S")
                print(f"Roll number '{roll}' does not exist ---- added to {json_file} file")
                with open(json_file, 'a') as json_file:
                    data = {
                        "roll": roll,
                        "date_time": date_time_string,
                        "status": "PKL file does not exist"
                    }
                    json_file.write(json.dumps(data) + '\n')

                return Response({'verified': False}, status=status.HTTP_200_OK)
            image = serializer.validated_data['image']
            decoded_image = base64.b64decode(image)
            nparr = np.frombuffer(decoded_image, np.uint8)
            frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
            frame2 = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            result = self.verify(roll, frame)
            if result:
                print(f"Request for Roll Number {roll} returns {result}")
            else:
                print(f"Request for Roll Number {roll} returns ---- {result} ---- added to {json_file} file")
                current_datetime = datetime.now()
                date_time_string = current_datetime.strftime("%Y-%m-%d %H:%M:%S")
                with open(json_file, 'a') as json_file:
                    data = {
                        "roll": roll,
                        "date_time": date_time_string,
                        "status" : " Image Not Matched"
                    }
                    json_file.write(json.dumps(data) + '\n')
                output_filename = f"false_results\\{roll}.jpg"
                output_filename=os.path.join(os.getcwd(),output_filename)
                img=Image.fromarray(frame2)
                img.save(output_filename)
                
            return Response({'verified': result}, status=status.HTTP_200_OK)
        return Response(serializer.errors, status=status.HTTP_400_BAD_REQUEST)
