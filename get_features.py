import cv2
import mediapipe as mp
import itertools
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

mp_face_mesh = mp.solutions.face_mesh

face_mesh_images = mp_face_mesh.FaceMesh(static_image_mode=True, 
                                         refine_landmarks=True,
                                         max_num_faces=1,
                                         min_detection_confidence=0,
                                         min_tracking_confidence=0)

mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles


LEFT_EYE_INDEXES = list(set(itertools.chain(*mp_face_mesh.FACEMESH_LEFT_EYE)))
RIGHT_EYE_INDEXES = list(set(itertools.chain(*mp_face_mesh.FACEMESH_RIGHT_EYE)))
LEFT_IRIS_INDEXES = list(set(itertools.chain(*mp_face_mesh.FACEMESH_LEFT_IRIS)))
RIGHT_IRIS_INDEXES = list(set(itertools.chain(*mp_face_mesh.FACEMESH_RIGHT_IRIS)))
CONTOUS_INDEXES = list(set(itertools.chain(*mp_face_mesh.FACEMESH_CONTOURS)))

INDEXES = LEFT_EYE_INDEXES + RIGHT_EYE_INDEXES + LEFT_IRIS_INDEXES + RIGHT_IRIS_INDEXES + CONTOUS_INDEXES

file = 'face_video.mp4'

def get_features(file):
    video = cv2.VideoCapture(file)

    points_over_time = []
    ret = True
    reduce_frame_rate = video.get(cv2.CAP_PROP_FPS)/50   # every 20 milliseconds
    i = 0

    while ret:
        if i % reduce_frame_rate == 0:
            ret, frame = video.read()

            if not ret:
                break

            image_height, image_width, _ = frame.shape
            face_mesh_results = face_mesh_images.process(frame[:,:,::-1])
            points = np.array([np.multiply([p.x, p.y], [image_width, image_height]).astype(int) for p in face_mesh_results.multi_face_landmarks[0].landmark])
            points[INDEXES]
            points_over_time.append((i,points[INDEXES]))

            i += 1

            cv2.imshow('frame', frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

        else:
            ret = video.grab()
            i += 1

    video.release()
    #video.destroyAllWindows()
    
    return points_over_time



df = pd.read_excel('EyeCodingResults.xlsx')
df.to_parquet('EyeCodingResults.parquet')
df.to_parquet('EyeCodingResults_cleaned.parquet')
df = pd.read_parquet('EyeCodingResults_cleaned.parquet')
