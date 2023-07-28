import cv2
import mediapipe as mp
import itertools
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os


mp_face_mesh = mp.solutions.face_mesh
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles
face_mesh_images = mp_face_mesh.FaceMesh(static_image_mode=True, 
                                         refine_landmarks=True,
                                         max_num_faces=1,
                                         min_detection_confidence=0,
                                         min_tracking_confidence=0)


LEFT_EYE_INDEXES = list(set(itertools.chain(*mp_face_mesh.FACEMESH_LEFT_EYE)))
RIGHT_EYE_INDEXES = list(set(itertools.chain(*mp_face_mesh.FACEMESH_RIGHT_EYE)))
LEFT_IRIS_INDEXES = list(set(itertools.chain(*mp_face_mesh.FACEMESH_LEFT_IRIS)))
RIGHT_IRIS_INDEXES = list(set(itertools.chain(*mp_face_mesh.FACEMESH_RIGHT_IRIS)))
CONTOUS_INDEXES = list(set(itertools.chain(*mp_face_mesh.FACEMESH_CONTOURS)))

INDEXES = LEFT_EYE_INDEXES + RIGHT_EYE_INDEXES + LEFT_IRIS_INDEXES + RIGHT_IRIS_INDEXES + CONTOUS_INDEXES


files = os.listdir('video_files')

hand_coded_df = pd.read_parquet('EyeCodingResults_cleaned.parquet')
# hand_coded_df = hand_coded_df.loc[hand_coded_df['Time'] % 20 == 0] # downsample to 20ms (use if needed)


def get_features(file):
    video = cv2.VideoCapture(file)

    points_over_time = []
    ret = True
    fps = video.get(cv2.CAP_PROP_FPS)
    reduce_frame_rate = int(fps/30)   # every 30 milliseconds (unless fps is less than 30)
    if reduce_frame_rate < 1:
        reduce_frame_rate = 1
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
            points_over_time.append((file,i*fps,points[INDEXES]))

            i += 1

            cv2.imshow('frame', frame)
            #if cv2.waitKey(1) & 0xFF == ord('q'):
            #    break

        else:
            ret = video.grab()
            i += 1

    video.release()
    #video.destroyAllWindows()
    points_over_time = pd.DataFrame(points_over_time, columns=['File','Time','Points'])
    return points_over_time


points_over_time_df = pd.DataFrame(columns=['File','Time','Points'])
for file in files[:10]:
    if file.endswith('.mp4'):
        points_over_time = get_features('video_files/'+file)
        points_over_time_df = pd.concat([points_over_time_df,points_over_time])









#df = pd.read_excel('EyeCodingResults.xlsx')
#df.to_parquet('EyeCodingResults.parquet')
#df.to_parquet('EyeCodingResults_cleaned.parquet')
#df = pd.read_parquet('EyeCodingResults_cleaned.parquet')
