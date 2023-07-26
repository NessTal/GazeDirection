import cv2
import mediapipe as mp
import itertools
import numpy as np
import matplotlib.pyplot as plt

mp_face_mesh = mp.solutions.face_mesh

face_mesh_images = mp_face_mesh.FaceMesh(static_image_mode=True, 
                                         refine_landmarks=True,
                                         max_num_faces=1,
                                         min_detection_confidence=0.5)

mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles

#mp_holistic = mp.solutions.holistic

sample_img = cv2.imread('face.png')
image_height, image_width, _ = sample_img.shape

### temp
plt.figure(figsize = [10, 10])
plt.title("Sample Image");plt.axis('off');plt.imshow(sample_img[:,:,::-1]);plt.show()
###

face_mesh_results = face_mesh_images.process(sample_img[:,:,::-1])

LEFT_EYE_INDEXES = list(set(itertools.chain(*mp_face_mesh.FACEMESH_LEFT_EYE)))
RIGHT_EYE_INDEXES = list(set(itertools.chain(*mp_face_mesh.FACEMESH_RIGHT_EYE)))
LEFT_IRIS_INDEXES = list(set(itertools.chain(*mp_face_mesh.FACEMESH_LEFT_IRIS)))
RIGHT_IRIS_INDEXES = list(set(itertools.chain(*mp_face_mesh.FACEMESH_RIGHT_IRIS)))
CONTOUS_INDEXES = list(set(itertools.chain(*mp_face_mesh.FACEMESH_CONTOURS)))

INDEXES = LEFT_EYE_INDEXES + RIGHT_EYE_INDEXES + LEFT_IRIS_INDEXES + RIGHT_IRIS_INDEXES + CONTOUS_INDEXES


points = np.array([np.multiply([p.x, p.y], [image_width, image_height]).astype(int) for p in face_mesh_results.multi_face_landmarks[0].landmark])
points[INDEXES]
