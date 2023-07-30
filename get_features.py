import cv2
import mediapipe as mp
import itertools
import pandas as pd
import numpy as np
import os

pd.set_option('display.max_columns', 500)
#pd.set_option('display.max_rows', 100)

mp_face_mesh = mp.solutions.face_mesh
face_mesh_images = mp_face_mesh.FaceMesh(static_image_mode=True, 
                                         refine_landmarks=True,
                                         max_num_faces=1,
                                         min_detection_confidence=0,
                                         min_tracking_confidence=0)


# Get indexes of all relevant landmarks (eyes, iris, and contour)
LEFT_EYE_INDEXES = list(set(itertools.chain(*mp_face_mesh.FACEMESH_LEFT_EYE)))
RIGHT_EYE_INDEXES = list(set(itertools.chain(*mp_face_mesh.FACEMESH_RIGHT_EYE)))
LEFT_IRIS_INDEXES = list(set(itertools.chain(*mp_face_mesh.FACEMESH_LEFT_IRIS)))
RIGHT_IRIS_INDEXES = list(set(itertools.chain(*mp_face_mesh.FACEMESH_RIGHT_IRIS)))
CONTOUS_INDEXES = list(set(itertools.chain(*mp_face_mesh.FACEMESH_CONTOURS)))

# Remove duplicate indexes
CONTOUS_INDEXES = [i for i in CONTOUS_INDEXES if i not in LEFT_EYE_INDEXES + RIGHT_EYE_INDEXES + LEFT_IRIS_INDEXES + RIGHT_IRIS_INDEXES]

# Combine all indexes to one list
INDEXES = LEFT_EYE_INDEXES + RIGHT_EYE_INDEXES + LEFT_IRIS_INDEXES + RIGHT_IRIS_INDEXES + CONTOUS_INDEXES


def get_features(file):
    '''
    Extracts the relevan landmarks for each time point in a video file using MediaPipe's FaceMesh model,
      to be used as features for a machine learning model.
    
    Inputs: file (str) - path to video file
    
    Outputs:
    points_over_time (pd.DataFrame) - dataframe with columns:
        'File' (str) - name of video file,
        Time (float) - time in milliseconds,
        and a column for each landmark ([x,y]).
    '''
    
    # Load video
    video = cv2.VideoCapture(file)

    points_over_time = []
    ret = True

    # Downsample if fps is greater than 30
    fps = video.get(cv2.CAP_PROP_FPS)
    if fps > 30:
        reduce_frame_rate = int(fps/30)   # every 30 milliseconds (unless fps is less than 30)
    else:
        reduce_frame_rate = 1
    i = 0

    # Iterate over frames and get landmarks
    while ret:
        if i % reduce_frame_rate == 0:
            ret, frame = video.read()

            if not ret:
                break

            #image_height, image_width, _ = frame.shape
            face_mesh_results = face_mesh_images.process(frame[:,:,::-1])
            points = np.array([[p.x, p.y] for p in face_mesh_results.multi_face_landmarks[0].landmark])
            #points = np.array([np.multiply([p.x, p.y], [image_width, image_height]).astype(int) for p in face_mesh_results.multi_face_landmarks[0].landmark])
            points_over_time.append((file,i*(1000/fps),*points[INDEXES]))

            i += 1

            #cv2.imshow('frame', frame)
            #if cv2.waitKey(1) & 0xFF == ord('q'):
            #    break

        else:
            ret = video.grab()
            i += 1

    video.release()
    #video.destroyAllWindows()

    # Convert to dataframe
    points_over_time = pd.DataFrame(points_over_time, columns=['File','Time',*INDEXES])

    return points_over_time


def iterate_over_files(folder,extension):
    '''
    Iterates over all files in a folder and extracts the relevant landmarks 
      for each time point in each video file using MediaPipe's FaceMesh model,
      to be used as features for a machine learning model.

    Inputs: folder (str) - path to folder containing video files,
            extension (str) - file extension of video files
    
    Outputs: points_over_time_df (pd.DataFrame) - dataframe containing the last batch of files.
    '''
    files = os.listdir(folder)
    files.sort()

    points_over_time_list = []
    file_count = 0
    for file in files:
        if file.endswith(extension):
            print(file_count)
            points_over_time = get_features(folder+'/'+file)
            points_over_time_list.append(points_over_time)
            file_count += 1
            if file_count % 500 == 0:
                points_over_time_df = pd.concat(points_over_time_list)
                points_over_time_df.to_parquet('points_over_time_'+str(file_count)+'.parquet')
                points_over_time_list = []
    points_over_time_df = pd.concat(points_over_time_list)
    points_over_time_df.to_parquet('points_over_time_'+str(file_count)+'.parquet')
    return points_over_time_df


# Run on all video files
folder = 'video_files'
extension = '.mp4'
points_over_time_df = iterate_over_files(folder,extension)


# Combine all parquet files into one dataframe
files = os.listdir()
files = [file for file in files if file.startswith('points_over_time_') and file.endswith('.parquet')]
points_over_time_df = pd.concat([pd.read_parquet(file) for file in files])
points_over_time_df.to_parquet('points_over_time.parquet')
# points_over_time_df = pd.read_parquet('points_over_time.parquet')

# Load hand-coded data
hand_coded_df = pd.read_parquet('EyeCodingResults_cleaned.parquet')
# hand_coded_df = hand_coded_df.loc[hand_coded_df['Time'] % 20 == 0] # downsample to 20ms (use if needed)

# Merge hand-coded data with points_over_time_df
points_over_time_df['File'] = points_over_time_df['File'].map(lambda x: x.split('/')[1])
points_over_time_df['Time'] = points_over_time_df['Time'].round(0).astype(int)
points_over_time_df = points_over_time_df.merge(hand_coded_df[['File','Time','Code']], on=['File','Time'], how='left')
points_over_time_df.to_parquet('landmarks_and_hand_coding.parquet')

# Check fps bug
hand_coded_max = pd.DataFrame(hand_coded_df.groupby('File')['Time'].max())
max_times_df = pd.DataFrame(points_over_time_df.groupby('File')['Time'].max()).merge(hand_coded_max, on='File', how='left',suffixes=['_points','_hand'])
max_times_df['Diff'] = max_times_df['Time_points'] - max_times_df['Time_hand']
max_times_df['Category'] = pd.cut(max_times_df['Diff'], bins=[-5000, 0, 350, 5000], include_lowest=True, labels=['minus', 'small', 'large'])


#df = pd.read_excel('EyeCodingResults.xlsx')
#df.to_parquet('EyeCodingResults.parquet')
#df.to_parquet('EyeCodingResults_cleaned.parquet')
#df = pd.read_parquet('EyeCodingResults_cleaned.parquet')
#df = pd.read_parquet('points_over_time.parquet')