import cv2
import numpy as np
import os
from matplotlib import pyplot as plt
import time
import mediapipe as mp

mp_holistic = mp.solutions.holistic # Holistic model
mp_drawing = mp.solutions.drawing_utils #araçlar

def mediapipe_detection(image, model):
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB) # Renk dönüşümü BGR'den RGB
    image.flags.writeable = False                  # Resim artık yazılabilir değil
    results = model.process(image)                 # Tahmin etmek
    image.flags.writeable = True                   # Resim artık yazılabilir
    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR) # Renk dönüşümü RGB'den BGR
    return image, results

def draw_styled_landmarks(image, results):
    # Yüz bağlantıları
    mp_drawing.draw_landmarks(image, results.face_landmarks, mp_holistic.FACEMESH_CONTOURS, 
                             mp_drawing.DrawingSpec(color=(80,110,10), thickness=1, circle_radius=1),#nokta stili
                             mp_drawing.DrawingSpec(color=(80,256,121), thickness=1, circle_radius=1) #cizgi stili
                             ) 
    # Poz bağlantıları
    mp_drawing.draw_landmarks(image, results.pose_landmarks, mp_holistic.POSE_CONNECTIONS,
                             mp_drawing.DrawingSpec(color=(80,22,10), thickness=2, circle_radius=4), 
                             mp_drawing.DrawingSpec(color=(80,44,121), thickness=2, circle_radius=2)
                             ) 
    # Sol el bağlatıları
    mp_drawing.draw_landmarks(image, results.left_hand_landmarks, mp_holistic.HAND_CONNECTIONS, 
                             mp_drawing.DrawingSpec(color=(121,22,76), thickness=2, circle_radius=4), 
                             mp_drawing.DrawingSpec(color=(121,44,250), thickness=2, circle_radius=2)
                             ) 
    # Sağ el bağlantıları
    mp_drawing.draw_landmarks(image, results.right_hand_landmarks, mp_holistic.HAND_CONNECTIONS, 
                             mp_drawing.DrawingSpec(color=(245,117,66), thickness=2, circle_radius=4), 
                             mp_drawing.DrawingSpec(color=(245,66,230), thickness=2, circle_radius=2)
                             ) 

def extract_keypoints(results):
    pose = np.array([[res.x, res.y, res.z, res.visibility] for res in results.pose_landmarks.landmark]).flatten() if results.pose_landmarks else np.zeros(33*4)
    face = np.array([[res.x, res.y, res.z] for res in results.face_landmarks.landmark]).flatten() if results.face_landmarks else np.zeros(468*3)
    lh = np.array([[res.x, res.y, res.z] for res in results.left_hand_landmarks.landmark]).flatten() if results.left_hand_landmarks else np.zeros(21*3)
    rh = np.array([[res.x, res.y, res.z] for res in results.right_hand_landmarks.landmark]).flatten() if results.right_hand_landmarks else np.zeros(21*3)
    return np.concatenate([pose, face, lh, rh])

DATA_PATH = os.path.join('MP_Data333') 
actions = np.array(['evet','gulegule','hayir','iyi','merhaba','nasilsin','ozurdilemek','sevmek'])
no_sequences = 30
sequence_length = 30
start_folder = 1
label_map = {label:num for num, label in enumerate(actions)}
for action in actions:
    for sequence in range(1,no_sequences+1):
        try: 
            os.makedirs(os.path.join(DATA_PATH, action, str(sequence)))
        except:
            pass

cap = cv2.VideoCapture(0)
# Mediapipe modelini ayarla
with mp_holistic.Holistic(min_detection_confidence=0.5, min_tracking_confidence=0.5) as holistic:
    # Eylemler arasında geçiş döngüsü
    for action in actions:
        # Videolar olarak belirtilen diziler arasında döngü
        for sequence in range(start_folder, start_folder+no_sequences):
            # Video uzunluk döngüsü,
            for frame_num in range(sequence_length):
                # Özet akışını oku
                ret, frame = cap.read()
                # Algılama yap
                image, results = mediapipe_detection(frame, holistic)

                # landmarks çizimleri
                draw_styled_landmarks(image, results)
                
                # Bekleme mantığını uygula
                if frame_num == 0: 
                    cv2.putText(image, 'STARTING COLLECTION', (120,200), 
                               cv2.FONT_HERSHEY_SIMPLEX, 1, (0,255, 0), 4, cv2.LINE_AA)
                    cv2.putText(image, 'Collecting frames for {} Video Number {}'.format(action, sequence), (15,12), 
                               cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1, cv2.LINE_AA)
                    # Ekrana göster
                    cv2.imshow('OpenCV Feed', image)
                    cv2.waitKey(500)
                else: 
                    cv2.putText(image, 'Collecting frames for {} Video Number {}'.format(action, sequence), (15,12), 
                               cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1, cv2.LINE_AA)
                    # Ekrana göster
                    cv2.imshow('OpenCV Feed', image)
                
                # keypoinleri dışa aktarma
                keypoints = extract_keypoints(results)
                npy_path = os.path.join(DATA_PATH, action, str(sequence), str(frame_num))
                
                
                # Kamerayı kapat yani cap.relase dönügüsünü sonlandır
                if cv2.waitKey(10) & 0xFF == ord('q'):
                    break
                    
    cap.release()
    cv2.destroyAllWindows()