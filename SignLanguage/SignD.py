import cv2
import numpy as np
import os
from matplotlib import pyplot as plt
import time
import mediapipe as mp

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout

from scipy import stats
from gtts import gTTS
import os
from pygame import mixer

import PIL.Image
import PIL.ImageTk
from tkinter import *


mp_holistic = mp.solutions.holistic # Holistic model
mp_drawing = mp.solutions.drawing_utils #araçlar

def mediapipe_detection(image, model):
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB) # Renk dönüşümü BGR'den RGB
    image.flags.writeable = False                  # Resim artık yazılabilir değil
    results = model.process(image)                 # Tahmin etmek
    image.flags.writeable = True                   # Resim artık yazılabilir
    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR) # Renk dönüşümü RGB'den BGR
    return image, results

def extract_keypoints(results):
    pose = np.array([[res.x, res.y, res.z, res.visibility] for res in results.pose_landmarks.landmark]).flatten() if results.pose_landmarks else np.zeros(33*4)
    face = np.array([[res.x, res.y, res.z] for res in results.face_landmarks.landmark]).flatten() if results.face_landmarks else np.zeros(468*3)
    lh = np.array([[res.x, res.y, res.z] for res in results.left_hand_landmarks.landmark]).flatten() if results.left_hand_landmarks else np.zeros(21*3)
    rh = np.array([[res.x, res.y, res.z] for res in results.right_hand_landmarks.landmark]).flatten() if results.right_hand_landmarks else np.zeros(21*3)
    return np.concatenate([pose, face, lh, rh])

DATA_PATH = os.path.join('MP_Data2') 

actions = np.array(['evet','gulegule','hayir','iyi','merhaba','nasilsin','ozurdilemek','sevmek'])
no_sequences = 30
sequence_length = 30

for action in actions:
    for sequence in range(1,no_sequences+1):
        try: 
            os.makedirs(os.path.join(DATA_PATH, action, str(sequence)))
        except:
            pass


model = Sequential()
model.add(LSTM(256, return_sequences=True, activation='sigmoid', input_shape=(30,1662)))
model.add(Dropout(0.2))
model.add(LSTM(128, return_sequences=True, activation='sigmoid'))
model.add(Dropout(0.2))
model.add(LSTM(64, activation='sigmoid'))
model.add(Dense(64))
model.add(Dropout(0.2))
model.add(Dense(64))
model.add(Dense(actions.shape[0], activation='softmax'))

model.load_weights('action.h5')
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
model.summary()

def prob_viz(res, actions, input_frame, colors):
    output_frame = input_frame.copy()
    for num, prob in enumerate(res):
        cv2.rectangle(output_frame, (0, 60 + num * 40), (int(prob * 100), 90 + num * 40), colors[num], -1)
        cv2.putText(output_frame, actions[num], (0, 85 + num * 40), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2,
        cv2.LINE_AA)

    return output_frame



def openCam():
    sequence = []
    sentence = []
    predictions = []
    threshold = 0.8
    cap = cv2.VideoCapture(0)
    with mp_holistic.Holistic(min_detection_confidence=0.5, min_tracking_confidence=0.5) as holistic:
        while cap.isOpened():

            ret, frame = cap.read()

            image, results = mediapipe_detection(frame, holistic)
            print(results)


            # 2. Tahmin Bölümü
            keypoints = extract_keypoints(results)
            sequence.append(keypoints)
            sequence = sequence[-30:]

            if len(sequence) == 30:
                res = model.predict_proba(np.expand_dims(sequence, axis=0))[0]
                print(actions[np.argmax(res)])
                predictions.append(np.argmax(res))

                if np.unique(predictions[-10:])[0] == np.argmax(res):
                    if res[np.argmax(res)] > threshold:
                        if len(sentence) > 0:
                            #son cümleyle aynı olup olmadığını sorgular
                            if actions[np.argmax(res)] != sentence[-1] and actions[np.argmax(res)] != "gulegule":
                                predictions.append(np.argmax(res))
                                sentence.append(actions[np.argmax(res)])
                                mixer.init()
                                mixer.music.load("ses/"+sentence[-1] + '.mp3')
                                mixer.music.play()
                        else:
                            sentence.append(actions[np.argmax(res)])
                            mixer.init()
                            mixer.music.load("ses/"+sentence[-1] + '.mp3')
                            mixer.music.play()

                if len(sentence) > 5:
                    sentence = sentence[-5:]
            cv2.rectangle(image, (0, 0), (640, 40), (245, 117, 16), -1)
            cv2.putText(image, ' '.join(sentence), (3, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2, cv2.LINE_AA)

            cv2.imshow('Sign Language', image)

            if cv2.waitKey(10) & 0xFF == ord('q'):
                break
        cap.release()
        cv2.destroyAllWindows()


bg_color= '#9a8c98'
bg_color2='#f2e9e4'
tx_color= '#22223b'
sub_tx_color= '#4a4e69'

window = Tk()
window.title("İşaret Dili Algılama")
canvas = Canvas(window, height=450, width=750, bg = bg_color)
canvas.pack()


frame_ust = Frame(window, bg = bg_color2)
frame_ust.place(relx=0.1, rely=0.1, relwidth=0.80, relheight=0.1)

frame_sol_alt = Frame(window, bg=bg_color2)
frame_sol_alt.place(relx=0.1, rely=0.21, relwidth=0.30, relheight=0.60)

frame_sag_alt = Frame(window, bg=bg_color2)
frame_sag_alt.place(relx=0.41, rely=0.21, relwidth=0.49, relheight=0.60)

Label(frame_sag_alt, bg=bg_color2, fg=tx_color, text = "Program Hakkında", font = "Verdana 12 bold").pack()
Label(frame_sag_alt, bg=bg_color2, text ='Program Türkçe işaret dilindeki 6 hareketi algılamak üzere eğitilmiştir.\n Programın algıladığı hareketler aşağıda sıralanmıştır:\n Merhaba \n Güle güle \n Hayır \n İyi \n Özür dilerim \nsevmek \n \n Daha fazla veriyle model genişletilebilir.\n Bu Program Ekin Mete Tahmilci, Ahmet Can Işıklar \n ve Ecenur Karakaya tarafından geliştirilmiştir.').pack(side = LEFT)

im = PIL.Image.open("icon/icons.png")
photo = PIL.ImageTk.PhotoImage(im)
label = Label(frame_sol_alt, image=photo)
label.image = photo
label.pack(pady=30)

etiket = Label(frame_ust, bg=bg_color2, fg=tx_color, text= "İşaret Dili Algılama", font = "Verdana 14 bold")
etiket.pack(padx=10, pady=10)

run_button = Button(frame_sol_alt, bg=bg_color, fg=tx_color, text="Programı Başlat", command = openCam)
run_button.pack(pady= 40, side = BOTTOM)

window.mainloop()


