import cv2
import numpy as np
import os
from matplotlib import pyplot as plt
import time
import mediapipe as mp
import PreTrain as PT
import Video as V
import Train as T

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout

from scipy import stats
from gtts import gTTS
import os
from pygame import mixer

import PIL.Image
import PIL
from tkinter import *



T.model.load_weights('action.h5')
T.model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
T.model.summary()

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
    with V.mp_holistic.Holistic(min_detection_confidence=0.5, min_tracking_confidence=0.5) as holistic:
        while cap.isOpened():

            ret, frame = cap.read()

            image, results = V.mediapipe_detection(frame, holistic)
            print(results)


            # 2. Tahmin Bölümü
            keypoints = V.extract_keypoints(results)
            sequence.append(keypoints)
            sequence = sequence[-30:]

            if len(sequence) == 30:
                res = T.model.predict(np.expand_dims(sequence, axis=0))[0]
                print(PT.actions[np.argmax(res)])
                predictions.append(np.argmax(res))

                if np.unique(predictions[-10:])[0] == np.argmax(res):
                    if res[np.argmax(res)] > threshold:
                        if len(sentence) > 0:
                            if PT.actions[np.argmax(res)] != sentence[-1] and PT.actions[np.argmax(res)] != "gulegule":
                                sentence.append(PT.actions[np.argmax(res)])
                                mixer.init()
                                mixer.music.load("ses/"+PT.actions[np.argmax(res)] + '.mp3')
                                mixer.music.play()
                        else:
                            sentence.append(PT.actions[np.argmax(res)])

                if len(sentence) > 5:
                    sentence = sentence[-5:]
            cv2.rectangle(image, (0, 0), (640, 40), (245, 117, 16), -1)
            cv2.putText(image, ' '.join(sentence), (3, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2, cv2.LINE_AA)

            cv2.imshow('OpenCV Feed', image)

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


