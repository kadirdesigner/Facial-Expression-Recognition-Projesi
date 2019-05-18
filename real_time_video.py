from keras.preprocessing.image import img_to_array
import imutils
import cv2
from keras.models import load_model
import numpy as np

detection_model_path = 'haarcascade_files/haarcascade_frontalface_default.xml' #yüz tespiti için haar cascade xmli
emotion_model_path = 'models/_new_model.73-0.63.hdf5' #modelimiz


face_detection = cv2.CascadeClassifier(detection_model_path) # xml yükleniyor.
emotion_classifier = load_model(emotion_model_path, compile=False) #model duygu sınıflandırıcı olarak yükleniyor.
EMOTIONS = ["kizgin", "igrenme", "korkmus", "mutlu", "uzgun", "saskin", "notr"] #duygularımız

#Kamera başlatılıyor
cv2.namedWindow('Duygu Tespiti')
camera = cv2.VideoCapture(0)
cnt = -1
maxBuffer = 24
M = np.ndarray([maxBuffer, len(EMOTIONS)], dtype=float)
while True:
    frame = camera.read()[1]
    #karenin okunması
    frame = imutils.resize(frame,width=400)
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_detection.detectMultiScale(gray,scaleFactor=1.1,minNeighbors=5,minSize=(30,30),flags=cv2.CASCADE_SCALE_IMAGE) #yüz tespit ediliyor.

    canvas = np.zeros((250, 300, 3), dtype="uint8") # sıfır matrisi oluşturuluyor
    frameClone = frame.copy()
    if len(faces) > 0:
        face = sorted(faces, reverse=True,key=lambda x: (x[2] - x[0]) * (x[3] - x[1]))[0]

        (fX, fY, fW, fH) = face
            #Siyah beyaz yüzden ilgi noktaları çıkarılır,48 x 48 boyutlarına dönüştürülür,
        #İlgi noktası CNN ile sınıflandırma için hazırlanır.
        roi = gray[fY:fY + fH, fX:fX + fW]
        roi = cv2.resize(roi, (48, 48))
        roi = roi.astype("float") / 255.0
        roi = img_to_array(roi)
        roi = np.expand_dims(roi, axis=0)
        preds = emotion_classifier.predict(roi)[0]

        if cnt == maxBuffer -1:
            M = np.roll(M, -7)
        else:
             cnt = cnt + 1

        # write preds to M(cnt,:) = preds
        M[cnt,:]=preds


        # avg_emo = mean(M(0:cnt,:))
        ort_duygu = np.mean(M[0:cnt,:],axis=0)
        ort_duygu[np.isnan(ort_duygu)] = 0
        preds = ort_duygu
        #preds2=ort_duygu
        emotion_probability = np.max(preds)
        label = EMOTIONS[preds.argmax()]
    else: continue


    for (i, (emotion, prob)) in enumerate(zip(EMOTIONS, preds)):

                text = "{}: {:.2f}%".format(emotion, prob * 100) #duygular ve olasılıkları
                w = int(prob * 300)
                cv2.rectangle(canvas, (7, (i * 35) + 5),
                (w, (i * 35) + 35), (0, 0, 255), -1)
                cv2.putText(canvas, text, (10, (i * 35) + 23),
                cv2.FONT_HERSHEY_SIMPLEX, 0.45,
                (255, 255, 255), 2)
                cv2.putText(frameClone, label, (fX, fY - 10),
                cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 0, 255), 2)
                cv2.rectangle(frameClone, (fX, fY), (fX + fW, fY + fH),
                              (0, 0, 255), 2)

    cv2.imshow('Kamera', frameClone)
    cv2.imshow("Yuzdeler", canvas)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

camera.release()
cv2.destroyAllWindows()
