
#CsvLogger: Batch sonuçları=> csvde loglanır. | ModelCheckpoint: Her epochtan sonra model save.| EarlyStopping: iyileştirme durduysa eğitimi tamamla.
from keras.callbacks import CSVLogger, ModelCheckpoint, EarlyStopping
#ReduceLROnPlateau: learning rate azaltımı
from keras.callbacks import ReduceLROnPlateau
#ImageDataGenerator: image preprocess classları
from keras.preprocessing.image import ImageDataGenerator
#train test ayırıcı
from sklearn.model_selection import train_test_split
#activation : relu veya softmax kullanılır: relu maxlama işlemi yapar max(x,0), softmax:• giriş görüntüsünün üretilen özelliklerini eğitim veri setine göre çeşitli sınıflara sınıflandırmak için bir softmax aktivasyon işlevi kullanır.
#Conv2d eldeki veriye filtre uygulayarak öznitelik çıkarımında bulunur. verilen parametreler filtre boyutları adeti vs.
from keras.layers import Activation, Conv2D
from keras.layers import BatchNormalization # Her batchte önceki katmanın aktivasyonlarını normalize edilir
from keras.layers import GlobalAveragePooling2D # geçici veriler pooling işlemi yapar. Özellik haritasındaki tüm öğelerin ortalamasını alarak her bir özellik haritasını skalar bir değere indirger.
from keras.models import Model #model ekleniyor
from keras.layers import Input
from keras.layers import MaxPooling2D # Conv2D ile çıkarılan özellik haritasının boyutunu yarı yarıya indirir.
from keras.layers import SeparableConv2D #SeparableConv2D, parametre sayısını azaltarak Conv2Dye göre hesaplamayı azaltır.
from keras import layers #katmanlar importlanıyor.
from keras.regularizers import l2 # iyileştirici importlanıyor
import pandas as pd
import cv2
import numpy as np
import tensorflow as tf
import keras

config = tf.ConfigProto( device_count = {'GPU': 1 , 'CPU': 56} )
sess = tf.Session(config=config)
keras.backend.set_session(sess)



dataset_path = 'fer2013/fer2013.csv' #eğitim seti iki sütun içeriyor. biri emotions(duygular):resme karşılık gelen duygu 0-6 numerik olarak verilmiş.(0-mutlu 1-üzgün vs).
image_size=(48,48)
# parameters
batch_size = 32 # batch boyutu 2nin katları şeklinde verilebilir. 32,64,128... donanın gücüne göre artırılabilir.
num_epochs = 100 # çevrilecek epoch sayısı
input_shape = (48, 48, 1) #alınan girdi verisinin 48x48 boyutlu ve gri olması
validation_split = .2
verbose = 1
num_classes = 7 # 7 duygu
patience = 50
base_path = 'models/' #model çıktı klasörü
l2_regularization=0.01


def load_fer2013():
	data = pd.read_csv(dataset_path) #panda ile csv okunur
	pixels = data['pixels'].tolist() #csvden pixels sütunu pixels'e çekilir.
	width, height = 48, 48
	faces = [] #yüzlerin datasetten çekilmesi
	for pixel_sequence in pixels:
		face = [int(pixel) for pixel in pixel_sequence.split(' ')]
		face = np.asarray(face).reshape(width, height)
		face = cv2.resize(face.astype('uint8'),image_size)
		faces.append(face.astype('float32'))
	faces = np.asarray(faces)
	faces = np.expand_dims(faces, -1) #boyut genişletme
	emotions = pd.get_dummies(data['emotion']).as_matrix()
	return faces, emotions #fonksiyon çağırıldığında yüzler ve duygular elde edilir.

def preprocess_input(x, v2=True): #girdi yüzler üzerine yapılan ön işlemler
    x = x.astype('float32')
    x = x / 255.0
    if v2:
        x = x - 0.5
        x = x * 2.0
    return x



# data generator: zoom boyut değişimleri flip vs gibi işlemler ile train sete yeni veriler katılmış olur.
data_generator = ImageDataGenerator(
                        featurewise_center=False,
                        featurewise_std_normalization=False,
                        rotation_range=10,
                        width_shift_range=0.1,
                        height_shift_range=0.1,
                        zoom_range=.1,
                        horizontal_flip=True)


regularization = l2(l2_regularization)

# base -- Conv2D katmanı uygulanır.(8: uygulanacak filtre sayısı, 3x3 boyutunda, kernel regularizasyonu Optimizasyon sırasında katman parametrelerine ceza uygulanmasını sağlar., use_bias: katman bias vektörü kullanacak mı, img_input
img_input = Input(input_shape)
x = Conv2D(8, (3, 3), strides=(1, 1), kernel_regularizer=regularization, use_bias=False)(img_input)
x = BatchNormalization()(x) #batch normalize işlemi yapılır.
x = Activation('relu')(x) # relu işlemi uygulanır
x = Conv2D(8, (3, 3), strides=(1, 1), kernel_regularizer=regularization, use_bias=False)(x) #tekrar Conv2d katmanı eklenir.
x = BatchNormalization()(x) # Batch Normalizasyon yapılır.
x = Activation('relu')(x) #relu işlemi yapılır.

# module 1
residual = Conv2D(16, (1, 1), strides=(2, 2), padding='same', use_bias=False)(x) #Conv2D işlemleri
residual = BatchNormalization()(residual) #Batch Normalizasyon
x = SeparableConv2D(16, (3, 3), padding='same', kernel_regularizer=regularization, use_bias=False)(x)
x = BatchNormalization()(x)
x = Activation('relu')(x)
x = SeparableConv2D(16, (3, 3), padding='same', kernel_regularizer=regularization, use_bias=False)(x)
x = BatchNormalization()(x)
x = MaxPooling2D((3, 3), strides=(2, 2), padding='same')(x)
x = layers.add([x, residual])

# module 2
residual = Conv2D(32, (1, 1), strides=(2, 2), padding='same', use_bias=False)(x)
residual = BatchNormalization()(residual)
x = SeparableConv2D(32, (3, 3), padding='same', kernel_regularizer=regularization, use_bias=False)(x)
x = BatchNormalization()(x)
x = Activation('relu')(x)
x = SeparableConv2D(32, (3, 3), padding='same', kernel_regularizer=regularization, use_bias=False)(x)
x = BatchNormalization()(x)
x = MaxPooling2D((3, 3), strides=(2, 2), padding='same')(x)
x = layers.add([x, residual])

# module 3
residual = Conv2D(64, (1, 1), strides=(2, 2),padding='same', use_bias=False)(x)
residual = BatchNormalization()(residual)
x = SeparableConv2D(64, (3, 3), padding='same',kernel_regularizer=regularization,use_bias=False)(x)
x = BatchNormalization()(x)
x = Activation('relu')(x)
x = SeparableConv2D(64, (3, 3), padding='same',kernel_regularizer=regularization,use_bias=False)(x)
x = BatchNormalization()(x)
x = MaxPooling2D((3, 3), strides=(2, 2), padding='same')(x)
x = layers.add([x, residual])

# module 4
residual = Conv2D(128, (1, 1), strides=(2, 2),padding='same', use_bias=False)(x)
residual = BatchNormalization()(residual)
x = SeparableConv2D(128, (3, 3), padding='same',kernel_regularizer=regularization,use_bias=False)(x)
x = BatchNormalization()(x)
x = Activation('relu')(x)
x = SeparableConv2D(128, (3, 3), padding='same',kernel_regularizer=regularization,use_bias=False)(x)
x = BatchNormalization()(x)
x = MaxPooling2D((3, 3), strides=(2, 2), padding='same')(x)
x = layers.add([x, residual])
x = Conv2D(num_classes, (3, 3), padding='same')(x)
x = GlobalAveragePooling2D()(x)
output = Activation('softmax',name='predictions')(x)

model = Model(img_input, output)
model.compile(optimizer='adam', loss='categorical_crossentropy',metrics=['accuracy'])
model.summary()

# Modelde kullanılacak callbackler oluşturulur.
log_file_path = base_path + '_emotion_training.log' #epoch döngüsü sırasında log tutulur
csv_logger = CSVLogger(log_file_path, append=False)
early_stop = EarlyStopping('val_loss', patience=patience)
reduce_lr = ReduceLROnPlateau('val_loss', factor=0.1, patience=int(patience/4), verbose=1)
trained_models_path = base_path + '_new_model' #model path
model_names = trained_models_path + '.{epoch:02d}-{val_acc:.2f}.hdf5' #_new_model.(epoch sayısı)-(doğruluk).hdf5
model_checkpoint = ModelCheckpoint(model_names, 'val_loss', verbose=1,save_best_only=True)
callbacks = [model_checkpoint, csv_logger, early_stop, reduce_lr]

# loading dataset
faces, emotions = load_fer2013()
faces = preprocess_input(faces)
num_samples, num_classes = emotions.shape
xtrain, xtest,ytrain,ytest = train_test_split(faces, emotions,test_size=0.2,shuffle=True) #train test set oluşturumu
model.fit_generator(data_generator.flow(xtrain, ytrain,
                                            batch_size),
                        steps_per_epoch=len(xtrain) / batch_size,
                        epochs=num_epochs, verbose=1, callbacks=callbacks,
                        validation_data=(xtest,ytest)) #model oluşumu
