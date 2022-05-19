# BaitapAI
import glob
import cv2
import numpy as np

x_train = []
x_test = []
y_train = []

dem = 0
for imgpath in glob.glob('/content/drive/MyDrive/Colab Notebooks/Nhandienkhuonmat/Nhom/*.bmp'):
  n = cv2.imread(imgpath)
  # print('BanHTuan' in imgpath)
  if dem%10 == 0:
    x_test.append(n)
  else:
    x_train.append(n)
  name1 = 'Tung1'
  name2 = 'Tuyen'
  name3 = 'VTung'
  if name1 in imgpath:
    y_train.append([0])
  elif name2 in imgpath:
    y_train.append([1])
  elif name3 in imgpath:
    y_train.append([2])
  dem += 1
x_train = np.array(x_train)
x_test = np.array(x_test)
y_train = np.array(y_train)

# y_train, y_test là output (như đánh label cho ảnh x_train, x_test á)
# y_train, y_test chỉ được đánh số từ 0 cho các đối tượng (output): VD HTuan đánh số 0, Huy số 1, Tuan19 số 2

y_test = []
i = 0
while x_train.shape[0] < y_train.shape[0]:
  y_test.append(y_train[i])
  y_train = np.delete(y_train, i)
  i += 9
y_test = np.array(y_test)
y_train = list(y_train)

import matplotlib.pyplot as plt
# plt.imshow(x_train[40])
# print(y_train[40])
plt.imshow(x_test[1])
print(y_test[1])
 
import matplotlib.pyplot as plt
# plt.imshow(x_train[40])
# print(y_train[40])
plt.imshow(x_test[0])
print(y_test[0])
 
import matplotlib.pyplot as plt
# plt.imshow(x_train[40])
# print(y_train[40])
plt.imshow(x_test[2])
print(y_test[2])
 
from keras.utils.np_utils import to_categorical

# Chuyen don vi mau thanh so thuc
x_train = x_train.astype('float32')
x_test = x_test.astype('float32')

# Chuyen anh trang den
x_train /= 255
x_test /= 255

# ...to_categorical(y_train, 10) với 10: số phần tử output (ở đây dùng cifar10 nên output = 10)
y_train = to_categorical(y_train, 3)
y_test = to_categorical(y_test, 3)
from tensorflow.keras.optimizers import RMSprop
from keras.layers import Activation, BatchNormalization, Dropout, Conv2D, MaxPooling2D, Flatten
from keras.models import Sequential

model = Sequential()
model.add(Conv2D(32, (3,3), activation='relu', kernel_initializer='he_uniform', padding='same', input_shape=(250,250,3))) 
# 32: số lần dùng bộ lọc (filter - mảng 3x3)
 # kernel_initializer='he_uniform': setup bộ lọc ban đầu với dạng he_uni...
# padding='same': để khi pad mảng filter ảnh sẽ ko bị thay đổi kích thước.
model.add(MaxPooling2D((2,2)))                                                                                                                        
model = Sequential()
model.add(Conv2D(64, (3,3), activation='relu', kernel_initializer='he_uniform', padding='same'))  # 32: số lần dùng bộ lọc (filter - mảng 3x3)
 # kernel_initializer='he_uniform': setup bộ lọc ban đầu với dạng he_uni...
 # padding='same': để khi pad mảng filter ảnh sẽ ko bị thay đổi kích thước.
model.add(MaxPooling2D((2,2)))                                           
model = Sequential()
model.add(Conv2D(128, (3,3), activation='relu', kernel_initializer='he_uniform', padding='same'))  # 32: số lần dùng bộ lọc (filter - mảng 3x3)
 # kernel_initializer='he_uniform': setup bộ lọc ban đầu với dạng he_uni...
# padding='same': để khi pad mảng filter ảnh sẽ ko bị thay đổi kích thước.
model.add(MaxPooling2D((2,2))
from keras.layers import Dense
model.add(Flatten())
model.add(Dense(128, activation='relu', kernel_initializer='he_uniform'))  # Dense: full-connected (tất cả các tế bào thần kinh đều kết nối với nhau)
model.add(Dense(3, activation='softmax'))
from tensorflow.keras.optimizers import SGD
opt = SGD(lr=0.01, momentum=0.9)  # lr: learning rate: tốc độ học, momentum: momen động lượng, sự dao động :D?
model.compile(optimizer=opt, loss='categorical_crossentropy', metrics='accuracy')
history = model.fit(x_train, y_train, epochs=15, batch_size=64, validation_data=(x_test, y_test), verbose=1)
 
import matplotlib.pyplot as plt
from tensorflow.keras.utils import load_img, img_to_array

img = load_img('Tung1_0005.bmp', target_size=(250,250))  
plt.imshow(img)
 
import matplotlib.pyplot as plt
from tensorflow.keras.utils import load_img, img_to_array

img = load_img('Tuyen_0001.bmp', target_size=(250,250))  
plt.imshow(img)
 
import matplotlib.pyplot as plt
from tensorflow.keras.utils import load_img, img_to_array

img = load_img('VTung_0040.bmp', target_size=(250,250))  
plt.imshow(img)
 
img = img_to_array(img)
img = img.reshape(1,250,250,3)
img = img.astype('float32')
img /= 255
print(model.predict(img))
np.argmax(model.predict(img), axis=1)
 
score = model.evaluate(x_test, y_test, verbose=0)
print('Test loss:', score[0])
print('Test accuracy:', score[1])
 

plt.plot(history.history['accuracy'])
plt.plot(history.history['val_accuracy'])
plt.title('Model accuracy')
plt.xlabel('epoch')
plt.legend(['Train', 'Validation'], loc='upper left')
plt.show()
