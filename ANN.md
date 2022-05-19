# BaitapAIfrom google.colab import drive
drive.mount('/content/drive')
import glob
import cv2
import numpy as np
x_train = []
x_test = []
y_train = []
y_test = []
dem = 0
for imgpath in glob.glob('/content/drive/MyDrive/Colab Notebooks/Nhandienkhuonmat/Tung/*.bmp'):
  n = cv2.imread(imgpath)
  if dem%10 == 0:
    x_test.append(n)
    y_test.append(0)
  else:
    x_train.append(n)
    y_train.append(0)
  dem += 1
x_train = np.array(x_train)
x_test = np.array(x_test)
y_train = np.array(y_train)

y_test = np.array(y_test)
y_train = list(y_train)
print(y_train)
print(y_test)
 
import matplotlib.pyplot as plt
plt.imshow(x_test[3])
print(y_test[3])
 

print(x_train.shape)
#print(y_train.shape)
print(y_test.shape)
print(x_test.shape)
 
from keras.utils.np_utils import to_categorical

x_train = x_train.reshape(52, 187500)
x_test = x_test.reshape(6, 187500)
# Chuyen don vi mau thanh so thuc
x_train = x_train.astype('float32')
x_test = x_test.astype('float32')

# Chuyen anh trang den
x_train /= 255
x_test /= 255

# ...to_categorical(y_train, 3) với 3: số phần tử output
y_train = to_categorical(y_train) from tensorflow.keras.optimizers import RMSprop
from keras.layers import Dense, Activation, BatchNormalization, Dropout
from keras.models import Sequential
model = Sequential()
model.add(Dense(512, kernel_initializer='normal', activation='relu', input_shape=(187500,)))  # 784: số tín hiệu đầu vào
model.add(Dense(512, activation='relu'))
# model.add(Dropout(0.2))
# model.add(Dense(512, activation='relu'))
model.add(Dropout(0.2)) # Không cần Dropout cũng đc :D?
model.add(Dense(1, activation='softmax')) 
# model.summary()
model.compile(loss='categorical_crossentropy', optimizer=RMSprop(), metrics=['accuracy'])
history = model.fit(x_train, y_train, batch_size=64, epochs=20, verbose=1, validation_data=(x_test, y_test))  # verbose = 0: trong qua trinh hoc ko hien thi ra man hinh, = 1: hien thi ..., = 2: hien thi 1/500:..., 2/500:...
 


Kết quả nhận diện
import cv2
img = cv2.imread(r'Tung6.jpg')
plt.imshow(img)
 
img_re = cv2.resize(img, (250, 250))
plt.imshow(img_re)
print(img_re.shape)
 
x_test_c = img_re.reshape(1, 187500) 
x_test_c = x_test_c.astype('float32')
x_test_c /= 255
plt.imshow(x_test_c)
  
y_pred = model.predict(x_test_c)
# print(y_test)
# acc_num_class0 = y_pred[0]
# acc_num_class1 = y_pred[1]
# np.max
print(y_pred) 
