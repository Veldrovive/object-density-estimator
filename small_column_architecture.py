from keras.layers import Input, Conv2D, MaxPooling2D, BatchNormalization, UpSampling2D
from keras.models import Model

input_img = Input(shape=(480, 640, 3))
x = Conv2D(24, (5, 5), activation='relu' , padding="same")(input_img)
#x = Conv2D(24, (5, 5), activation='relu' , padding="same")(x)
x = MaxPooling2D((2, 2), padding="same")(x)

x = Conv2D(48, (3, 3), activation='relu', padding="same")(x)
#x = Conv2D(48, (3, 3), activation='relu', padding="same")(x)
x = MaxPooling2D((2, 2), padding="same")(x)

x = Conv2D(24, (3, 3), activation='relu', padding="same")(x)
x = Conv2D(12, (3, 3), activation='relu', padding="same")(x)

out = Conv2D(1, (1,1), activation='relu', padding="same")(x)

# x = UpSampling2D((2, 2))(x)
# x = UpSampling2D((2, 2))(x)