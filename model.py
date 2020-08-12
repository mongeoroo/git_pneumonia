from keras.layers import GlobalAveragePooling2D,Dense
from keras.applications  import densenet
from keras.layers import Reshape,multiply,add
from keras.models import Model

def SEblock(dim, ratio,tensor_in):
    squeeze = GlobalAveragePooling2D()(tensor_in)

    excitation = Dense(units=dim // ratio, activation='relu', kernel_initializer='he_normal')(squeeze)

    excitation = Dense(units=dim, activation='sigmoid', kernel_initializer='he_normal')(excitation)

    excitation = Reshape((1, 1, dim))(excitation)

    scale = multiply([tensor_in, excitation])
    add_ = add([tensor_in, scale])

    return add_

den = densenet.DenseNet169(weights=None,input_shape=(224,224,3),include_top=False)
den.summary()
x = SEblock(int(den.output.shape[-1]),4,den.output)
x = GlobalAveragePooling2D()(x)
x = Dense(1, activation='sigmoid', kernel_initializer='he_normal', name='fc3')(x)

model_1 = Model(inputs=den.input,outputs = x)


if __name__== '__main__':
    model_1.summary()
