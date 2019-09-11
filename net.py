
from keras.layers import Conv1D,MaxPooling1D,Dropout,Flatten,Dense,BatchNormalization,AveragePooling1D,add,Input
from keras.models import Sequential,Model,load_model
import keras
from keras import backend as K



def build_model(input_shape=(5000,8),num_classes=55):
    model = Sequential()
 
    model.add(Conv1D(16, 16,strides=2, activation='relu',padding="same",input_shape=input_shape))
    model.add(Conv1D(16, 16,strides=1, activation='relu',padding="same"))
    model.add(MaxPooling1D(2))

 
    model.add(Conv1D(64, 16,strides=2, activation='relu',padding="same"))
    model.add(Conv1D(64, 16,strides=1, activation='relu',padding="same"))
    model.add(MaxPooling1D(2))
 
    model.add(Conv1D(128, 8,strides=2, activation='relu',padding="same"))
    model.add(Conv1D(128, 8,strides=1, activation='relu',padding="same"))
    model.add(MaxPooling1D(2))
 
    model.add(Conv1D(256, 8,strides=2, activation='relu',padding="same"))
    model.add(Conv1D(256, 8,strides=1, activation='relu',padding="same"))
    model.add(MaxPooling1D(2))
    model.add(Dropout(0.5))
 
    model.add(Conv1D(512, 4,strides=2, activation='relu',padding="same"))
    model.add(Conv1D(512, 4,strides=1, activation='relu',padding="same"))
    model.add(MaxPooling1D(2))

    model.add(Flatten())
    model.add(Dropout(0.5))
    model.add(Dense(512))
    model.add(Dense(num_classes, activation='sigmoid'))
    return(model)
 
def conv_block(x,filters,kernel_size,strides,padding):
    x = Conv1D(filters,kernel_size,strides=strides,padding=padding,activation='relu')(x)
    x = BatchNormalization()(x)
    return(x)
 
 
def res_layer(x,filters,kernel_size,strides,short=False):
    x1 = x
    x2 = conv_block(x1,filters,kernel_size,strides=strides,padding="same")
    x2 = conv_block(x2,filters,kernel_size,strides=1,padding="same")
    if(short):
        x1 = AveragePooling1D(pool_size=2,strides=strides,padding="same")(x1)
        x1 = conv_block(x1,filters,kernel_size,strides=1,padding="same")
    y = add([x1,x2])
    return(y)
 
 
def build_res_model(input_shape=(5000,8),num_classes=55):
    inp = Input(shape=input_shape)
 
    x = conv_block(inp,16,16,2,padding="same")
    x = conv_block(x,16,16,2,padding="same")
    x = MaxPooling1D(pool_size=2,strides=2)(x)
    x = res_layer(x,16,8,1,short=False)
 
 
    x = res_layer(x,64,8,2,short=True)
    x = res_layer(x,64,8,2,short=True)
 
 
    x = res_layer(x,128,8,2,short=True)
    x = res_layer(x,128,8,2,short=True)
 
 
 
    x = res_layer(x,256,8,2,short=True)
    x = res_layer(x,256,8,2,short=True)
 
    x = Dropout(0.5)(x)
 
    x = res_layer(x,512,4,2,short=True)
    x = res_layer(x,512,4,2,short=True)
 
 
    x = Flatten()(x)
 
    y = Dense(num_classes,activation="sigmoid")(x)
    
    model = Model(inp,y)
 
    return(model)

def multi_acc(y_true, y_pred,t=0.5):

    return K.equal(K.mean(K.equal((y_true>=t),(y_pred>=t)),axis=-1),1)

def my_binary_crossentropy(y_true, y_pred,k=4):
    bcy = K.binary_crossentropy(y_true,y_pred)
    loss = bcy *(1+y_true * (k-1))*(k+1)/(2*k)
    losses = K.mean(loss,axis=-1)
 
    return(losses)





if __name__ == "__main__":
    pass
