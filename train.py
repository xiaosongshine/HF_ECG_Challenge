from glob import glob
from tqdm import tqdm
import pandas as pd
import numpy as np
import math
from keras.models import Sequential,Model,load_model
from keras.optimizers import Adam,RMSprop
import keras
from net import build_model,build_res_model,multi_acc,my_binary_crossentropy


All_len = 24106
Test_len = 4000
Train_len = All_len - Test_len

def get_arrythmias(arrythmias_path):
    with open(arrythmias_path,"r") as f:
        data = f.readlines()
    arrythmias = [d.strip() for d in data]
    return arrythmias

def get_dict(arrythmias):
    str2ids = {}
    id2strs = {}
    for i,a in enumerate(arrythmias):
        str2ids[a] = i
        id2strs[i] = a
    return str2ids,id2strs


def get_train_label(label_path,str2ids,csv_path):

    with open(label_path,"r",encoding='UTF-8') as f:
        data = f.readlines()
    labels = [d.strip() for d in data]
    label_dicts = {}
    label_dicts["index"]=[]
    label_dicts["age"]=[]
    label_dicts["sex"]=[]
    label_dicts["one_label"]=[]
    i = 0
    for l in tqdm(labels):
        i += 1
        ls = l.split("\t")
        #print(l,len(ls))
        label_dicts["index"].append(ls[0])
        label_dicts["age"].append(ls[1])
        label_dicts["sex"].append(ls[2])
        one_label = np.zeros(len(str2ids),)
        for ls1 in ls[3:]:
            one_label[str2ids[ls1]] = 1
        
        label_dicts["one_label"].append(list(one_label))
        
    df = pd.DataFrame(label_dicts)
    df = df.sample(frac=1)
    print(df.head(5))
    df.to_csv(csv_path,index=None)

def get_test_index(label_path,str2ids,csv_path):

    with open(label_path,"r",encoding='UTF-8') as f:
        data = f.readlines()
    labels = [d.strip() for d in data]
    label_dicts = {}
    label_dicts["index"]=[]
    label_dicts["age"]=[]
    label_dicts["sex"]=[]
    label_dicts["one_label"]=[]

    for l in tqdm(labels):

        ls = l.split("\t")
        #print(l,len(ls))
        while(len(ls)<3):
            ls.append("")
        label_dicts["index"].append(ls[0])
        label_dicts["age"].append(ls[1])
        label_dicts["sex"].append(ls[2])
        one_label = np.zeros(len(str2ids),)
        for ls1 in ls[3:]:
            one_label[str2ids[ls1]] = 1
        
        label_dicts["one_label"].append(list(one_label))
        
    df = pd.DataFrame(label_dicts)
    print(df.head(5))
    df.to_csv(csv_path,index=None)

#进行归一化
def normalize(v):
    return (v - v.mean(axis=1).reshape((v.shape[0],1))) / (v.max(axis=1).reshape((v.shape[0],1)) + 2e-12)


def get_feature(wav_file,BASE_DIR,Lens=5000,train=True,pred=False):
 
    file_path = BASE_DIR+wav_file

    df = pd.read_csv(file_path,sep=" ")
 
    data = df.values[:Lens]
 
    if(not pred):
 
        if(train):
            pass
            
        feature = data
 
    else:
        feature = data
 
    return(feature)


def xs_gen(train_df,batch_size,test_len,BASE_DIR,train=True):

    data_df = pd.DataFrame()
    data_df["index"],data_df["one_label"]=train_df["index"],train_df["one_label"]
    data_list = data_df.values
    if train :
 
        img_list = data_list[test_len:]
        print("Found %s train items."%len(img_list))
        print("list 1 is",img_list[0])
        steps = math.ceil(len(img_list) / batch_size)    # 确定每轮有多少个batch
    else:
        img_list = data_list[:test_len]
        print("Found %s test items."%len(img_list))
        print("list 1 is",img_list[0])
        steps = math.ceil(len(img_list) / batch_size)    # 确定每轮有多少个batch
    while True:
        if(train):
            np.random.shuffle(img_list)
        for i in range(steps):
 
            batch_list = img_list[i * batch_size : i * batch_size + batch_size]
            np.random.shuffle(batch_list)
            batch_x = np.array([get_feature(file,BASE_DIR,train=True) for file in batch_list[:,0]])
            batch_y = np.array([np.array(y) for y in batch_list[:,1]])
 
            yield batch_x, batch_y


def train(epochs,train_iter,test_iter,BZ):
    model = build_res_model()
    print(model.summary())
    opt = Adam(0.001,decay=0.001)


    model.compile(loss="binary_crossentropy",
                optimizer=opt, metrics=['binary_accuracy',multi_acc])

    

    ckpt = keras.callbacks.ModelCheckpoint(
        filepath='ckpt/build_res_model.h5',
        monitor="val_multi_acc",mode="max",save_best_only=True,verbose=1)


    H = model.fit_generator(
        generator=train_iter,
        steps_per_epoch=math.ceil(Train_len/BZ),
        epochs=0+epochs,
        initial_epoch=0,
        validation_data = test_iter,
        validation_steps =math.ceil((Test_len)/BZ),
        callbacks=[ckpt],
    )

def get_index(y_pred,t=0.5):
    index = []
    for i,y in enumerate(y_pred):
        if(y>=t):
            index.append(i)
    if(len(index)==0):
        index.append(0)
    return(index)

def app(model_path,val_iter,BZ):
    model = load_model(model_path,custom_objects={"multi_acc":multi_acc})
    predicts = model.predict_generator(generator=val_iter,
        steps=math.ceil(8036/BZ),verbose=1)
    indexs = []
    for p in predicts:
        indexs.append(get_index(p))
    return indexs

Get_train_label = False
BZ = 32
EPS = 16
Train = False
if __name__ == "__main__":
    arrythmias = get_arrythmias("./datasets/hf_round1_arrythmia.txt")
    str2ids,id2strs = get_dict(arrythmias)
    
    if(Get_train_label):
        get_train_label("./datasets/hf_round1_label.txt",str2ids,"./datasets/hf_round1_label.csv")

    if Train:
        train_df = pd.read_csv("./datasets/hf_round1_label.csv")
        train_df["one_label"] = train_df["one_label"].map(lambda x:eval(x))

        train_iter = xs_gen(train_df,BZ,Test_len,"./datasets/train/",train=True)
        test_iter = xs_gen(train_df,BZ,Test_len,"./datasets/train/",train=False)
        train(EPS,train_iter,test_iter,BZ)
    
    else:
        get_test_index("./datasets/hf_round1_subA.txt",str2ids,"./datasets/hf_test_label.csv")
        test_df = pd.read_csv("./datasets/hf_test_label.csv")
        test_df["one_label"] = test_df["one_label"].map(lambda x:eval(x))
        val_iter = xs_gen(test_df,BZ,8036,"./datasets/testA/",train=False)
        indexs = app("ckpt/build_res_model.h5",val_iter,BZ)
        print(len(indexs))
        with open("datasets/subA.txt","w",encoding="utf-8") as f:
            for i,index in tqdm(enumerate(indexs)):
                name = test_df["index"][i]
                age = test_df["age"][i]
                sex = test_df["sex"][i]

                if math.isnan(age):
                    age = ""
                else:
                    age = int(age)
                if(type(sex) != type("str")):
                    if math.isnan(sex):
                        sex = ""
                labels = [id2strs[int(inx)] for inx in index]
                f.write(name+"\t")
                f.write(str(age)+"\t")
                f.write(str(sex)+"\t")
                for label in labels:
                    f.write(label+"\t")
                f.write("\n")





    