import os;
import sys;
import pickle;
import librosa;
import numpy as np;
import keras.losses;
from keras.utils.generic_utils import get_custom_objects
from keras.models import Model,model_from_json;
from keras import backend as K;
from keras.layers.embeddings import Embedding;
from keras.utils.vis_utils import plot_model;
from keras.models import Sequential, load_model;
from keras.optimizers import rmsprop, adam, adagrad, SGD;
from keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau;
from keras.preprocessing.text import text_to_word_sequence, one_hot, Tokenizer;
from keras.layers import Input, Dense, merge, Dropout, BatchNormalization, Activation, Conv1D, Lambda;
DIR=os.getcwd();maxlen_char=47;
with open(DIR+"/char_index","rb") as f:
    char_index=pickle.load(f);

with open(DIR+"/index_char","rb") as f:
    index_char=pickle.load(f);

input_tensor=Input(shape=(673,20));
x=Conv1D(kernel_size=1,filters=192,padding="same")(input_tensor);
x=BatchNormalization(axis=-1)(x);
x=Activation("tanh")(x);
def res_block(x,size,rate,dim=192):
    x_tanh=Conv1D(kernel_size=size,filters=dim,dilation_rate=rate,padding="same")(x);
    x_tanh=BatchNormalization(axis=-1)(x_tanh);
    x_tanh=Activation("tanh")(x_tanh);
    x_sigmoid=Conv1D(kernel_size=size,filters=dim,dilation_rate=rate,padding="same")(x);
    x_sigmoid=BatchNormalization(axis=-1)(x_sigmoid);
    x_sigmoid=Activation("sigmoid")(x_sigmoid);
    out=merge([x_tanh,x_sigmoid],mode="mul");
    out=Conv1D(kernel_size=1,filters=dim,padding="same")(out);
    out=BatchNormalization(axis=-1)(out);
    out=Activation("tanh")(out);
    x=merge([x,out],mode="sum");
    return x,out;

skip=[];
for i in np.arange(0,3):
    for r in [1,2,4,8,16]:
        x,s=res_block(x,size=7,rate=r);
        skip.append(s);

def ctc_lambda_function(args):
    y_true_input, logit, logit_length_input, y_true_length_input=args;
    return K.ctc_batch_cost(y_true_input,logit,logit_length_input,y_true_length_input);

skip_tensor=merge([s for s in skip],mode="sum");
logit=Conv1D(kernel_size=1,filters=192,padding="same")(skip_tensor);
logit=BatchNormalization(axis=-1)(logit);
logit=Activation("tanh")(logit);
logit=Conv1D(kernel_size=1,filters=len(char_index)+1,padding="same",activation="softmax")(logit);
base_model=Model(inputs=input_tensor,outputs=logit);
logit_length_input=Input(shape=(1,));
y_true_input=Input(shape=(maxlen_char,));
y_true_length_input=Input(shape=(1,));
loss_out=Lambda(ctc_lambda_function,output_shape=(1,),name="ctc")([y_true_input,logit,logit_length_input,y_true_length_input])
model=Model(inputs=[input_tensor,logit_length_input,y_true_input,y_true_length_input],outputs=loss_out);
model.compile(loss={'ctc': lambda y_true, y_pred: y_pred},optimizer="adam");
model.load_weights(DIR+"/listen_model.chk");

def listen(audio_path):
    wav, sr = librosa.load(audio_path, mono=True);
    b = librosa.feature.mfcc(wav, sr)
    mfcc = np.transpose(b, [1, 0]);
    input_vec=np.zeros((1,673,20));
    for i in np.arange(0,len(mfcc)):
        for j,ele in enumerate(mfcc[i]):
            input_vec[0,i,j]=ele;
    y_pred=base_model.predict(input_vec);
    ctc_decode = K.ctc_decode(y_pred,input_length=np.ones(y_pred.shape[0]) * y_pred.shape[1])[0][0]
    out = K.get_value(ctc_decode)[0];
    answer=[index_char[i] for i in out];
    final_chinese="".join(answer);
    return final_chinese;