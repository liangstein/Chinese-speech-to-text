## Chinese-speech-to-text
Speech recognition trained by [THCHS30 open Chinese speech database](http://data.cslt.org/thchs30/standalone.html).

## Dependency
* Python3.6(numpy, scipy, pickle, h5py, librosa),
* Keras2.02,
* Tensorflow v1.1 backend, (Not tested with Theano backend)
* Cuda8.0, Cudnn6.0 (If GPU is used)

## Neural Network Implementation
The neural network used is Wavenet, which is firstly raised in [Deepmind's paper](https://arxiv.org/abs/1609.03499). The recognition is done on character level (no need to vectorizing 10000 words), therefore the dimension is much smaller than recurrent neural network. The structure is [Here](https://github.com/liangstein/Chinese-speech-to-text/blob/master/model.png).

The training dataset is small (only 10000 samples), and the ctc loss is decreased to 0.2768 after 124 epochs. The training time on a GTX 1080 is 15 hours. 

## Results
```
audio: [1.wav]
listened: 一九九山年二二十的上午务四穿声看月显安人向武村碰加工嫂五人进城都体一服
ground text:一九九三年二月二十三日上午四川省安岳县岳源乡五村彭家姑嫂五人进城购置衣服

audio: [2.wav]
listened: 看亚够考前跑后你直惊准的山却也得他王欧尧起声王的在山进回道
ground text:看羊狗跑前跑后一只惊飞的山雀惹得它汪汪汪咬几声嗡嗡嗡的在山间回荡

audio: [3.wav]
listened: 北积穿过云层易下一片鱼海又时头过喜过的运物一些可件然国冲绿的群山大底
ground text:飞机穿过云层眼下一片云海有时透过稀薄的云雾依稀可见南国葱绿的群山大地

audio: [4.wav]
listened: 王宁看被墙颠后不范云燕孙场起来及自卫不均为抓活
ground text:王英汉被枪毙后部分余孽深藏起来几次围捕均未抓获

audio: [5.wav]
listened: 其书有现人原本于穷取来观邪不凑找他了莫银杷要求看四损面看富面
ground text:其中有些人原本与陈曲澜关系不错找他软磨硬泡要求不看僧面看佛面
```

We test the recognition ability using the audio files from test set. Although the training dataset is small (10000 samples), it can recognize key words already. Right now the model isn't trained for recognitions in noisy environments. Larger and more complex training dataset can have better recognition results. 

## Authors
liangstein (lxxhlb@gmail.com, lxxhlb@mail.ustc.edu.cn)
Contact me if needed.

