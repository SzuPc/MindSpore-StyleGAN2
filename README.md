## 基于MindSpore的StyleGAN2的代码复现

论文地址（paperwithcode）：https://paperswithcode.com/paper/analyzing-and-improving-the-image-quality-of

我们主要是使用脸部数据集作为训练，因为官方的数据集比较大，有80G，所以这里只提供了其中一个子集大概1G

数据集地址：链接：https://pan.baidu.com/s/1Io0uTO9y9Bf5g3ITrwWN0g 
提取码：7lqh 

使用的是框架版本：MindSpore2.0 推荐使用启智社区平台进行训练和推理

数据集存放的格式：以zip的格式存放到代码目录下面的dataset文件夹即可

![image](https://github.com/SzuPc/MindSpore-StyleGAN2/assets/100685842/cbeaf997-df38-429b-853a-b39409dfb491)




然后进入到src的文件夹里进行代码的训练和推理：

训练：

```
python train.py --data_dir=../dataset/ffhq.zip --batch_size=1 --start_over=True --xflips=True --out_dir=./output_ffhq
```

注意batch_size 只能取1，如果取其他值需要在model/block.py的574行修改tensor的大小（因为我改了batch_size报错了一整天，但是感觉会有点麻烦）并且创建输出权重的文件夹（src/output_ffhq）

因为从头训练时间非常的长（至少要20h，本来数据集少而且batch_size=1），所以我这里提供了预训练的权重，可以继续进行训练或者进行推理
链接：https://pan.baidu.com/s/1ZaHvDCujayStYxKVQdhtrw 
提取码：q851 


继续训练：

```
python train.py --data_dir=../dataset/ffhq.zip --batch_size=1 --resume_train=./output_ffhq/stylegan2-ffhq-config-f --xflips=True --out_dir=./output_ffhq

```

训练完成后，我们的权重文件会生成在src/output_ffhq的文件夹里面

然后我们可以进行推理
```
python infer.py --seed=66,1518,389,230 --ckpt=./ckpt/ffhq/network-snapshot-000000-G_ema.ckpt --img_res=1024 --truncation_psi=1

```

seed是随机数种子，你可以更换不同的随机数种子以获得不同的图片

然后会出现warming在本地服务器训练的时候会出现CANN报错的情况，估计是npu-HBM爆满了或者是CANN版本的问题，在训练的过程中第一轮时间比较慢不要紧，等多一会就好了

代码参考：https://github.com/yangyucheng000/StyleGAN-V2
