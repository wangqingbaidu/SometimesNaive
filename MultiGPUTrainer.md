# AutoMultiGPU For Tensorflow
### 1. Usage
```sh
python MultiGPUTrainer.py -h

optional arguments:
  -h, --help            show this help message and exit
  -batch_size BATCH_SIZE
                        Number of batch size.
  -num_gpus NUM_GPUS    Number of gpus to use.
  -initial_learning_rate INITIAL_LEARNING_RATE
                        Inital learning rate.
  -decay_steps DECAY_STEPS
                        Decay learning rate after decay steps.
  -learning_rate_decay_factor LEARNING_RATE_DECAY_FACTOR
                        Decay learning rate decay factor.
  -optimizer OPTIMIZER  Optimizer type.
  -iter_save ITER_SAVE  After number of iters to save params.
  -iter_start ITER_START
                        Number of iters to start.
  -iter_verbose ITER_VERBOSE
                        Print log each iter_verbose.
  -graph_def_path GRAPH_DEF_PATH
                        Path to the defination of model graph.
  -graph_name GRAPH_NAME
                        Graph name.
  -dp_def_path DP_DEF_PATH
                        Path to the defination of data provider.
  -dp_name DP_NAME      Data provider name.
  -model_dir MODEL_DIR  Path to save model.
  -model_name MODEL_NAME
                        Model name.
  -pre_trained_model_path PRE_TRAINED_MODEL_PATH
                        Pretrained model path.
            
```

使用一下几个参数，其它参数参见-h列表。

```sh
-graph_def_path ./model/AudioRNN.py 
-graph_name AudioRNNModel 
-dp_def_path ./model/AudioDataProvider.py 
-dp_name AudioDataProviderNumpy 
-num_gpus 1
```
其中`-graph_def_path`前向传播的计算图文件，`-graph_name`文件中的类的名字，通过`__init__()`函数进行计算图的初始化。`-dp_def_path `用于DataProvider的文件,该文件用于生成计算图所依赖的数据，`-dp_name `文件中的provider的名字，通过`__init__()`函数进行计算图的初始化。`-num_gpus`设置gpu的个数。

### 2. Attention！
使用该脚本并行时，计算图必须包括的属性`loss`, `grads`, `batch_size`, `reuse`, `idx_mapping`。

参数名 | 作用
---- | ---
loss| 一次前向传播的损失。
grads| variables对应的梯度。
batch_size| 不解释了。
reuse| 由于可能用多卡训练，需要在所有的`variable_scope`添加`reuse`参数。
idx_mapping| 非常重要！对应于provider返回值的顺序与model中的那个tensor相对应。

dataprovider必须包括的属性。`batch_size`, `get_batch_data`

参数名 | 作用
---- | ---
batch_size| 不解释了。
get\_batch\_data| 获取下一个batch的方法，返回与`idx_mapping`参数相对应的数据个数。

### 3. One More Thing.
为了增加灵活性，可以自行修改计算图以及DataProvider的参数。

#### 修改计算图的相关参数，在第101行。

```python
for idx, d in enumerate(devices):
    with tf.device(d):
        with tf.name_scope('tower_%d' %idx):
            print ('\tBuilding forward computing graph for device %s...' %d)    
            model = graph(batch_size=batch_size, reuse=reuse, is_multi_gpu=True, is_training=True)
            tf.add_to_collection('train_model', model)
            grads.append(model.grads)
            tf.add_to_collection('losses', model.loss)
            reuse=True
```
`model = graph(batch_size=batch_size, reuse=reuse, is_multi_gpu=True, is_training=True)`

#### 修改DataProvider的相关参数，在第166行。

```python
dp = cu.load_dp_config(args.dp_def_path, args.dp_name)(data_pattern='/home/zhangzhiwei/audio_data',
                                                       batch_size=args.batch_size)                                             
```
