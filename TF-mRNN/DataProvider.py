# -*- coding: UTF-8 -*- 
'''
Authorized  by Vlon Jang
Created on Mar 2, 2017
Blog: www.wangqingbaidu.cn
Email: wangqingbaidu@gmail.com
From Institute of Computing Technology
©2015-2017 All Rights Reserved.
'''
import tensorflow as tf
import os, traceback, logging
import numpy as np
    
logger = logging.getLogger('TfDataProvider')
logging.basicConfig(
    format="[%(asctime)s - %(filename)s:line %(lineno)4s] %(message)s",
    datefmt='%d %b %H:%M:%S')
logger.setLevel(logging.INFO)    

class DataProvider:
    def __init__(self,
                 data_pattern,
                 max_seq_len,
                 bos_ind,
                 batch_size = 64,
                 vf_size = 1024,
                 af_size = 128,
                 fill_up_to = 4096):
        assert data_pattern and max_seq_len and bos_ind, "data_pattern and max_seq_len can't be None or 0"
        assert fill_up_to >= batch_size, "fill_up_to %d must >= batch_size %d" %(fill_up_to, batch_size)
        
        self.data_pattern = data_pattern
        self.max_seq_len = max_seq_len
        self.bos_ind = bos_ind
        
        self.batch_size = batch_size
        self.vf_size = vf_size
        self.fill_up_to = fill_up_to
        
        # Empty data queue and initialize records list, set epoch_done = False
        self._init_provider()
    
    def _init_provider(self):        
        self.video_id_list = []
        self.labels_list = []
        self.visual_features_list = []
        self.audio_features_list =[]    
        self.epoch_done = False
        
        self.record_files = []
        if os.path.exists(self.data_pattern):
            self.record_files = os.listdir(self.data_pattern)
        else:
            print ("Data pattern %s doesn't exits" %self.data_pattern)
            exit(0)
        
    def get_batch_data(self):
        if self.epoch_done:
            return None
        
        if len(self.video_id_list) < self.batch_size:
            while len(self.record_files) and len(self.video_id_list) < self.fill_up_to:
                filename = self.record_files.pop(0)
                self._parser_video_record(os.path.join(self.data_pattern, filename))
        
        if len(self.video_id_list) <= self.batch_size and len(self.record_files) == 0:
            self.epoch_done = True
            
        ret_vid = self.video_id_list[:self.batch_size]
        ret_labels = self.labels_list[:self.batch_size]
        ret_vf = self.visual_features_list[:self.batch_size]
        ret_af = self.audio_features_list[:self.batch_size]
        
        self.video_id_list = self.video_id_list[self.batch_size:]
        self.labels_list = self.labels_list[self.batch_size:]
        self.visual_features_list = self.visual_features_list[self.batch_size:]
        self.audio_features_list = self.audio_features_list[self.batch_size:]
        
        return self._convert_video_record_to_ndarray(ret_vid, ret_labels, ret_vf, ret_af)
    
    def _convert_video_record_to_ndarray(self, ret_vid, ret_labels, ret_vf, ret_af):     
        x = np.zeros([self.batch_size, self.max_seq_len], dtype=np.int32)
        y = np.zeros([self.batch_size, self.max_seq_len], dtype=np.int32)
        vf = np.array(ret_vf, dtype=np.float32)
        af = np.array(ret_af, dtype=np.float32)
        fg = np.zeros([self.batch_size, self.max_seq_len], dtype=np.float32)
        sl = np.zeros([self.batch_size], dtype=np.int32)
        
        x[:, 0] = self.bos_ind
        for i in range(len(ret_labels)):
            # Fill labels to X, y
            if len(ret_labels[i]) > self.max_seq_len - 1:
                x[i, 1:] = ret_labels[i][:self.max_seq_len-1]
                y[i, :self.max_seq_len-1] = ret_labels[i][:self.max_seq_len-1]
                y[i, self.max_seq_len-1] = self.bos_ind
                fg[i, :] = np.ones([self.max_seq_len], dtype=np.float32)
                sl[i] = self.max_seq_len
            else:
                l = len(ret_labels[i])
                x[i, 1:l+1] = ret_labels[i]
                y[i, :l] = ret_labels[i]
                y[i, l] = self.bos_ind
                fg[i, :l+1] = np.ones([l+1], dtype=np.float32)
                sl[i] = l + 1
        return (x, y, vf, af, fg, sl, ret_vid)
    
    def _parser_video_record(self, filename):
        if os.path.exists(filename):
            rec_count = 0
            try:
                for serialized_example in tf.python_io.tf_record_iterator(filename):
                    example = tf.train.Example()
                    example.ParseFromString(serialized_example)
                  
                    video_id = example.features.feature['video_id'].bytes_list.value
                    labels = example.features.feature['labels'].int64_list.value
                    visual_features = example.features.feature['mean_rgb'].float_list.value
                    audio_features = example.features.feature['mean_audio'].float_list.value
                    
                    self.video_id_list.append(video_id)
                    self.labels_list.append(labels)
                    self.visual_features_list.append(visual_features)
                    self.audio_features_list.append(audio_features)
                    
                    rec_count += 1
                    
                logger.info("File %s contains %d records" %(filename, rec_count))
            except:
                traceback.print_exc()
        else:
            print ('File %s not exits' %filename)
        
        
        
if __name__ == '__main__':
    import time
    dp = DataProvider(data_pattern='../data', max_seq_len=10, bos_ind=1, fill_up_to=64)
    while not dp.epoch_done:
        x, y, vf, af, fg, sl, ret_vid = dp.get_batch_data()
        print(x)
#         time.sleep(3)
        
        
        
