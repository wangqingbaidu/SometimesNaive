# -*- coding: UTF-8 -*- 
'''
Authorized  by Vlon Jang
Created on Feb 1, 2017
Blog: www.wangqingbaidu.cn
Email: wangqingbaidu@gmail.com
From Institute of Computing Technology
©2015-2017 All Rights Reserved.
'''
import os, argparse, re
parser = argparse.ArgumentParser("Change XXX@XXX@(XXX) file name.")
parser.add_argument('-dir', default='.', help='Directory of file.')
args = parser.parse_args()

if __name__ == '__main__':
    if os.path.exists(args.dir) and os.path.isdir(args.dir):
        for f in os.listdir(args.dir):
            if (os.path.isdir(f)):
                for sub_file in os.listdir(f):
                    try:
                        os.rename(os.path.join(f, sub_file), 
                                  os.path.join(f, ' '.join(re.split("\(|\)", sub_file)[-2:])))
                    except:
                        print 'Error dealing with file',  sub_file
                
                os.rename(f, ' '.join(re.split("\(|\)", f)[-2:]))
                          
    
    
