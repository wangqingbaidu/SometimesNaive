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
parser.add_argument('-dir', default='l:\\clean', help='Directory of file.')
args = parser.parse_args()

if __name__ == '__main__':
#     print os.path.isdir(args.dir)
    if os.path.exists(args.dir) and os.path.isdir(args.dir):
        for f in os.listdir(args.dir):
            if (os.path.isdir(os.path.join(args.dir,f))):
                try:
                    items = re.split("\(|\)", f)
                    bw_comp=items[1]
                    bw_num = items[3]
                    bw_name = items[4]
                    new_name = bw_num + '-' + bw_name
                    if len(os.listdir(os.path.join(args.dir, f))) == 2:
                        for sub_file in os.listdir(os.path.join(args.dir, f)):
                            file_extension = os.path.splitext(sub_file)[1] 
                            os.rename(os.path.join(args.dir, f, sub_file), 
                                      os.path.join(args.dir, f, new_name + file_extension))
                        
                        os.rename(os.path.join(args.dir, f), 
                                  os.path.join(args.dir, new_name))
                    elif len(os.listdir(os.path.join(args.dir, f))) == 3:
                        print f
                        for sub_file in os.listdir(os.path.join(args.dir, f)):
                                os.rename(os.path.join(args.dir, f, sub_file), 
                                          os.path.join(args.dir, f, '-'.join(re.split("\(|\)", sub_file)[-2:])))
                        os.rename(os.path.join(args.dir, f), 
                                  os.path.join(args.dir, new_name))
                    else:
                        print f
                  
                except Exception,e:
                    print 'Error dealing with file',  f, e

    
