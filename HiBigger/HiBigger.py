# -*- coding: UTF-8 -*- 
'''
Authorized  by Vlon Jang
Created on Jan 12, 2017
Blog: www.wangqingbaidu.cn
Email: wangqingbaidu@gmail.com
From Institute of Computing Technology
©2015-2017 All Rights Reserved.
'''
import pymysql, random, subprocess
import os
import time

styles = []
class PullImage:
    def __init__(self, pull2, h5_home,
                 host='10.25.0.118', port=3306,user='root', passwd='111111', db='alimusic'):
        self.pull2 = pull2
        self.h5_home = h5_home
        self.lastID = 0;
        
        self.currentID = []
        self.remote = user + "@" + host + h5_home
        self.mysql_cn= pymysql.connect(host=host, port=port,user=user, passwd=passwd, db=db)
    
    def __del__(self):
        if self.mysql_cn:
            self.mysql_cn.close()
            
    def pull(self, num=100):
        sql = """
            SELECT id, style, path FROM Image 
            WHERE id > {lastID} and done is null
            LIMIT {num}
        """.format(lastID = self.lastID, num = num)
        result = None        
        self.currentID = []
        try:
            with self.mysql_cn.cursor() as cursor:
                cursor.execute(sql)
            result = cursor.fetchall()
            self.mysql_cn.commit()
        except Exception,e:
            print e
            
        if result:
            os.system('rm -rf %s/*' %self.pull2)
            for s in styles:
                os.system("mkdir %s" %os.path.join(self.pull2, s))
            for pid, style, path in result:
                if pid > self.lastID:
                    self.lastID = pid
                self.currentID.append(pid)
                if not style:
                    style = random.choice(styles)
                
                if subprocess.call(["scp", os.path.join(self.remote, path), os.path.join(self.pull2, style)],
                                   stdout=subprocess.PIPE,stderr=subprocess.PIPE, shell=True):
                    print "No such file or directory."
                    
        else:
            time.sleep(60)
            print 'Did not get any data!'
            
        return self.currentID
    
class StyleTransform:
    def __init__(self, pull2, output_dir):
        self.pull2 = pull2
        self.output_dir = output_dir
        
    def fast(self):
        for s in styles:
            subprocess.call(['th fast_neural_style.lua -gpu 0   -model models/eccv16/starry_night.t7 -input_dir em -output_dir ./'])
