#!/usr/bin/python
# -*- coding: UTF-8 -*- 
'''
Authorized  by vlon Jang
Created on Dec 4, 2017
All Rights Reserved.
'''
import os, sys, re, json

FACE_HOME = '/root/face_reco/face_retrival/'
TEXT_HOME = '/root/face_reco/TextDetection/'

def parse_template(templates_file):
	# Reading templates
	templates = set()
	if os.path.exists(templates_file) and os.path.isfile(templates_file):
		with open(templates_file) as f:
			for l in f.readlines():
				templates.add(l.strip().decode('utf8'))
	else:
		print '%s not exist.' %templates_file
		exit()
	return templates
	
def post_process(results, templates, save_to):
	# Change to main dir.
	os.chdir(TEXT_HOME)
	json_output = {}
	match_results = []
	for r in results:
	    items = r.strip().split('\t')
	    image = os.path.split(items[0])[-1]
	    if len(items) > 1:
		images_splits = re.split('_|\.', image)
	        bbox = [int(imb) for imb in images_splits[-5:-1]]
	        seconds = max(int(images_splits[-7]) - 2, 0)
	        if re.split('_.', image)[0] not in json_output:
		    json_output[re.split('_.', image)[0]] = []
		for temp in templates:
		    mt = [temp]
		    if len(temp) >= 3:
                        mt += [temp[:2], temp[-2:]]
		    for t in mt:
			if t in items[1].decode('utf8'):
			    json_output[re.split('_.', image)[0]].append(
				{"names": items[1], 
				"begin":"%.2d:%.2d:%.2d" %(seconds / 3600, (seconds / 60) % 60, seconds % 60),
				"end":"%.2d:%.2d:%.2d" %((seconds+1) / 3600, ((seconds+1) / 60) % 60, (seconds+1) % 60)})
			    break
	# Save template results.
	"""
	for r in json_output:
	    for temp in templates:
		mt = [temp]
		if len(temp) >= 3:
		    mt += [temp[:2], temp[-2:]]
		for t in mt:
		    if t in json_output[r]['names'].decode('utf8'):
			match_results.append({"file":r, 'results':json_output[r]})
			break
	"""
	json_str = json.dumps(json_output, ensure_ascii=False, indent=True, sort_keys=True)
	json_temp = json.dumps([{"file":j, "results": json_output[j]} for j in json_output], 
			ensure_ascii=False, indent=True)
	#print json_str
	# print json_temp
	with open(save_to, 'w') as f:
	    f.write(json_temp)
	    
def main_func(templates_file, save_to, all_images=FACE_HOME + 'all_image'):
	# Get targets.
	templates = parse_template(templates_file)
	if not os.path.exists(save_to):
		os.makedirs(save_to)
	# Remove the last run middle results.
	os.system('rm %s' %(os.path.join(TEXT_HOME, 'jpg/*.jpg')))
	# Change to OCR detection dir.
	os.chdir(os.path.join(TEXT_HOME, 'iie-text'))
	print 'python demo.py --gpu 0 --model models/Release.lyf20161223.caffemodel --dir %s' %all_images
	os.system('python demo.py --gpu 0 --model models/Release.lyf20161223.caffemodel --dir %s' %all_images)
	# Change to OCR recognition dir.
	print 'Recognizing text...'
	os.chdir(os.path.join(TEXT_HOME, 'text_reco'))
	print 'th demo.lua %s' %os.path.join(TEXT_HOME, 'jpg')
	results = os.popen('th demo.lua %s' %os.path.join(TEXT_HOME, 'jpg')).readlines()
	# Post process for results.
	post_process(results, templates, os.path.join(save_to, 'ocr_results.json'))

if __name__ == "__main__":
	main_func(templates_file=sys.argv[1], save_to=sys.argv[2])

