import os, sys
with open(sys.argv[1]) as f:
	for l in f.readlines():
		ori_name, new_name = l.strip().split(' ')
		new_name_l = []
		for s in new_name.split(','):
			if s:
				new_name_l.append(s)
		new_name = '-'.join(new_name_l)
		print "mv %s %s" %(os.path.join(sys.argv[2], ori_name), 
			os.path.join(sys.argv[2], new_name))
		os.system("mv %s %s" %(os.path.join(sys.argv[2], ori_name), 
			os.path.join(sys.argv[2], new_name)))
