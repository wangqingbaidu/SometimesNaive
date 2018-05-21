import os, sys
images = [os.path.join(sys.argv[1], i) for i in os.listdir(sys.argv[1]) if i.endswith('.jpg')]
with open(sys.argv[2]) as f:
	photoid = set([i.strip() for i in f.readlines()])

for i in images:
	pid = os.path.basename(i).split('.')[0]
	if pid not in photoid:
		print ('rm %s' %i)
		#os.system('rm %s' %i)
