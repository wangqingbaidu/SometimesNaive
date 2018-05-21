import sys
data = {}
with open(sys.argv[1]) as f:
	for l in f.readlines():
		photoid, flag = l.strip().split()
		flag = True if flag == 'true' else False
		data[photoid] = flag

moving_data = open(sys.argv[3], 'w')

with open(sys.argv[2]) as f:
	for l in f.readlines():
		photoid, c = l.strip().split()
		if photoid in data and data[photoid]:
			pass
		else:
			moving_data.write('%s %s\n' %(photoid, c))
			
车公庄大街9号五栋大楼A2-701
