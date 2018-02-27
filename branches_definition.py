import sys, json
with open(sys.argv[1]) as f:
	labels = [l.strip() for l in f.readlines()]
	num_classes = len(labels)
	index = {}
	anno = {}
	for idx, label in enumerate(labels):
		index[str(idx)] = idx
		anno[str(idx)] = label

	json_output = {'num_classes':num_classes, 'index':index, 'anno':anno}

with open(sys.argv[2], 'w') as f:
	f.write(json.dumps(json_output, ensure_ascii=False, indent=True))
