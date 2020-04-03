import os, sys
images = [os.path.join(sys.argv[1], i) for i in os.listdir(sys.argv[1]) if i.endswith('.jpg')]
with open(sys.argv[2]) as f:
	photoid = set([i.strip() for i in f.readlines()])

for i in images:
	pid = os.path.basename(i).split('.')[0]
	if pid not in photoid:
		print ('rm %s' %i)
		#os.system('rm %s' %i)
{'features': [{
    'feat_file': os.path.join(train_data_dir, 'topic_features.txt'),
    'attach_column': [2, 1],
    'feat_length': 14,
    'feat_format': 'topic',
    'separator': '\t',
    'feat_type': float,
    'missing_value': -999
    # 'missing_value': [-999] + [0] * 13
    # 'missing_value': [-999] + [None, None] + [0] * 11,
    'attach_only_has_key': False
}]}
