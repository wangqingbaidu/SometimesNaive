import sys 
max_l = 0 
with open(sys.argv[1]) as f:
  for l in f.readlines():
    items = l.strip().split()
    if len(items[1].decode('utf8')) > max_l:
      max_l = len(items[1].decode('utf8'))
print max_l    
#print items[0].split('_')[0], items[1]
