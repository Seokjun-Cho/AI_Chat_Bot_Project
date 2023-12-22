
import yaml
import glob
import json


Outputpath = "C:\\Users\\Chris\\Desktop\\chatterbot-corpus-master\\"
a = dict()
a["intent"] = []
for path in (glob.glob("C:\\Users\\Chris\\Desktop\\chatterbot-corpus-master\\chatterbot-corpus-master\\chatterbot_corpus\\data\\english\\*")):
	print(path)
	with open(path , 'r') as f:
		b = dict()
		b["pattern"] = set()
		b["response"] = set()
		b["tag"] = []

		road = yaml.load(f, Loader=yaml.FullLoader)

		for loop in road["categories"]:
			b["tag"].append(loop)
		x = 0
		for loop in road["conversations"]:
			for lines in loop:
				if x%2==0:
					b["pattern"].add(lines)
				else:
					b["response"].add(lines)
				x = x + 1	
		b["pattern"] = 	list(b["pattern"])
		b["response"] = list(b["response"])		
		a["intent"].append(b)

output = json.dumps(a)

f = open(Outputpath + "file.json", "w")
f.write(output)
f.close()

path = "C:\\Users\\Chris\\Desktop\\chatterbot-corpus-master\\file.json"
with open(path) as f:
	d = json.load(f)