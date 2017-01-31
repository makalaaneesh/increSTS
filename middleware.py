import json

THRESHOLD = 0.6

f = open("finaldictionary.json","r")
x = f.read()
d = json.loads(x)
newd = {}
# print d
for key, value in d.items():
	if len(value)>0:
		if value[0][1] > THRESHOLD:
			newd[key] = value[0][0]

for key, value in newd.items():
	print key, value
f.close()
ff = open("cleanedcomments.txt","r")
fff = open("ultracleanedcomments.txt","w")
xx = ff.read()
xx = xx.split("\n")

for line in xx:
	line = line.split(" ")
	newline = []
	for word in line:
		if word in newd:
			oldword = word
			nword = newd[word]
			newline.append(nword)
			print "Replaced",oldword,"with", nword
		else:
			newline.append(word)
	newline = " ".join(newline).strip()
	fff.write(newline)
	fff.write("\n")

ff.close()
fff.close()

