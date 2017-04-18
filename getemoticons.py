import requests
from bs4 import BeautifulSoup

r = requests.get("http://unicode.org/emoji/charts/full-emoji-list.html")
target = open("emoticonlist","w")
target.truncate()
if r.status_code == 200:
	c = r.content
	soup = BeautifulSoup(c)
	tables = soup.find_all("table")
	for t in tables:
		rows = t.find_all("tr")
		for i in range(3,len(rows)+1):
			cols = rows[i].find_all("td")
			cols = [ele.text for ele in cols]
			print cols
			if len(cols)>0:
				line_to_write = str(cols[2].encode("utf8")) + ">" + str(cols[len(cols)-1].encode("utf8")) + "\n"
				target.write(line_to_write)

