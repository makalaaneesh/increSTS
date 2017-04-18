from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
import json
import re

def get_sentiment_score(text):
	analyzer = SentimentIntensityAnalyzer()
	jsonscore = analyzer.polarity_scores(text)
	return jsonscore["compound"]

def get_emoticon_score(sentence,emoticonlist):
	emoticons = re.findall(u'[\uD800-\uDBFF][\uDC00-\uDFFF]', sentence.decode('utf-8'))
	final_score = 0.0
	if len(emoticons)>0:
		for code in emoticons:
			try:
				tags = emoticonlist[code]
				tags = tags.replace(" ", "")
				tags = tags.split("|")
				score = 0.0
				for t in tags:
					score = score + get_sentiment_score(t)
				score = score/len(tags)
				final_score = final_score + score
			except:
				score = 0
		final_score = final_score / len(emoticons)
		return final_score
	else:
		return final_score



# print get_sentiment_score("omg her dress is so amazing")