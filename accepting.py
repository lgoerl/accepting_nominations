import pandas as pd
import os
import numpy as np 
from numpy import linalg
import HTMLParser
import matplotlib.pyplot as plt
import matplotlib as mpl

html_parser = HTMLParser.HTMLParser()


url = 'http://www.plosone.org/article/fetchSingleRepresentation.action?uri=info:doi/10.1371/journal.pone.0026752.s001'
labmt = pd.read_csv(url,sep='\t', index_col=0)

average = labmt.happiness_average.mean()
happiness = (labmt.happiness_average - average).to_dict()
 
def score(words):
    return sum([happiness.get(word.lower(), 0.0) for word in words]) / len(words)
 

def clean(transcript):
	x = html_parser.unescape(transcript.decode("utf8")).replace(u"\u2026",u" ").encode('ascii','ignore').replace('\n',' ')
	return re.sub('[^a-z ]+','',x.lower()).split( )


def sentSeries(l,k,txt):	
	wordlist = clean(txt)
	overlap = (len(wordlist)-1-k)/(l-1)
	temp = [[i*overlap,i*overlap+k] for i in range(l)]
	if temp[-1][1]!=len(wordlist):temp[-1][1]=len(wordlist)
	srs = [score(wordlist[temp[i][0]:temp[i][1]]) for i in range(l)]
	return srs



data = 'directory for speech transcripts'
corpus = os.listdir(data)
text = {}
for f in corpus:
	ff = os.path.join(data, f)
	with open(ff) as doc:
		text[f] = doc.read()

A = [sentSeries(20,440,text[key]) for key in text.keys()]
U,s,modes = np.linalg.svd(np.array(A),full_matrices=True)
modeCoefficients = U*s
singularValues=s

x=np.linspace(0.05,1,20)

plt.figure(1)

cmap = mpl.cm.autumn
plt.subplot(211)
for i,y in enumerate(A):
	plt.plot(x,np.array(y),color=cmap(i/float(len(A))))

cmap = mpl.cm.gnuplot
plt.subplot(212)
for i,y in enumerate(modes[[0,1,2,3,4],:]):
	plt.plot(x,y,color=cmap(i/float(5)))

plt.show()


