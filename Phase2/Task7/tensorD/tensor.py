import csv
import datetime
import math
import pandas as pd
from scipy.spatial.distance import cosine
import numpy as np
from scipy.spatial import distance
from heapq import nsmallest
from lxml import etree
import sys
from collections import defaultdict


def euclidean_distance(X, Y):
	return math.sqrt(sum((X.get(d,0) - Y.get(d,0))**2 for d in set(X) | set(Y)))

def union2(dict1, dict2):
    return dict(list(dict1.items()) + list(dict2.items()))

def cosine_similarity(v1,v2):
    #"compute cosine similarity of v1 to v2: (v1 dot v2)/{||v1||*||v2||)"

    sumxx, sumxy, sumyy = 0, 0, 0
    for i in range(len(v1)):
        x = v1[i]; y = v2[i]
        sumxx += x*x
        sumyy += y*y
        sumxy += x*y
    # extracting 3 max terms contributing most to the similarity can't be picked in case of cosine similarity
    # as it normalizes the vector
    return sumxy/math.sqrt(sumxx*sumyy)

def cosine_dic(dic1,dic2):
    numerator = 0
    dena = 0
    for key1,val1 in dic1:
        numerator += val1*dic2.get(key1,0.0)
        dena += va1*val1
    denb = 0
    for val2 in dic2.values():
        denb += val2*val2
    return numerator/math.sqrt(dena*denb)

def main():
	#Load text file into list with CSV module
	# with open('/home/riya/Downloads/MWDB/desctxt/onewordPOI.txt', 'rt') as f:
	# 	reader = csv.reader(f, delimiter = ' ', skipinitialspace=True)
	# 	lineData = list()
	# 	cols = next(reader)

	# 	for line in reader:
	# 		if line != []:
	# 			lineData.append(line)


	# with open('/home/riya/Downloads/MWDB/desctxt/twowordsPOI.txt', 'rt') as f:
	# 	reader = csv.reader(f, delimiter = ' ', skipinitialspace=True)
	# 	lineData2 = list()
	# 	cols = next(reader)

	# 	for line in reader:
	# 		if line != []:
	# 			lineData2.append(line)

	# with open('/home/riya/Downloads/MWDB/desctxt/threewordsPOI.txt', 'rt') as f:
	# 	reader = csv.reader(f, delimiter = ' ', skipinitialspace=True)
	# 	lineData3 = list()
	# 	cols = next(reader)

	# 	for line in reader:
	# 		if line != []:
	# 			lineData3.append(line)

	# with open('/home/riya/Downloads/MWDB/desctxt/fourwordsPOI.txt', 'rt') as f:
	# 	reader = csv.reader(f, delimiter = ' ', skipinitialspace=True)
	# 	lineData4 = list()
	# 	cols = next(reader)

	# 	for line in reader:
	# 		if line != []:
	# 			lineData4.append(line)

	elemdict1 = {}
	elemdict = {}
	elemdict2 = {}
	list1=[]
	list2=[]
	list3=[]
	listUserLocImages1={}
	listUserLocImages3={}
	listUserLocImages2={}

	# loc_id = sys.argv[1]
	# model = sys.argv[2]
	# k = sys.argv[3]
	flag=0

	doc = etree.parse("/home/riya/Code mwdb/devset_topics.xml")

	topicElem = doc.getroot()
	locIdNameDict={}
	for topic in topicElem:
		locIdNameDict[topic[0].text]= topic[1].text.replace("_"," ")
	with open('/home/riya/Code mwdb/desctxt2/devset_textTermsPerPOI.wFolderNames.txt', 'rt') as f:
		reader = csv.reader(f, delimiter = ' ', skipinitialspace=False)
		lineData = list()
		cols = next(reader)
		
		for line in reader:
			loc_name=" ".join(line[0].split("_"))
			startIndex=len(line[0].split("_"))
			newIndex=startIndex*2
			elemdict[loc_name]=line[newIndex-startIndex+1:]


	with open('/home/riya/Code mwdb/desctxt2/devset_textTermsPerUser.txt', 'rt') as f:
		reader = csv.reader(f, delimiter = ' ', skipinitialspace=False)
		lineData1 = list()
		cols = next(reader)

		for line in reader:
			if line != []:
				lineData1.append(line)

	with open('/home/riya/Code mwdb/desctxt2/devset_textTermsPerImage.txt', 'rt') as f:
		reader = csv.reader(f, delimiter = ' ', skipinitialspace=False)
		lineData2 = list()
		cols = next(reader)

		for line in reader:
			if line != []:
				lineData2.append(line)

	k=0
	for key, value in elemdict.items():
		locArr=value
		i=0
		elemtermdict={}
		while(i < len(locArr)-1):
			# listUserLocImages1[locArr[i]] = float(locArr[i+3])
			list1.append(locArr[i])

			i=i+4
		elemdict[key]=elemtermdict
		listUserLocImages1[k]=list1
		k+=1

	k=0
	for i in range(len(lineData1)):
		noOfTerms = (len(lineData1[i])-2)/4
		temp = 1

		elemtermdict1={}
		for j in range(int(noOfTerms)):
			term = lineData1[i][temp].replace('"','')
			list2.append(term)
			# listUserLocImages2[term] = float(lineData1[i][temp+3])
			temp=temp+4
		# elemdict1[lineData1[i][0]]=elemtermdict1
		print(k)
		listUserLocImages2[k] = list2
		k+=1
			# elemdict[lineData[i][0]]
		
	k=0
	for i in range(len(lineData2)):
		noOfTerms = (len(lineData2[i])-2)/4
		temp = 1
		elemtermdict1={}
		for j in range(int(noOfTerms)):
			term = lineData2[i][temp].replace('"','')
			# listUserLocImages3[term] = float(lineData2[i][temp+3])
			list3.append(term)
			# listUserLocImages3[term] = lineData2[i][0]
			temp=temp+4
		listUserLocImages3[k] = list3
		elemdict2[lineData2[i][0]]=elemtermdict1
		k+=1

	# print(j)

	for k1,val1 in listUserLocImages1.items():
		for k2,val2 in listUserLocImages2.items():
			for k3,val3 in listUserLocImages3.items():
				for i in range(len(val1)):
					if(val1[i] in val2):
						if(val1[i] in val3):
							np.array([k1,k2,k2])
							print(k1)
							break





	dd = defaultdict(list)
	for d in (listUserLocImages1, listUserLocImages2, listUserLocImages3): # you can list as many input dicts as you want here
	    for key, value in d.items():
	        dd[key].append(value)

	for k,v in dd.items():
		if(len(v)==3):
			print(k,v)



	# print(dd)
main()

