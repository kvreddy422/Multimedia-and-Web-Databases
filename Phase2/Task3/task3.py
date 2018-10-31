'''
Author: Gaurav Mittal
MWDB Phase 2 Project - Task 3
ASU ID: 1213185595
'''

import xml.etree.ElementTree as ET
import sys
import csv
import math
from scipy import spatial
from collections import OrderedDict
from datetime import datetime
from collections import OrderedDict
from scipy import linalg
import numpy as np
from sklearn.decomposition import PCA
import lda

DEBUG = 0
NeededModels = [sys.argv[1]]
givenK = int(sys.argv[2])
givenImage = sys.argv[3]
MEDIAN = 0
#decompositions = [sys.argv[4]]

L2 = "L2"
Intersection = "Intersection"

distanceMetric = {'CM': L2,
				  'CM3x3': L2,
				  'CN': Intersection,
				  'CN3x3': Intersection,
				  'CSD': Intersection,
				  'GLRLM': L2,
				  'GLRLM3x3': L2,
				  'HOG': Intersection,
				  'LBP': Intersection,
				  'LBP3x3': L2}

'''
 	Parse the devset_topics.xml file

 	@return xmlDict: Dictionary with (LocationID, LocationName) pairs
'''
def parseXML():
	xmlFile = "devset_topics.xml"

	xmlDict = {}
	LocToIdDict = {}
	tree = ET.parse(xmlFile)
	root = tree.getroot()

	for a in root:
		xmlDict[int(a[0].text)] = a[1].text

	for a in xmlDict:
		LocToIdDict[xmlDict[a]] = a

	return xmlDict, LocToIdDict

'''
	Return the csv file's address for the location and the model.

	@parameter locationname: Name of the location
	@return locationAddress: String with address
'''
def fileAddress(locationname, model):
	locationAddress = "descvis/img/" + locationname + " " + model + ".csv"
	return locationAddress

'''
 	Return distance of the vectors based on DISTANCEMETRIC

 	@parameter m: Vector 1
 	@parameter t: Vector 2
 	@return returnDistance: Float distance between the two vectors
'''
def getDistance(vec1, vec2):
	DISTANCEMETRIC = L2 #distanceMetric[NeededModels[0]]
	l1 = 0
	l2 = 0
	intersectionMin = 0
	intersectionMax = 0
	for i in range(len(vec1)):
		temp1 = vec1[i]
		temp2 = vec2[i]
		l1 = l1 + abs(temp1-temp2)
		l2 = l2 + (abs(temp1-temp2) * abs(temp1-temp2))
		intersectionMin = intersectionMin + min(temp1, temp2)
		intersectionMax = intersectionMax + max(temp1, temp2)

	l2 = math.sqrt(l2)

	cosDistance = 1 - spatial.distance.cosine(vec1, vec2)

	returnDistance = 0
	if DISTANCEMETRIC == "L1":
		returnDistance = l1
	elif DISTANCEMETRIC == L2:
		returnDistance = l2
	elif DISTANCEMETRIC == "COS":
		returnDistance = cosDistance
	elif DISTANCEMETRIC == Intersection:
		returnDistance = float(intersectionMin)/float(intersectionMax)

	return returnDistance


'''
	Parses the Data from all files into a useable ordered dictionary

	@parameter locationDict: Dictionary of LocationId-LocationName
	@parameter modelArray: Array with all the models
						     - #BUGHACK - keep GLRLM3x3 as the last element in the array
	@return imageModelDict: Ordered Dictionary of data and locations
							| - imageId
							  | - "data" (Dictionary with the parsed data)
							  	| - "CM" (Array with CM Data normalized)
							  	| - "CN" (Array with CN Data normalized)
							  	| - "CM3x3" (Array with CM3x3 Data normalized)
							  	| - "CN3x3" (Array with CN3x3 Data normalized)
							  	...
							  	...
							  	...
							  | - "location" (Array of all locations the imageId is available)
							  | - "index" (Index on the orderedDict)
	@return indexArray: Array of imageId indexed w.r.t their order in the imageModelDict
							 - #NOTE - MetaData (used internally)

	@return locationIndex: Dictionary of Location and their bounds
'''
def getMatrix(locationDict, modelArray):
	imageModelDict = OrderedDict()
	index = 0
	indexArray = []
	locationIndex = {}
	prevLoc = ""
	for location in locationDict:
		locationIndex[location] = {}
		locationIndex[location]["start"] = index
		locationIndex[location]["additional"] = []
		
		if index != 0:
			locationIndex[prevLoc]["end"] = index - 1
		
		prevLoc = location

		dataDict = {}
		loc = locationDict[location]
		for model in modelArray:
			locationFileAddress = fileAddress(loc, model)
			openFile = open(locationFileAddress, "rb")
			locMatrix = list(csv.reader(openFile))
			for i in range(len(locMatrix)):
				imageId = ""
				tempArray = []
				for j in range(len(locMatrix[0])):
					if j == 0:
						imageId = str(locMatrix[i][j])
					else:
						tempArray.append(float(locMatrix[i][j]))
				minVal = float(min(tempArray))
				maxVal = float(max(tempArray))

				if imageId not in imageModelDict:
					imageModelDict[imageId] = {}
					imageModelDict[imageId]["location"] = [loc]
					imageModelDict[imageId]["data"] = {}
					imageModelDict[imageId]["index"] = index
					indexArray.append(imageId)
					index = index + 1

				if model not in imageModelDict[imageId]["data"]:
					imageModelDict[imageId]["data"][model] = []   

				if imageId in imageModelDict:
					if loc in imageModelDict[imageId]["location"]:
						for x in tempArray:
							imageModelDict[imageId]["data"][model].append(((x - minVal) / (maxVal - minVal)))
				
				if model == "GLRLM3x3":
					if loc not in imageModelDict[imageId]["location"]:
						imageModelDict[imageId]["location"].append(loc)

			
			openFile.close()
	locationIndex[prevLoc]["end"] = index - 1
	return imageModelDict, indexArray, locationIndex


def locationDistance(imageVector, LocationMatrix):
	imgVector = np.array(imageVector)
	locMatrix = np.array(LocationMatrix)
	
	if MEDIAN:
		locMatrix = np.median(locMatrix, axis=0)
		return getDistance(imgVector, locMatrix)

	distanceArray = []
	for i in range(len(LocationMatrix)):
		distanceArray.append(getDistance(imgVector, locMatrix[i]))

	return np.mean(distanceArray)


if __name__ == "__main__":
	if DEBUG:
		startTime = datetime.now()
		print "Start Time: ",
		print startTime

	xmlDict, LocToIdDict = parseXML()
	
	visualModels = ["CM",
					"CN",
					"CSD",
					"HOG",
					"LBP",
					"CM3x3",
					"CN3x3",
					"GLRLM",
					"LBP3x3",
					"GLRLM3x3"]

	noOfCols = {'CM': 9,
				'CM3x3': 81,
				'CN': 11,
				'CN3x3': 99,
				'CSD': 64,
				'GLRLM': 44,
				'GLRLM3x3': 396,
				'HOG': 81,
				'LBP': 16,
				'LBP3x3': 144}

	imageModelDict, indexArray, locationIndex = getMatrix(xmlDict, visualModels)

	DataMatrix = []
	for k in imageModelDict:
		if len(imageModelDict[k]["location"]) != 1:
			for ij in range(len(imageModelDict[k]["location"])):
				additionalLocation = imageModelDict[k]["location"][ij]
				additionalLocationID = LocToIdDict[additionalLocation]
				additionalLocationStart = locationIndex[additionalLocationID]["start"]
				additionalLocationEnd = locationIndex[additionalLocationID]["end"]
				imageIndex = imageModelDict[k]["index"]
				if not (imageIndex <= additionalLocationEnd and imageIndex >= additionalLocationEnd):
					locationIndex[additionalLocationID]["additional"].append(imageIndex)
		semiMatrix = []
		for model in NeededModels:
			for x in imageModelDict[k]["data"][model]:
				semiMatrix.append(x)
		DataMatrix.append(semiMatrix)
#
#	for i in locationIndex:
#		if len(locationIndex[i]["additional"]) != 0:
#			print xmlDict[i] + ":"
#			for j in range(len(locationIndex[i]["additional"])):
#				print "\t" + indexArray[locationIndex[i]["additional"][j]]

	decompositions = ["SVD", "PCA", "LDA"]

	for decomposition in decompositions:
		print "\nFor " + decomposition + " decomposition"



		if decomposition == "SVD":
			dm = np.array(DataMatrix)
			U, s, Vh = linalg.svd(dm, full_matrices = False)

			print "\tPART 1: " + str(givenK) + " Latent Semantic (in terms of 10 most strongest features):"
			for i in range(int(givenK)):
				print "\t\tLatent Semantic " + str(i+1) + ":"
				modArray = []
				for lol in Vh[i]:
					modArray.append(abs(lol))
				argSorted = np.argsort(modArray)
				if len(Vh[i].tolist()) > 10:
					argSorted = argSorted[0:10]
				argSorted[::-1]

				for j in range(len(argSorted)):
					sortedIndex = argSorted[j]
					print "\t\t\t" + str(Vh[i][argSorted[j]]) + " (" + NeededModels[0] + " - Feature Number: " + str(noOfCols[NeededModels[0]] - sortedIndex) + ")"
					#for ll in visualModels:
					#	tempIndex = tempIndex + noOfCols[ll] - 1
					#	if sortedIndex <= tempIndex:
					#		print "\t\t\t" + str(Vh[i][argSorted[j]]) + " (" + ll + " - Feature Number: " + str(noOfCols[ll] - (tempIndex - sortedIndex)) + ")"
					#		break
				print ""

			latentFeatureMatrix = U.tolist()
			for i in range(len(latentFeatureMatrix)):
				latentFeatureMatrix[i] = latentFeatureMatrix[i][0:givenK]

			print "\n\tPART 2: Finding 5 closest images to " + givenImage + " (" + imageModelDict[givenImage]["location"][0] + ")" + " on model " + NeededModels[0] + ":"
			ourIndex = imageModelDict[givenImage]["index"]
			distanceDict = {}
			for index in range(len(latentFeatureMatrix)):
				distanceDict[indexArray[index]] = getDistance(latentFeatureMatrix[index], latentFeatureMatrix[ourIndex])

			srtd = sorted(distanceDict.items(), key=lambda kk: kk[1])[0:5]
			for i in srtd:
				print "\t\t" + str(i[0]) + " (" + imageModelDict[i[0]]["location"][0] + ") " + " : " + str(i[1])
		
			LocationDistance = {}
			
			print "\n\tPART 3: " + " 5 closest Location to " + givenImage + " (" + imageModelDict[givenImage]["location"][0] + ")"
			for location in xmlDict:
				startIndex = locationIndex[location]["start"]
				endIndex = locationIndex[location]["end"]
				imageVector = latentFeatureMatrix[ourIndex]
				locationData = latentFeatureMatrix[startIndex:endIndex+1]
				for additionalImageIndex in locationIndex[location]["additional"]:
					locationData.append(latentFeatureMatrix[additionalImageIndex])
				LocationDistance[location] = locationDistance(imageVector, locationData)

			locationSorted = sorted(LocationDistance.items(), key=lambda kk: kk[1])[0:5]
			for i in locationSorted:
				print "\t\t" + str(xmlDict[i[0]]) + " : " + str(i[1])



		if decomposition == "PCA":
			dm = np.array(DataMatrix)
			pca = PCA(n_components=givenK)
			pca.fit(dm)
			pca.transform(dm)
			val = pca.components_
			latentMatrix = np.dot(dm, val.transpose())
			
			print "\tPART 1: " + str(givenK) + " Latent Semantic (in terms of 10 most strongest features):"
			for i in range(int(givenK)):
				print "\t\tLatent Semantic " + str(i+1) + ":"
				modArray = []
				for lol in val[i]:
					modArray.append(abs(lol))
				argSorted = np.argsort(modArray)
				if len(val[i].tolist()) > 10:
					argSorted = argSorted[0:10]
				argSorted[::-1]

				for j in range(len(argSorted)):
					sortedIndex = argSorted[j]
					print "\t\t\t" + str(val[i][argSorted[j]]) + " (" + NeededModels[0] + " - Feature Number: " + str(noOfCols[NeededModels[0]] - sortedIndex) + ")"
					#tempIndex = 0
					#for ll in visualModels:
					#	tempIndex = tempIndex + noOfCols[ll] - 1
					#	if sortedIndex <= tempIndex:
					#		print "\t\t\t" + str(val[i][argSorted[j]]) + " (" + ll + " - Feature Number: " + str(noOfCols[ll] - (tempIndex - sortedIndex)) + ")"
					#		break
				print ""

			print "\n\tPART 2: Finding 5 closest images to " + givenImage + " (" + imageModelDict[givenImage]["location"][0] + ") " + " on model " + NeededModels[0] + ":"
			ourIndex = imageModelDict[givenImage]["index"]
			distanceDict = {}
			for index in range(len(latentMatrix)):
				distanceDict[indexArray[index]] = getDistance(latentMatrix[index], latentMatrix[ourIndex])
			
			srtd = sorted(distanceDict.items(), key=lambda kk: kk[1])[0:5]
			for i in srtd:
				print "\t\t" + str(i[0]) + " (" + imageModelDict[i[0]]["location"][0] + ") " + ": " + str(i[1])

			LocationDistance = {}
			
			print "\n\tPART 3: " + " 5 closest Location to " + givenImage + " (" + imageModelDict[givenImage]["location"][0] + ")"
			for location in xmlDict:
				startIndex = locationIndex[location]["start"]
				endIndex = locationIndex[location]["end"]
				imageVector = latentMatrix[ourIndex]
				locationData = latentMatrix[startIndex:endIndex+1]
				for additionalImageIndex in locationIndex[location]["additional"]:
					np.append(locationData, [latentMatrix[additionalImageIndex]], axis = 0)
				LocationDistance[location] = locationDistance(imageVector, locationData)

			locationSorted = sorted(LocationDistance.items(), key=lambda kk: kk[1])[0:5]
			for i in locationSorted:
				print "\t\t" + str(xmlDict[i[0]]) + " : " + str(i[1])



		if decomposition == "LDA":
			dm = np.array(DataMatrix)
			lda = lda.LDA(n_topics=givenK, n_iter=200, random_state=1)
			lda.fit(np.array(dm, dtype=int))
			com = lda.components_
			latentMatrix = np.dot(dm, com.transpose())
			
			print "\tPART 1: " + str(givenK) + " Latent Semantic (in terms of 10 most strongest features):"
			for i in range(int(givenK)):
				print "\t\tLatent Semantic " + str(i+1) + ":"
				modArray = []
				for lol in com[i]:
					modArray.append(abs(lol))
				argSorted = np.argsort(modArray)
				if len(com[i].tolist()) > 10:
					argSorted = argSorted[0:10]
				argSorted[::-1]

				for j in range(len(argSorted)):
					sortedIndex = argSorted[j]
					print "\t\t\t" + str(com[i][argSorted[j]]) + " (" + NeededModels[0] + " - Feature Number: " + str(noOfCols[NeededModels[0]] - sortedIndex) + ")"
					#tempIndex = 0
					#for ll in visualModels:
					#	tempIndex = tempIndex + noOfCols[ll] - 1
					#	if sortedIndex <= tempIndex:
					#		print "\t\t\t" + str(com[i][argSorted[j]]) + " (" + ll + " - Feature Number: " + str(noOfCols[ll] - (tempIndex - sortedIndex)) + ")"
					#		break
				print ""

			print "\n\tPART 2: Finding 5 closest images to " + givenImage + " (" + imageModelDict[givenImage]["location"][0] + ") " + " on model " + NeededModels[0] + ":"
			ourIndex = imageModelDict[givenImage]["index"]
			distanceDict = {}
			for index in range(len(latentMatrix)):
				distanceDict[indexArray[index]] = getDistance(latentMatrix[index], latentMatrix[ourIndex])
			
			srtd = sorted(distanceDict.items(), key=lambda kk: kk[1])[0:5]
			for i in srtd:
				print "\t\t" + str(i[0]) + " (" + imageModelDict[i[0]]["location"][0] + ") " + ": " + str(i[1])

			print "\n\tPART 3: " + " 5 closest Location to " + givenImage + " (" + imageModelDict[givenImage]["location"][0] + ")"
			for location in xmlDict:
				startIndex = locationIndex[location]["start"]
				endIndex = locationIndex[location]["end"]
				imageVector = latentMatrix[ourIndex]
				locationData = latentMatrix[startIndex:endIndex+1]
				for additionalImageIndex in locationIndex[location]["additional"]:
					np.append(locationData, [latentMatrix[additionalImageIndex]], axis = 0)
				LocationDistance[location] = locationDistance(imageVector, locationData)

			locationSorted = sorted(LocationDistance.items(), key=lambda kk: kk[1])[0:5]
			for i in locationSorted:
				print "\t\t" + str(xmlDict[i[0]]) + " : " + str(i[1])

	

	if DEBUG:
		print "\n"
		endTime = datetime.now()
		print "End Time: ",
		print endTime

		print "Total Time: ",
		print endTime - startTime
