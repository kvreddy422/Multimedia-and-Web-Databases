import sys
import datetime
import xml.etree.ElementTree
import numpy
import sklearn.preprocessing
from sklearn.decomposition import PCA
from lda import LDA

allLocDict = {}

# A Dictionary to keep track of the Models
no_of_cols = {'CM': 9, 'CM3x3': 81, 'CN': 11, 'CN3x3': 99, 'CSD': 64, 'GLRLM': 44,
              'GLRLM3x3': 396, 'HOG': 81, 'LBP': 16, 'LBP3x3': 144}

# This function will run in the beginning to create a dictionary
# of Location ID vs Location name
def getAllLoc():
    e = xml.etree.ElementTree.parse('devset_topics.xml').getroot()
    for topic in e.findall('topic'):
        allLocDict[topic.find('number').text] = topic.find('title').text

# Getting the Visual Descriptor file in a numpy Array
def getMatrix(fl):
    flPath = r'./descvis/img/' + fl
    return numpy.genfromtxt(flPath, delimiter=',')[:, 1:]

def main():
    # Input Location ID
    inplocID = str(sys.argv[1])
    # Visual Descriptor
    measure = str(sys.argv[2])
    # Number of latent Semantics Required
    k = int(sys.argv[3])
    # SVD / PCA / LDA
    method = str(sys.argv[4])

    start = datetime.datetime.now()

    getAllLoc()
    inpLocfile = allLocDict[inplocID] + " " + measure + ".csv"
    mat_to_np = getMatrix(inpLocfile)

    allLocTransformDict = {}
    feature_to_LS = []
    min_max_scaler = sklearn.preprocessing.MinMaxScaler()

    LS_to_feature = []

    # Decomposing the Input Location [Object X Feature] Matrix based on the given Input
    if method == "PCA":
        mat_to_np = min_max_scaler.fit_transform(mat_to_np)
        pca = PCA(n_components=k)
        pca.fit(mat_to_np)
        LS_to_feature = pca.components_
        feature_to_LS = numpy.array(pca.components_).transpose()
    elif method == "SVD":
        u, s, v = numpy.linalg.svd(mat_to_np)
        feature_to_LS = v[:k].transpose()
        LS_to_feature = v[:k]
    elif method == "LDA":
        # Normalizing the Input Array and then coverting it into an Integer Matrix
        # in order to utilize the LDA Decomposition
        mat_to_np = min_max_scaler.fit_transform(mat_to_np)
        mat_to_np = numpy.array(mat_to_np) * 100
        mat_to_np = numpy.array(mat_to_np, dtype=int)
        lda = LDA(n_topics=k, n_iter=200, random_state=1).fit(mat_to_np)
        feature_to_LS = (lda.components_).transpose()
        LS_to_feature = lda.components_
    else:
        print("Invalid Input")

    # Printing the Latent Semantics and the contribution of features from
    # the Visual Descriptor being used
    for rw in range(0, len(LS_to_feature)):
        abs_arry = numpy.absolute(LS_to_feature[rw])
        srtd_arry = abs_arry.argsort()[::-1]
        print("Top contributors for Latent Semantics #"+ str(rw + 1))
        for index in range(0, 9):
            print("Original Feature: ", str(srtd_arry[index]), " -> Weight: ", str(LS_to_feature[rw][srtd_arry[index]]))
        print("----------------------------------------------------------------------------")

    # Transforming the Location [Img X Feature] to [Img X Latent Semantics] with
    # help of the [feature X Latent Semantics] from the decomposition step
    for index in allLocDict:
        tempMat = getMatrix(allLocDict[index] + " " + measure + ".csv")
        allLocTransformDict[index] = min_max_scaler.fit_transform(numpy.matmul(tempMat, feature_to_LS))
        # allLocTransformDict[index] = numpy.matmul(tempMat, feature_to_LS)


    # Comparing Each Image in the Input location with the images
    # available in the current location being compared.
    distances = {}
    for index in allLocTransformDict:
        total = 0;
        for inp_row in allLocTransformDict[inplocID]:
            for curr_row in allLocTransformDict[index]:
                total += numpy.linalg.norm(inp_row - curr_row)
        distances[index] = (total / float((len(allLocTransformDict[inplocID]) * len(allLocTransformDict[index]))))

    srted_dis = sorted(distances.items(), key=lambda kv: kv[1])[:5]

    print("The Top 5 Locations similar to ", allLocDict[inplocID])
    for tpl in srted_dis:
        print(allLocDict[tpl[0]] + " - " + str(tpl[1]))

    end = datetime.datetime.now()
    print("---------------------------------------------------------------------------------")
    print("TOTAL TIME - " + str(end - start))


if __name__ == "__main__":
    main()