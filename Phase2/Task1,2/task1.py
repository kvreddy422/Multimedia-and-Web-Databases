# -*- coding: utf-8 -*-
"""
Created on Fri Oct  5 16:46:08 2018

@author: Omnipotent
"""
import numpy as np
import pandas as pd
from scipy import linalg
import scipy
import sys
import io
import concurrent.futures
#import gensim.corpora as corpora
import time
from sklearn.decomposition import PCA
import multiprocessing
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import LatentDirichletAllocation
from scipy.sparse.linalg import svds
import pickle
import xml.etree.ElementTree
#from gensim.test.utils import common_corpus
#import gensim
try:
    import lda
except:
    pass

pickle_fname = ""

def getloc_id_mapping():
    e = xml.etree.ElementTree.parse('devset_topics.xml').getroot()
    moc1 = e.findall('topic')
    location_title = []
    location_id = []
    for moc in moc1:
        for node in moc.getiterator():
            if node.tag=='title':
                location_title.append(node.text)
            if node.tag=='number':
                location_id.append(node.text)
    return dict(zip(location_id,location_title))

def getLocId(loc):
    loc_map = getloc_id_mapping()
    for key in loc_map.keys():
        if loc_map[key] == loc:
            return key
def dowork(line):
    index = 3
    #if location:
    #   line = line[line.find("\"")-1:]
    line_split = line.split(' ')
    df1 = pd.DataFrame()
    try:
        for i in range(1,len(line_split)-4)[::4]:
            df1[line_split[i]] = pd.Series(float(line_split[i+index]))
    except Exception:
        print("something went wrong")
    return df1

def concatDfs(df):
    global fulldf
    fulldf = pd.concat(fulldf,df)
def getMatrix(file):
    if "devset_textTermsPerPOI.wFolderNames" in file:
        loc_map = getloc_id_mapping()
        content = open(file,encoding="utf8").read()
        for loc in loc_map.keys():
            content = content.replace(loc_map[loc].replace('_',' '),'').replace('  ',' ')
    else:
        content = open(file,encoding="utf8").read()
    lines = content.split('\n')
    with concurrent.futures.ProcessPoolExecutor(max_workers = multiprocessing.cpu_count()) as executor:
        r = executor.map(dowork,lines)
    val = (tuple(r))
    bigdf = pd.concat(val)
    return bigdf.fillna(0)

def getLSemantics(mat,method,k):
    term_weights = []
    global pickle_fname
    if method == "pca":
        pca = PCA(n_components=k)
        pca.fit(mat)
        sorted_indexes = []
        for pc in pca.components_:
            sorted_indexes.append(np.argsort(pc)[::-1])
        count = 0
        for sorted_index in sorted_indexes:
            print("----------------"+"pc"+str(count+1)+"-----------------")
            for index in sorted_index[:10]:
                print(str(list(mat.columns.values)[index])+" ===> "+str(pca.components_[count][index]))
            print("------------------------------------")
            count+=1
        pcl_f = open("pickles/"+pickle_fname+".pkl",'wb')
        comp = np.array(pca.components_).transpose()
        pickle.dump(comp,pcl_f)
        pcl_f.close()
        return comp
    elif method == "svd":
        #u, s, v = np.linalg.svd(mat)
        u, s, v = svds(mat, k)
        #u,s,v = linalg.svd(mat,lapack_driver='gesvd')
        sorted_indexes = []
        for sv in v[:k]:
            sorted_indexes.append(np.argsort(sv)[::-1]) #for some reason it is reverse
        count = 0
        for sorted_index in sorted_indexes:
            print("----------------"+"sv"+str(count+1)+"-----------------")
            for index in sorted_index[:10]:
                print(str(list(mat.columns.values)[index])+" ===> "+str(v[count][index]))
            print("------------------------------------")
            count+=1
        pcl_f = open("pickles/"+pickle_fname+".pkl",'wb')
        pickle.dump(v,pcl_f)
        pcl_f.close()
        return v
    elif method =="lda":
        vocab = list(mat.columns.values)
        model = lda.LDA(n_topics=k, n_iter=100, random_state=1)
        model.fit(np.array(np.array(mat)*100,dtype=int))# model.fit_transform(X) is also available
        topic_word = model.topic_word_  # model.components_ also works
        count = 0
        for i, topic_dist in enumerate(topic_word):
            srtd = np.argsort(topic_dist)[::-1]
            print("----------------"+"Topic-"+str(count+1)+"-----------------")
            for index in srtd[:10]:
                print(np.array(vocab)[index]+" ===> "+str(topic_dist[index]))
            print("------------------------------------")
            count += 1
            #topic_words = np.array(vocab)[srtd][(n_top_words+1)::1]
            #print('Topic {}: {}'.format(i, ' '.join(topic_words)))
        pcl_f = open("pickles/"+pickle_fname+".pkl",'wb')
        comp = (model.components_).transpose()
        pickle.dump(comp,pcl_f)
        pcl_f.close()
        return comp
if __name__ == "__main__":
    timestr = time.strftime("%Y%m%d-%H%M%S")
    start = time.time()
    vspace = str(sys.argv[1]).lower()
    method = str(sys.argv[2]).lower()
    k = int(sys.argv[3])
    orig_stdout = sys.stdout
    pickle_fname = vspace+"-"+method+"-"+str(k)+"-"+timestr
    f = io.open("outputs/t1-"+pickle_fname+".log", mode='w',encoding="utf-8")
    sys.stdout = f
    filename = ""
    if vspace =="user":
        filename = "devset_textTermsPerUser.txt"
    elif vspace == "image":
        filename = "devset_textTermsPerImage.txt"
    elif vspace == "location":
        filename = "devset_textTermsPerPOI.wFolderNames.txt"
    try:
        mat = pd.read_pickle("pickles/"+filename[:len(filename)-4]+'_idf.pickle')
    except FileNotFoundError:
        mat = getMatrix("C:/Users/Omnipotent/Google Drive/US docs/Mission-MS/Acadamic/Fall 2018/MWD - CSE515/Project/phase1/devset/desctxt/"+filename)
        mat.to_csv("csv/"+filename[:len(filename)-4]+"_idf.csv", sep=',', encoding='utf-8',header=None,index=False)
        mat.to_pickle("pickles/"+filename[:len(filename)-4]+"_idf"+'.pickle')
    print("Total execution time till reading dataframe "+str(time.time()-start))
    if 'all' in method:
        l1 = getLSemantics(mat,'pca',k)
        l2 = getLSemantics(mat,'svd',k)
        l3 = getLSemantics(mat,'lda',k)
        with open('pickles/'+pickle_fname+".pkl", 'wb') as f:
            pickle.dump(([l1,l2,l3]),f)
    else:
        output = getLSemantics(mat,method,k)
    #print(output)
    #term_weights = getLSemantics(mat,method,k)
    #if term_weights:
    #   for term_weight in term_weights:
    #       print(term_weight)
    #sp_mat = scipy.sparse.csr_matrix(mat.values)
    #plt.plot(np.cumsum(pca.explained_variance_ratio_))
    #plt.xlabel('PCA components')
    #plt.ylabel('Cumulative Variance')
    #plt.show()
    print("Total execution time "+str(round(time.time()-start,2))+" sec")
    sys.stdout = orig_stdout
    f.close()
