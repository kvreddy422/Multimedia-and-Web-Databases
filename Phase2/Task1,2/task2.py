# -*- coding: utf-8 -*-
"""
Created on Fri Oct  5 16:46:08 2018

@author: Omnipotent
"""
import task1
import sys
import time
import pandas as pd
import numpy as np
import io
import pickle
def getDiff(r1,r2):
    return np.linalg.norm(r1-r2)
def make_it_proper(df1):
    f = open('list_of_features.txt','r',encoding="utf8")
    df = pd.DataFrame(columns=f.read().splitlines())
    for v in df.columns.values:
        try:
            df[v] = df1[v]
        except:
            pass
    return df


def getMatrix(argv):
    pfilename = ""
    k = None
    ip = []
    method = None
    print("Using the latent semantic from the file")
    filename = str(argv[1])
    entity = str(argv[2])
    ip = filename.split('-')
    k = ip[2]
    if ip[0] =="user":
        pfilename = "devset_textTermsPerUser"
    elif ip[0] == "image":
        pfilename = "devset_textTermsPerImage"
    elif ip[0] == "location":
        pfilename = "devset_textTermsPerPOI.wFolderNames"
        loc_map = task1.getloc_id_mapping()
        entity = loc_map[entity]
    user_mat = np.array(pd.read_pickle("pickles/"+"devset_textTermsPerUser"+'_idf.pickle'))
    image_mat =(np.array(pd.read_pickle("pickles/"+"devset_textTermsPerImage"+'_idf.pickle')).transpose()[1:]).transpose()
    #df2 = pd.read_pickle("pickles/"+"devset_textTermsPerImage"+'_idf.pickle')
    location_mat = np.array(pd.read_pickle("pickles/"+"devset_textTermsPerPOI.wFolderNames"+'_idf.pickle'))
    #filehandler = open("image_pickle.pkl","wb")
    #pickle.dump(image_mat,filehandler)
    #with open('list_of_features.txt','r',encoding="utf8") as f:
        #x = pd.DataFrame([df,df1],columns=list(str(f.read().split(','))))
#    df2 = pd.read_pickle("pickles/"+"devset_textTermsPerImage"+'_idf.pickle')
#    df3 = pd.read_pickle("pickles/"+"devset_textTermsPerPOI.wFolderNames"+'_idf.pickle')
#    df = pd.concat([df3,df1.iloc[0].reset_index(drop=True),df2.iloc[0].reset_index(drop=True)],axis=1)
    #print(df)
    #loc_mat = np.array(df)
    mat = [user_mat,image_mat,location_mat]
    return mat,entity,method,k,pfilename,ip,filename
def getDiffMat(pfilename,ip,lsMat,mat,entity):
    if(ip[0] == "user"):
        fname = "devset_textTermsPerUser"
    elif(ip[0]=="image"):
        fname = "devset_textTermsPerImage"
    elif(ip[0]=="location"):
        fname = "devset_textTermsPerPOI.wFolderNames"
    lines_me = open("C:/Users/Omnipotent/Google Drive/US docs/Mission-MS/Acadamic/Fall 2018/MWD - CSE515/Project/phase1/devset/desctxt/"+fname+".txt",encoding="utf8").read().split('\n')
    lines1 = open("C:/Users/Omnipotent/Google Drive/US docs/Mission-MS/Acadamic/Fall 2018/MWD - CSE515/Project/phase1/devset/desctxt/"+"devset_textTermsPerUser"+".txt",encoding="utf8").read().split('\n')
    lines2 = open("C:/Users/Omnipotent/Google Drive/US docs/Mission-MS/Acadamic/Fall 2018/MWD - CSE515/Project/phase1/devset/desctxt/"+"devset_textTermsPerImage"+".txt",encoding="utf8").read().split('\n')
    lines3 = open("C:/Users/Omnipotent/Google Drive/US docs/Mission-MS/Acadamic/Fall 2018/MWD - CSE515/Project/phase1/devset/desctxt/"+"devset_textTermsPerPOI.wFolderNames"+".txt",encoding="utf8").read().split('\n')
    lines = [lines1,lines2,lines3]
    user_map = {}
    user_map["user"] = 0
    user_map["image"] = 1
    user_map["location"] = 2
    count=0
    for line in lines_me:
        if entity in line:
            break
        count+=1
    diffMats = []
    results = []
    if "svd" in ip[1]:
        object_k = np.matmul(lsMat,mat[user_map[ip[0]]].transpose()).transpose()[count]
    else:
        if(mat[user_map[ip[0]]].shape[1] == lsMat.shape[0]):
            object_k = np.matmul(mat[user_map[ip[0]]],lsMat)[count]
        else:
            object_k = np.matmul(lsMat,mat[user_map[ip[0]]])[count]
    if ip[1] == "pca":
        for m in mat:
            ma = m
            if(ma.shape[1] == lsMat.shape[0]):
                result = np.matmul(np.array(ma),np.array(lsMat))
            else:
                result = np.matmul(lsMat,ma)
            results.append(result)
        count = 0
        for result in results:
            diffMat = []
            for res in result:
                diffMat.append(getDiff(object_k,res))
            diffMats.append(tuple((diffMat,lines[count])))
            count+=1
    elif ip[1] == "svd":
        results = []
        diffMats = []
        for m in mat:
            results.append(np.matmul(m,np.array(lsMat).transpose()))
        count = 0
        for result in results:
            diffMat = []
            for res in result:
                diffMat.append(getDiff(object_k,res))
            diffMats.append(tuple((diffMat,lines[count])))
            count+=1
    elif ip[1] == "lda":
        results = []
        diffMats = []
        for m in mat:
            results.append(np.matmul(m,lsMat))
        count = 0
        for result in results:
            diffMat = []
            for res in result:
                diffMat.append(getDiff(object_k,res))
            diffMats.append(tuple((diffMat,lines[count])))
            count+=1
#        object_k = result[count]
#        for res in result:
#            diffMat.append(getDiff(object_k,res))
    return diffMats,lines

def print_results(diffMat,lines,ip,method=None):
    srtd = np.argsort(diffMat)
    count = 1
    print("-------------start---------------------")
    if ip[0] == "location":
        for index in srtd[:6]:
            print(str(count)+")"+str(lines[index].split()[0])+"(id:"+(task1.getLocId(str(lines[index].split()[0])))+")"+"===>"+str(diffMat[index]))
            count+=1
    else:
        for index in srtd[:6]:
            print(str(count)+")"+str(lines[index].split()[0])+"===>"+str(diffMat[index]))
            count+=1
    print("-------------end----------------------")


if __name__ == "__main__":
    start = time.time()
    timestr = time.strftime("%Y%m%d-%H%M%S")
    mat,entity,method,k,pfilename,ip,filename = getMatrix(sys.argv)
    f = io.open("outputs/t2-"+ip[0]+"-"+ip[1]+"-"+entity+"-"+timestr+".log", mode='w',encoding="utf-8")
    sys.stdout = f
    lsMat = None
    if True:
        l1,l2,l3 = (None,)*3
        with open('pickles/'+str(sys.argv[1]), 'rb') as f:
            big_l = pickle.load(f)
            l1 = np.array(big_l)
        #ip[1] = "pca"

        diffMats,lines = getDiffMat(pfilename,ip,l1,mat,entity)
        for diffMat in diffMats:
            print_results(diffMat[0],diffMat[1],ip)
        #ip[1] = "svd"
        #diffMat,lines = getDiffMat(pfilename,ip,l2,mat,entity)
        #print_results(diffMat,lines,ip)
        #ip[1] = "lda"
        #diffMat,lines = getDiffMat(pfilename,ip,l3,mat,entity)
        #print_results(diffMat,lines,ip)
print("-------------------------------------------------------")
print("Total execution time "+str(round(time.time()-start,2))+" sec")