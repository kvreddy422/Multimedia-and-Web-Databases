# import necessary packages
from tensorD.factorization.env import Environment
from tensorD.dataproc.provider import Provider
from tensorD.factorization.cp import CP_ALS
import tensorD.demo.DataGenerator as dg
import tensorflow as tf
import numpy as np
import sys
import time
from sklearn.cluster import KMeans
import io

def getTerms(inp):
    path ='desctxt/devset_textTermsPer'+ inp+'.txt'
    a = open(path).readlines()
    entity = []
    entity_terms = []
    vocab = []
    s = time.time()
    for i in range(len(a)):
        splits = a[i].split(' ')
        entity.append(splits[0])
        terms = []
        for j in range(len(splits)):
            if splits[j].startswith('\"'):
                terms.append(splits[j])
                if inp == 'nothing' and splits[j] not in vocab:
                    vocab.append(splits[j])
        entity_terms.append(terms)
    e = time.time()
    return vocab, entity, entity_terms


def main():
    orig_stdout = sys.stdout
    timestr = time.strftime("%Y%m%d-%H%M%S")
    f = io.open("file"+ timestr +" "+ str(sys.argv[1])+ ".txt", mode='w',encoding="utf-8")
    sys.stdout = f
    # sess = tf.Session(config=tf.ConfigProto(inter_op_parallelism_threads=1,intra_op_parallelism_threads=1))
    data = np.load("tensor.npy")
    locDict={}
    ImageDict={}
    UserDict={}
    # k latent semantics
    k=int(sys.argv[1])
    data_provider = Provider()
    data_provider.full_tensor = lambda: data
    env = Environment(data_provider, summary_path='/tmp/cp_demo_' + '30')
    cp = CP_ALS(env)
    # set rank=10 for decomposition
    args = CP_ALS.CP_Args(rank=k, validation_internal=1)
    # build decomposition model with arguments
    cp.build_model(args)
    # train decomposition model, set the max iteration as 100
    cp.train(10)
    vocab, loc, loc_terms = getTerms('POI.wFolderNames')
    _, images, image_terms = getTerms('Image')
    _, users, user_terms = getTerms('User')
    # obtain factor matrices from trained model
    factor_matrices = cp.factors
    # Iterating to get Factor matrices and applying K-means on each of these
    for matrix in factor_matrices:
        # Applying K means
        kmeans = KMeans(n_clusters=k, random_state=0).fit(matrix)
        lenMatrix=len(matrix)
        # Finding groupings for locations
        if(lenMatrix==len(loc)):
            for i in range(len(kmeans.labels_)):
                if kmeans.labels_[i] in locDict.keys():
                    locDict[kmeans.labels_[i]].append(loc[i])
                else:
                    locDict[kmeans.labels_[i]]=[loc[i]]
            print("*********************************Groupings For Locations**************************")
            l=1
            for key,v in locDict.items():
                print("Group ",l)
                print(v)
                l+=1
            locDict={}
            with open("/home/mwd/mwd-phase2-divya/tensorD/task7fM.txt","a+") as f:
                f.write( "***************** Locations Factor Matrix***************\n")
                np.savetxt(f,matrix,newline="\n")
        # Finding grouping for Users
        elif(lenMatrix==len(users)):
            for i in range(len(kmeans.labels_)):
                if kmeans.labels_[i] in locDict.keys():
                    locDict[kmeans.labels_[i]].append(users[i])
                else:
                    locDict[kmeans.labels_[i]]=[users[i]]
            print("*********************************Groupings For Users**************************")
            l=1
            for key,v in locDict.items():
                print("Group ",l)
                print(v)
                l+=1
            locDict={}
            with open("/home/mwd/mwd-phase2-divya/tensorD/task7fM.txt","a+") as f:
                f.write( "***************** Users Factor Matrix***************\n")
                np.savetxt(f,matrix,newline="\n")
        # Finding grouping for Images
        else:
            for i in range(len(kmeans.labels_)):
                if kmeans.labels_[i] in locDict.keys():
                    locDict[kmeans.labels_[i]].append(images[i])
                else:
                    locDict[kmeans.labels_[i]]=[images[i]]
            
            
            print("*********************************Groupings For Images**************************")
            l=1
            for key,v in locDict.items():
                print("Group ",l)
                print(v)
                l+=1
            locDict={}
            
            with open("/home/mwd/mwd-phase2-divya/tensorD/task7fM.txt","a+") as f:
                f.write( "***************** Images Factor Matrix***************\n")
                np.savetxt(f,matrix,newline="\n")
    # obtain scaling vector from trained model
    lambdas = cp.lambdas

if __name__ == "__main__":
    main()
    # print(lambdas)



