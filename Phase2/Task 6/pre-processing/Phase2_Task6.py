import glob
import pandas as pd
import xml.etree.ElementTree as ET
abc = 'CM'
aaa=0
array_metric=['CM', 'CM3x3', 'CN', 'CN3x3','CSD','GLRLM', 'GLRLM3x3','HOG','LBP', 'LBP3x3']
op = ""
doc = ET.parse("C:/Users/Vaibhav Kalakota/PycharmProjects/MWDB-Phase1/devset_topics.xml")
# list of dataframes
dfs_list = []

topicElem = doc.getroot()
locIdNameDict={}
for topic in topicElem:
    locIdNameDict[topic[0].text]= topic[1].text.replace("_"," ")
# // Store different dataframes
fileNo=0;
for array_element in array_metric:
    cde ='C:/Users/Vaibhav Kalakota/PycharmProjects/MWDB-Phase1/img/*'+array_element+'.csv'
    z=0
    op = ""
    outFileHere = "C:/Users/Vaibhav Kalakota/PycharmProjects/MWDB-Phase1/Task5_1/Task5_"+str(fileNo)+"_entity_id_term_metrics_loc.csv"
    fileNo=fileNo+1
    for f in glob.glob(cde):
        data=pd.read_csv(f,encoding = "ISO-8859-1")
        dataMean = data.median()
        # print(dataMean)
        z=z+1
        op +=locIdNameDict[str(z)]+','
        for i in dataMean:
            op +=str(i)+","
        op = op[:-1]
        op +="\n"
    with open(outFileHere, 'w', encoding="utf8") as f:
        f.write(op)
#         dfs_1=pd.read_csv(outFileHere,encoding = "ISO-8859-1",header=None)
#         dfs_list.append(dfs1)
#         aaa=aaa+1