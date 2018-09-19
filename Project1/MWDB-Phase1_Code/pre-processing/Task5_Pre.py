import glob
import pandas as pd
import xml.etree.ElementTree as ET
abc = 'CM'
array_metric=['CM', 'CM3x3', 'CN', 'CN3x3','CSD','GLRLM', 'GLRLM3x3','HOG','LBP', 'LBP3x3']
op = ""
doc = ET.parse("../devset_topics.xml")

topicElem = doc.getroot()
locIdNameDict={}
for topic in topicElem:
    locIdNameDict[topic[0].text]= topic[1].text.replace("_"," ")
for array_element in array_metric:
    cde ='..\img/*'+array_element+'.csv'
    z=0
    for f in glob.glob(cde):
        data=pd.read_csv(f,encoding = "ISO-8859-1")
        dataMean = data.mean()
        print(dataMean)
        z=z+1
        op +=locIdNameDict[str(z)]+','
        for i in dataMean:
            op +=str(i)+","
        op = op[:-1]
        op +="\n"
with open("../Task5_entity_id_term_metrics_loc.csv", 'w', encoding="utf8") as f:
    f.write(op)
