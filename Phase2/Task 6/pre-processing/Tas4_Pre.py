import glob
import pandas as pd
import xml.etree.ElementTree as ET
abc = "LBP3x3"
cde ='../img/*'+abc+'.csv'
op = ""
z=0
doc = ET.parse("../devset_topics.xml")

topicElem = doc.getroot()
locIdNameDict={}
for topic in topicElem:
    locIdNameDict[topic[0].text]= topic[1].text.replace("_"," ")
for f in glob.glob(cde):
    data=pd.read_csv(f,encoding = "ISO-8859-1")
    dataMean = data.mean()
    z=z+1
    op +=locIdNameDict[str(z)]+','
    for i in dataMean:
        op +=str(i)+","
    op = op[:-1]
    op +="\n"
with open("../Task4_entity_id_term_metrics_loc_new.csv", 'w', encoding="utf8") as f:
    f.write(op)
