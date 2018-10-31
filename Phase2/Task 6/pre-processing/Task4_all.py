import glob
import pandas as pd
import xml.etree.ElementTree as ET
abc = 'CM'
cde ='../img/*'+abc+'.csv'
op = ""
z=0
doc = ET.parse("../devset_topics.xml")

topicElem = doc.getroot()
locIdNameDict={}
for topic in topicElem:
    locIdNameDict[topic[0].text]= topic[1].text.replace("_"," ")
appended_data = []
appended_data1 = []
for f in glob.glob(cde):
    data=pd.read_csv(f,encoding = "ISO-8859-1")
    appended_data.append(data)
    dataMean = data.mean()
    z=z+1
    op +=locIdNameDict[str(z)]+','
    for i in dataMean:
        op +=str(i)+","
    op = op[:-1]
    op +="\n"
    appended_data1 = pd.concat(appended_data, axis=1)
with open("../Task4_entity_id_term_metrics_loc.csv", 'w', encoding="utf8") as f:
    f.write(op)
appended_data.to_csv('../Task4_entity_id_term_metrics_loc.csv')
print(appended_data1)