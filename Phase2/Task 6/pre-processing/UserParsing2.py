lines = open("../devset_textTermsPerPOI.txt", encoding="utf8").read().split('\n')
op = ""
for line in lines:
    data = line.split(' ')
    first = True
    for index in range(1, len(data) - 3)[::4]:
        data[index] = data[index].replace(',', '')
        op += data[0] + "," + str(data[index]).replace("\"", "") + "," + str(data[index + 1]) + "," + str(
            data[index + 2]) + "," + str(data[index + 3]) + "\n"
with open("../entity_id_term_metrics.csv", 'w', encoding="utf8") as f:
    f.write(op)