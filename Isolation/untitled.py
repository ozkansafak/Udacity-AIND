



import codecs
import json
inputFile = open('isolation-result-30854.json')
j = inputFile.read()
a = json.dumps(j, sort_keys=True, indent=4)
print(codecs.decode(codecs.decode(codecs.decode(codecs.decode(a, 'unicode_escape'),'unicode_escape'),'unicode_escape'),'unicode_escape'))
