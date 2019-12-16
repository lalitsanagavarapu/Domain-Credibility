import sys
import random
# print("Output from Python") 
# print("Requested URl: " + sys.argv[1]) 
# print("WebCred Score: " + str(random.randint(1,100))) 
RequestedURLS = str(sys.argv[1])
RequestedURLS = RequestedURLS.split("::--::--::")
response = ""
for url in RequestedURLS:
    response += str(random.randint(1,100))
    response += "::--::--::"
print(response)