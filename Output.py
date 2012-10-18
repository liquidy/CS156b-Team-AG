import pickle

a = [1,2]
output = open('userFeature', 'wb')
pickle.dump(a, output)
output.close()

read_file = open('userFeature', 'rb')
b = pickle.load(read_file)
print b