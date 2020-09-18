import pickle

with open('/home/tupm/HDD/projects/tensorflow-object-detection-cpp/assets/text_models/char_mapping_addr_3591.pkl', 'rb') as f:
    data = pickle.load(f)

# arr = [424, 246, 1180, 2947]

arr = [3222,
1223,
2397,
3418]

f = open('/home/tupm/HDD/projects/tensorflow-object-detection-cpp/assets/text_models/labels.txt', 'w')

for i, e in zip(data.keys(), data.values()):
    
    # if e == 10:
    #     break
    # print(i, e)
    f.write(i+'\n')

# for v in arr:
#     for i, e in zip(data.keys(), data.values()):
        
#         if e == v-3:
#             print(i, e)
#             break
        