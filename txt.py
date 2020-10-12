# 2020-10-06
# 11:25:03
# 48
# 1
# 0

f = open("./output.txt")
for line in f:
    # text = line.replace('-\n', '')
    text = line.splitlines()
    print(text)

# tmp = []

# f = open('./output.txt')
# header = f.readline()
# header_list = header.split()

# for line in f:
#     # print(len(line))
#     if len(line) == 1 :
#         pass
#     # print(line)
#     else : 
#         data = line.split()
#         tmp.append(data)
# print(tmp)