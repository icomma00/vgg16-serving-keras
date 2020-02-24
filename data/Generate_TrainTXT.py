import os 

# 生成标签文件：0为cat,1为dog
# cat.0.jpg;0
# cat.1.jpg;0
# cat.10.jpg;0
with open('./data/train.txt','w') as f:
    after_generate = os.listdir("./data/image/train")
    for image in after_generate:
        if image.split(".")[0]=='cat':
            f.write(image + ";" + "0" + "\n")
        else:
            f.write(image + ";" + "1" + "\n")
