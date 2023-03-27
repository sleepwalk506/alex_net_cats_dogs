import os
from shutil import copy
import random


def mkfile(file):
    if not os.path.exists(file):
        os.makedirs(file)


# 鑾峰彇data鏂囦欢澶逛笅鎵€鏈夋枃浠跺す鍚嶏紙鍗抽渶瑕佸垎绫荤殑绫诲悕锛�
file_path = 'D:/deep learning2/hualidefeng_nets/alex_net/data_name'
flower_class = [cla for cla in os.listdir(file_path)]

# 鍒涘缓 璁�缁冮泦train 鏂囦欢澶癸紝骞剁敱绫诲悕鍦ㄥ叾鐩�褰曚笅鍒涘缓5涓�瀛愮洰褰�
mkfile('data/train')
for cla in flower_class:
    mkfile('data/train/' + cla)

# 鍒涘缓 楠岃瘉闆唙al 鏂囦欢澶癸紝骞剁敱绫诲悕鍦ㄥ叾鐩�褰曚笅鍒涘缓瀛愮洰褰�
mkfile('data/val')
for cla in flower_class:
    mkfile('data/val/' + cla)

# 鍒掑垎姣斾緥锛岃��缁冮泦 : 楠岃瘉闆� = 9 : 1
split_rate = 0.2

# 閬嶅巻鎵€鏈夌被鍒�鐨勫叏閮ㄥ浘鍍忓苟鎸夋瘮渚嬪垎鎴愯��缁冮泦鍜岄獙璇侀泦
for cla in flower_class:
    cla_path = file_path + '/' + cla + '/'  # 鏌愪竴绫诲埆鐨勫瓙鐩�褰�
    images = os.listdir(cla_path)  # iamges 鍒楄〃瀛樺偍浜嗚�ョ洰褰曚笅鎵€鏈夊浘鍍忕殑鍚嶇О
    num = len(images)
    eval_index = random.sample(images, k=int(num * split_rate))  # 浠巌mages鍒楄〃涓�闅忔満鎶藉彇 k 涓�鍥惧儚鍚嶇О
    for index, image in enumerate(images):
        # eval_index 涓�淇濆瓨楠岃瘉闆唙al鐨勫浘鍍忓悕绉�
        if image in eval_index:
            image_path = cla_path + image
            new_path = 'data/val/' + cla
            copy(image_path, new_path)  # 灏嗛€変腑鐨勫浘鍍忓�嶅埗鍒版柊璺�寰�

        # 鍏朵綑鐨勫浘鍍忎繚瀛樺湪璁�缁冮泦train涓�
        else:
            image_path = cla_path + image
            new_path = 'data/train/' + cla
            copy(image_path, new_path)
        print("\r[{}] processing [{}/{}]".format(cla, index + 1, num), end="")  # processing bar
    print()

print("processing done!")
