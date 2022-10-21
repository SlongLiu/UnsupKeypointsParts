import os, sys
import os.path as osp

datafolder = '/data/shilong/data/imm/celeba'

def gen_all_wo_train_test():
    imgfolder = osp.join(datafolder, 'Img/img_align_celeba_hq')
    test_list_path = osp.join(datafolder, 'MAFL', 'testing.txt')
    train_list_path = osp.join(datafolder, 'MAFL', 'training.txt')
    outputpath = osp.join(datafolder, 'MAFL', 'all_wo_train_test.txt')
    if osp.exists(outputpath):
        print("%s has been existed. return." % outputpath)
        return

    # read txt
    test_list = []
    with open(test_list_path, 'r') as f:
        for line in f:
            test_list.append(line.strip())
    
    with open(train_list_path, 'r') as f:
        for line in f:
            test_list.append(line.strip())

    test_set = set(test_list)
    print("len of exclude file:", len(test_list))
    
    # all except test list
    aet_list = []
    imglist = os.listdir(imgfolder)
    for imgname in imglist:
        if imgname in test_set:
            continue
        aet_list.append(imgname)

    print('len(aet_list):', len(aet_list))
    aet_list.sort()

    # write
    with open(outputpath, 'w') as f:
        for item in aet_list:
            f.write(item + '\n')

def gen_all_except_testing():
    imgfolder = osp.join(datafolder, 'Img/img_align_celeba_hq')
    test_list_path = osp.join(datafolder, 'MAFL', 'testing.txt')
    outputpath = osp.join(datafolder, 'MAFL', 'all_except_testing.txt')
    if osp.exists(outputpath):
        print("%s has been existed. return." % outputpath)
        return

    # read txt
    test_list = []
    with open(test_list_path, 'r') as f:
        for line in f:
            test_list.append(line.strip())
    test_set = set(test_list)
    print("len of exclude file:", len(test_list))
    
    # all except test list
    aet_list = []
    imglist = os.listdir(imgfolder)
    for imgname in imglist:
        if imgname in test_set:
            continue
        aet_list.append(imgname)

    print('len(aet_list):', len(aet_list))
    aet_list.sort()

    # write
    with open(outputpath, 'w') as f:
        for item in aet_list:
            f.write(item + '\n')

def main():
    gen_all_except_testing()
    gen_all_wo_train_test()

if __name__ == "__main__":
    main()