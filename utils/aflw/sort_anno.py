import os, sys
import os.path as osp


annodir = '/data/shilong/data/AFLW/wild_aflw/anno'
anno_5kp = 'anno_5kp.txt'
anno_5kp_out = 'anno_5kp_sort.txt'

anno_21kp = 'anno_21kp.txt'
anno_21kp_out = 'anno_21kp_sort.txt'

filter_data = 'filter_data_all.txt'
filter_data_out = 'filter_data_all_sort.txt'

def sort_file(inputfile, outputfile, skipfirst2line=True):
    datalist = [line.strip() for line in open(inputfile, 'r')]
    if skipfirst2line:
        datalist_raw = sorted(datalist[2:])
    else:
        datalist_raw = sorted(datalist)
    with open(outputfile, 'w') as f:
        if skipfirst2line:
            f.write(datalist[0].strip() + '\n')
            f.write(datalist[1].strip() + '\n')

        for item in datalist_raw:
            f.write(item.strip() + '\n')
    
# sort_file(osp.join(annodir, anno_5kp), osp.join(annodir, anno_5kp_out))
# sort_file(osp.join(annodir, anno_21kp), osp.join(annodir, anno_21kp_out))
sort_file(osp.join(annodir, filter_data), osp.join(annodir, filter_data_out), skipfirst2line=False)

