import os, sys
import os.path as osp
import xml.etree.ElementTree as ET

data_dir = '/data/shilong/data/pascal_voc/2012/VOCdevkit/VOC2012'
seglistfile = osp.join(data_dir, 'ImageSets/Segmentation/trainval.txt')
seglist = [line.strip() for line in open(seglistfile, 'r')]

outputdir = osp.join(data_dir, 'ImageSets/SelectSeg3')
os.makedirs(outputdir, exist_ok=True)

for idx, itemname in enumerate(seglist):
    imgpath = osp.join(data_dir, 'JPEGImages', itemname+'.jpg')
    xmlpath = osp.join(data_dir, 'Annotations', itemname+'.xml')

    # parse xml
    tree = ET.parse(xmlpath)
    root = tree.getroot()
    size = root.find('size')
    width = int(size.find('width').text)
    height = int(size.find('height').text)

    oblist = root.findall('object')
    oblist_select = []
    for ob in oblist:
        obname = ob.find('name').text
        truncated = int(ob.find('truncated').text)
        if truncated:
            continue
        bndbox = ob.find('bndbox')
        xmin = int(bndbox.find('xmin').text)
        ymin = int(bndbox.find('ymin').text)
        xmax = int(bndbox.find('xmax').text)
        ymax = int(bndbox.find('ymax').text)
        w = xmax - xmin
        h = ymax - ymin
        oblist_select.append((obname, w*h / (width * height)))
    oblist_select.sort(key=lambda x: x[1], reverse=True)
    if len(oblist_select) >= 1:
        obname_s, obarea_s = oblist_select[0]
        if (obarea_s > 0.2) and (obname_s not in [i[0] for i in oblist_select[1:]]):
            with open(osp.join(outputdir, 'list_%s.txt' % obname_s), 'a') as f:
                f.write(itemname+'\n')

        
