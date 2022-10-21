import os
import pandas as pd
import numpy as np


class VggFace2(facedataset.FaceDataset):

    meta_folder = 'meta/bb_landmark'
    image_folder = 'data'

    def __init__(self, root, cache_root=None, train=True, crop_source='bb_ground_truth',
                 return_modified_images=False, min_face_height=100, **kwargs):

        assert(crop_source in ['bb_ground_truth', 'lm_ground_truth', 'lm_openface'])

        self.split_folder = 'train' if train else 'test'
        fullsize_img_dir = os.path.join(root, self.image_folder, self.split_folder)
        self.annotation_filename = 'loose_bb_{}.csv'.format(self.split_folder)

        super().__init__(root=root,
                         cache_root=cache_root,
                         fullsize_img_dir=fullsize_img_dir,
                         crop_source=crop_source,
                         crop_dir=os.path.join(self.split_folder, 'crops'),
                         return_landmark_heatmaps=False,
                         return_modified_images=return_modified_images,
                         **kwargs)

        self.min_face_height = min_face_height

        # shuffle images since dataset is sorted by identities
        import sklearn.utils
        self.annotations = sklearn.utils.shuffle(self.annotations)

        print("Removing faces with height <= {:.2f}px...".format(self.min_face_height))
        self.annotations = self.annotations[self.annotations.H > self.min_face_height]
        print("Number of images: {}".format(len(self)))
        print("Number of identities: {}".format(self.annotations.ID.nunique()))


    @property
    def cropped_img_dir(self):
        return os.path.join(self.cache_root, self.split_folder, 'crops', self.crop_source)

    def get_crop_extend_factors(self):
        return 0.05, 0.1

    @property
    def ann_csv_file(self):
        return os.path.join(self.root, self.meta_folder, self.annotation_filename)

    def _read_annots_from_csv(self):
        print('Reading CSV file...')
        annotations = pd.read_csv(self.ann_csv_file)
        print(f'{len(annotations)} lines read.')

        # assign new continuous ids to persons (0, range(n))
        print("Creating id labels...")
        _ids = annotations.NAME_ID
        _ids = _ids.map(lambda x: int(x.split('/')[0][1:]))
        annotations['ID'] = _ids

        return annotations

    def _load_annotations(self, split):
        path_annotations_mod = os.path.join(self.cache_root, self.annotation_filename + '.mod_full.pkl')
        if os.path.isfile(path_annotations_mod):
            annotations = pd.read_pickle(path_annotations_mod)
        else:
            annotations = self._read_annots_from_csv()
            annotations.to_pickle(path_annotations_mod)
        return annotations

    @property
    def labels(self):
        return self.annotations.ID.values

    @property
    def heights(self):
        return self.annotations.H.values

    @property
    def widths(self):
        return self.annotations.W.values

    @staticmethod
    def _get_identity(sample):
        return sample.ID

    def __len__(self):
        return len(self.annotations)

    def __getitem__(self, idx):
        sample = self.annotations.iloc[idx]
        # bb = self.get_adjusted_bounding_box(sample.X, sample.Y, sample.W, sample.H)
        bb = [sample.X, sample.Y, sample.X+sample.W, sample.Y+sample.H]
        bb = extend_bbox(bb, dt=0.05, db=0.10)
        landmarks_for_crop = sample.landmarks.astype(np.float32) if self.crop_source == 'lm_ground_truth' else None
        return self.get_sample(sample.NAME_ID+'.jpg', bb, landmarks_for_crop)


def extend_bbox(bbox, dl=0, dt=0, dr=0, db=0):
    '''
    Move bounding box sides by fractions of width/height. Positive values enlarge bbox for all sided.
    e.g. Enlarge height bei 10 percent by moving top:
    extend_bbox(bbox, dt=0.1) -> top_new = top - 0.1 * height
    '''
    l, t, r, b = bbox

    if t > b:
        t, b = b, t
    if l > r:
        l, r = r, l
    h = b - t
    w = r - l
    assert h >= 0
    assert w >= 0

    t_new, b_new = int(t - dt * h), int(b + db * h)
    l_new, r_new = int(l - dl * w), int(r + dr * w)

    return np.array([l_new, t_new, r_new, b_new])