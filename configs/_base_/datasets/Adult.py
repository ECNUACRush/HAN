dataset_info = dict(
    dataset_name='cepha',
    paper_info=dict(
        author='Lin, Tsung-Yi and Maire, Michael and '
        'Belongie, Serge and Hays, James and '
        'Perona, Pietro and Ramanan, Deva and '
        r'Doll{\'a}r, Piotr and Zitnick, C Lawrence',
        title='Microsoft coco: Common objects in context',
        container='European conference on computer vision',
        year='2014',
        homepage='http://cocodataset.org/',
    ),
    keypoint_info={
        0:
        dict(name='sella_turcica', id=0, color=[51, 153, 255], type='', swap=''),
        1:
        dict(
            name='nasion',
            id=1,
            color=[51, 153, 255],
            type='',
            swap=''),
        2:
        dict(
            name='orbitale',
            id=2,
            color=[51, 153, 255],
            type='',
            swap=''),
        3:
        dict(
            name='porion',
            id=3,
            color=[51, 153, 255],
            type='',
            swap=''),
        4:
        dict(
            name='subspinale',
            id=4,
            color=[51, 153, 255],
            type='',
            swap=''),
        5:
        dict(
            name='supramentale',
            id=5,
            color=[0, 255, 0],
            type='',
            swap=''),
        6:
        dict(
            name='pogonion',
            id=6,
            color=[255, 128, 0],
            type='',
            swap=''),
        7:
        dict(
            name='menton',
            id=7,
            color=[0, 255, 0],
            type='',
            swap=''),
        8:
        dict(
            name='gnathion',
            id=8,
            color=[255, 128, 0],
            type='',
            swap=''),
        9:
        dict(
            name='gonion',
            id=9,
            color=[0, 255, 0],
            type='',
            swap=''),
        
    },
    # skeleton_info={
    #     0:
    #     dict(link=('nasion', 'sella_turcica'), id=0, color=[255,0,0]),
    #     1:
    #     dict(link=('sella_turcica', 'porion'), id=1, color=[255,0,0]),
    #     2:
    #     dict(link=('porion', 'articulate'), id=2, color=[255,0,0]),
    #     3:
    #     dict(link=('articulate', 'posterior_nasal_spine'), id=3, color=[255,0,0]),
    #     4:
    #     dict(link=('articulate', 'gonion'), id=4, color=[255,0,0]),
    #     5:
    #     dict(link=('posterior_nasal_spine', 'anterior_nasal_spine'), id=5, color=[255,0,0]),
    #     6:
    #     dict(link=('nasion', 'orbitale'), id=6, color=[255,0,0]),
    #     7:
    #     dict(
    #         link=('orbitale', 'anterior_nasal_spine'),
    #         id=7,
    #         color=[255,0,0]),
    #     8:
    #     dict(link=('anterior_nasal_spine', 'subnasale'), id=8, color=[255,0,0]),
    #     9:
    #     dict(
    #         link=('subnasale', 'upper_lip'), id=9, color=[255,0,0]),
    #     10:
    #     dict(link=('upper_lip', 'lower_lip'), id=10, color=[255,0,0]),
    #     11:
    #     dict(link=('lower_lip', 'soft_tissue_pogonion'), id=11, color=[255,0,0]),
    #     12:
    #     dict(link=('anterior_nasal_spine', 'subspinale'), id=12, color=[255,0,0]),
    #     13:
    #     dict(link=('subspinale', 'upper_incisal_incision'), id=13, color=[255,0,0]),
    #     14:
    #     dict(link=('upper_incisal_incision', 'lower_incisal_incision'), id=14, color=[255,0,0]),
    #     15:
    #     dict(link=('lower_incisal_incision', 'supramentale'), id=15, color=[255,0,0]),
    #     16:
    #     dict(link=('supramentale', 'pogonion'), id=16, color=[255,0,0]),
    #     17:
    #     dict(link=('pogonion', 'gnathion'), id=17, color=[255,0,0]),
    #     18:
    #     dict(
    #         link=('gnathion', 'menton'), id=18, color=[255,0,0]),
    #     19:
    #     dict(
    #         link=('menton', 'gonion'), id=19, color=[255,0,0]),
    #     20:
    #     dict(
    #         link=('gonion', 'articulate'), id=20, color=[255,0,0])
    # },
    joint_weights=[
        1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1.,
        1., 1., 1.
    ],
    sigmas=[]
    # sigmas=[
    #     0.026, 0.025, 0.025, 0.035, 0.035, 0.079, 0.079, 0.072, 0.072, 0.062,
    #     0.062, 0.107, 0.107, 0.087, 0.087, 0.089, 0.089, 0.089, 0.089
    # ]
)
# 获取关键点个数
# NUM_KEYPOINTS = len(dataset_info['keypoint_info'])
# dataset_info['joint_weights'] = [1.0] * NUM_KEYPOINTS
# dataset_info['sigmas'] = [0.025] * NUM_KEYPOINTS