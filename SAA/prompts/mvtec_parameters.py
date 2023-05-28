manual_prompts = {
    'carpet': [
        # prompts, filtered phrase
        ['black hole', 'carpet'],
        ['thread', 'carpet'],
        ['defect.', 'carpet'],

    ],

    'grid': [
        # prompts, filtered phrase
        ['irregular pattern', 'grid'],
        ['defect.', 'grid'],
    ],

    'leather': [
        ['defect.', 'leather'],
    ],

    'tile': [
        ['defect.', 'tile'],
    ],

    'wood': [
        ['defect.', 'wood'],
    ],

    'bottle': [
        # prompts, filtered phrase
        ['broken part. contamination. white broken.', 'bottle'],
    ],

    'cable': [
        # prompts, filtered phrase
        ['crack. flawed golden wire. black hole.', 'cable'],
    ],

    'capsule': [
        ['white crack. hole.', 'capsule'],
        # ['hole on capsule', 'capsule']

    ],

    'hazelnut': [
        # prompts, filtered phrase
        ['white print. crack. thread.', 'hazelnut'],
    ],

    'metal_nut': [
        # prompts, filtered phrase
        ['blue defect. black defect. red defect. scratch.', 'nut'],
    ],

    'pill': [
        # prompts, filtered phrase
        ['red defect. yellow defect. blue defect. crack. scratch.', 'pill'],
    ],

    'screw': [
        ['defect.', 'screw'],
    ],

    'toothbrush': [
        ['defect.', 'toothbrush'],
    ],

    'transistor': [
        ['defect.', 'transistor'],
    ],

    'zipper': [
        ['crack. broken leather.', 'zipper']
    ]
}

property_prompts = {
    'carpet': 'the image of carpet have 1 dissimilar carpet, with a maximum of 5 anomaly. The anomaly would not exceed 0.9 object area. ',
    'grid': 'the image of grid have 1 dissimilar grid, with a maximum of 5 anomaly. The anomaly would not exceed 0.9 object area. ',
    'leather': 'the image of leather have 1 dissimilar leather, with a maximum of 5 anomaly. The anomaly would not exceed 0.9 object area. ',
    'tile': 'the image of tile have 1 dissimilar tile, with a maximum of 5 anomaly. The anomaly would not exceed 0.9 object area. ',
    'wood': 'the image of wood have 1 dissimilar wood, with a maximum of 5 anomaly. The anomaly would not exceed 0.9 object area. ',
    'bottle': 'the image of bottle have 1 dissimilar bottle, with a maximum of 5 anomaly. The anomaly would not exceed 0.3 object area. ',
    'cable': 'the image of cable have 1 dissimilar cable, with a maximum of 5 anomaly. The anomaly would not exceed 0.9 object area. ',
    'capsule': 'the image of capsule have 1 dissimilar capsule, with a maximum of 5 anomaly. The anomaly would not exceed 0.6 object area. ',
    'hazelnut': 'the image of hazelnut have 1 dissimilar hazelnut, with a maximum of 5 anomaly. The anomaly would not exceed 0.9 object area. ',
    'metal_nut': 'the image of metal_nut have 1 dissimilar metal_nut, with a maximum of 5 anomaly. The anomaly would not exceed 1. object area. ',
    'pill': 'the image of pill have 1 dissimilar pill, with a maximum of 5 anomaly. The anomaly would not exceed 1. object area. ',
    'screw': 'the image of screw have 1 dissimilar screw, with a maximum of 5 anomaly. The anomaly would not exceed 0.1 object area. ',
    'toothbrush': 'the image of toothbrush have 1 dissimilar toothbrush, with a maximum of 5 anomaly. The anomaly would not exceed 0.5 object area. ',
    'transistor': 'the image of transistor have 1 dissimilar transistor, with a maximum of 5 anomaly. The anomaly would not exceed 1. object area. ',
    'zipper': 'the image of zipper have 1 dissimilar zipper, with a maximum of 5 anomaly. The anomaly would not exceed 0.5 object area. ',
}
