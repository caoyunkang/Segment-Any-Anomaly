general_anomaly_description = [
    'defect on {}',
    'damage on {}',
    'flaw on {}',
]


def build_general_prompts(category):
    return [[f.format(category), category] for f in general_anomaly_description]
