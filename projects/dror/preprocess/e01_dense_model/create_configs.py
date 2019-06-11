import os
from copy import copy


function = type(lambda: None)


TemplatePath = r'C:\Users\Danielle\PycharmProjects\Shmekel\projects\dror\preprocess\e01_dense_model\template.json'
ConfigsDir = r'C:\Users\Danielle\PycharmProjects\Shmekel\projects\dror\preprocess\e01_dense_model\configs'

if not os.path.exists(ConfigsDir):
    os.makedirs(ConfigsDir)

variables = [
    ('__BNBool__', ['false']),
    ('__Depth__', ['1', '2', '3']),
    ('__OutputShape__', ['[2]']),
    ('__SkipBool__', ['false']),
    ('__Width__', ['16', '32', '64']),
    ('__ProjectName__', ['preprocess']),
    ('__Epochs__', ['10']),
    ('__TrainAugmentations__', ['null']),
    ('__ValAugmentations__', ['null']),
    ('__FeatureList__', ['null']),
    ('__OutputFeatureList__', ['null']),
    ('__TimeSampleLength__', ['1', '3', '5']),
    ('__InputShape__', [lambda x: '[%s, 5]' % x['__TimeSampleLength__']]),
    ('__Name__', [lambda x: 't_%s--d_%s--w_%s' % (x['__TimeSampleLength__'],
                                                  x['__Depth__'],
                                                  x['__Width__'])]),
]

indices = [[0] * len(variables)]
for i, (_, options) in enumerate(variables):
    tmp = indices
    indices = []
    for j, v in enumerate(options):
        new = [copy(t) for t in tmp]
        for n in new:
            n[i] = j
        indices.extend(new)

combs = [{key: options[i] for i, (key, options) in zip(ind, variables)} for ind in indices]

for comb in combs:
    for key, val in comb.items():
        if isinstance(val, function):
            comb[key] = val(comb)

with open(TemplatePath) as f:
    o = f.read().split('__VARIABLES__')[0].strip()

for comb in combs:
    s = o
    for key, val in comb.items():
        s = s.replace(key, val)

    with open(os.path.join(ConfigsDir, comb['__Name__'] + '.json'), 'w') as f:
        f.write(s)


