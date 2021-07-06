import json

from tqdm import tqdm


def main():
    for split in ['train', 'test', 'valid']:
        print(split)
        with open(f'cmd_gen.0.2/{split}.json') as f:
            data = json.load(f)
        data['graph_index'] = json.loads(data['graph_index'])
        rkey = list(data['graph_index']['relations'].keys())[-1]
        rkey = str(int(rkey) + 1)
        ekey = list(data['graph_index']['entities'].keys())[-1]
        ekey = str(int(ekey) + 1)
        data['graph_index']['entities'][ekey] = 'examined'

        # Add cookbook is examined relation
        data['graph_index']['relations'][rkey] = [
            int(list(data['graph_index']
                ['entities'].values()).index('cookbook')),
            int(ekey),
            int(list(data['graph_index']
                ['relation_types'].values()).index('is')),
        ]

        # go through examples and
        #   1. get graph index of examine cookbook and add above rkey
        #   2. add 'Add , cookbook , examined , is
        """
{'game': 'tw-cooking-recipe1+take1+drop+go9-yGMMf1gdtY2giP5e.z8',
 'step': [0, 0],
 'observation': "you are hungry ! ",
 'previous_action': 'restart',
 'previous_graph_seen': 0,
 'target_commands': ['add , exit , livingroom , east_of',
  'add , player , livingroom , at',
  'add , sofa , livingroom , at']}
        """
        for i, example in enumerate(tqdm(data['examples'])):
            # example = json.loads(example)
            if example['previous_action'] == 'examine cookbook':
                example['target_commands'].append(
                    'add , cookbook , examined , is')
                graph_idx = example['previous_graph_seen']
                data['graph_index']['graphs'][str(graph_idx)].append(int(rkey))
            data['examples'][i] = example
        data['graph_index'] = json.dumps(data['graph_index'])
        with open(f'cmd_gen.0.2/{split}_aug.json', 'w') as f:
            json.dump(data, f)


if __name__ == "__main__":
    main()
