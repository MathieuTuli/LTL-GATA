import re

from logic import Proposition, Variable, proposition_from_textworld_logic
import pdb

prep_map = {
    'fry': 'fried',
    'roast': 'roasted',
    'grill': 'grilled',
    'chop': 'chopped',
    'dice': 'diced',
    'slice': 'sliced',
}

ingredients = {
    'banana', 'pork chop', 'carrot', 'parsley',
    'chicken leg', 'yellow bell pepper',
    'red potato', 'chicken wing', 'purple potato',
    'yellow potato', 'black pepper', 'block of cheese',
    'flour', 'cilantro', 'white onion', 'olive oil',
    'orange bell pepper', 'water', 'red hot pepper',
    'salt', 'red onion', 'red apple'
}


class Translator:
    def __init__(self, level: int) -> None:
        self.level = level
        self.formulas = list()
        self.first = True
        self.cookbook_read = False
        self.violated = False

    def generate_ltl(self, obs: str) -> None:
        if self.cookbook_read or self.violated:
            return
        if 'check the cookbook' in obs and self.first:  # first obs
            self.first = False
            if self.level in {5, 9, 'mixed'}:
                at_kitchen = Proposition(
                    name='at',
                    arguments=[Variable('player', type='I'),
                               Variable('kitchen', type='r')],
                    seperator='_')
                self.formulas.append(('eventually', str(at_kitchen)))
            examined_cookbook = Proposition(
                name='examined',
                arguments=[Variable('cookbook', type='o')],
                seperator='_')
            # self.formulas.append(('next', str(examined_cookbook)))
            consumed = Proposition(
                name='consumed',
                arguments=[Variable('meal', type='meal')],
                seperator='_')
            self.formulas.append(('and', ('next', str(examined_cookbook)),
                                  ('eventually', str(consumed))))
            # self.formulas.append(('eventually', str(consumed)))
        elif not self.cookbook_read and \
                'ingredients :' in obs and 'directions :' in obs:
            self.cookbook_read = True
            if 'gather' in obs and 'follow the directions' in obs:
                split_ing = obs.split('ingredients : ')[-1]
                split = split_ing.split(' directions :')
                ings = [x for x in ingredients if x in split[0].strip()]
                string = split[0].strip()
                for i, ing in enumerate(ings):
                    string = string.replace(ing, f"xx{i}")
                indices = list()
                for x in string.split('xx'):
                    try:
                        x = int(x)
                        indices.append(x)
                    except Exception:
                        pass
                ings = [ings[i] for i in indices]
                prepare = 'prepare meal' in obs
                preps = list()
                if prepare:
                    # ' dice the orange bell pepper roast the orange bell pepper chop the purple potato fry the purple potato '
                    # string = split[1].replace('prepare meal', '').replace(
                    #     f'the {ing}', '').strip()
                    string = split[1].replace(
                        'prepare meal', '').replace('the', '').strip()
                    string = re.sub('\s+', ' ', string)
                    content = string.split(' ')
                    action = None
                    actionable_ing = None
                    if content == [""]:
                        content = list()
                    prev_word = ''
                    for word in content:
                        if word in prep_map.keys() and not (prev_word == 'pork' and word == 'chop'):
                            if action is not None:
                                actionable_ing = re.sub(
                                    '\s+', ' ', actionable_ing)
                                actionable_ing = actionable_ing.strip()
                                preps.append((action, actionable_ing))
                            action = word
                            actionable_ing = ''
                        else:
                            actionable_ing += f' {word}'
                        prev_word = word
                    if action is not None:
                        actionable_ing = re.sub(
                            '\s+', ' ', actionable_ing)
                        actionable_ing = actionable_ing.strip()
                        preps.append((action, actionable_ing))

                    # preps = string.split(' ')
                    if preps == ['']:
                        preps = list()
                props = [
                    ('eventually', str(Proposition(
                        name='in',
                        arguments=[
                            Variable(ing, type='f'),
                            Variable('player', type='I'), ],
                        seperator='_'))) for ing in ings]
                for p, ing in preps:
                    try:
                        method = prep_map[p]
                    except:
                        pdb.set_trace()
                    props.append(('eventually', str(Proposition(
                        name=method,
                        arguments=[
                            Variable(ing, type='f'),
                        ],
                        seperator='_'))))
                if prepare:
                    props.append(('eventually', str(Proposition(
                        name='in',
                        arguments=[
                            Variable('meal', type='meal'),
                            Variable('player', type='I'), ],
                        seperator='_'))))
                # props.append(self.formulas[-1])
                curr_form = None
                prev_form = None
                for i, prop in enumerate(reversed(props)):
                    rel = 'and'
                    if prev_form is None:
                        curr_form = prop
                    else:
                        curr_form = (rel, prop, prev_form)
                    prev_form = curr_form
                ltl = curr_form
                self.formulas.append(tuple(ltl))
