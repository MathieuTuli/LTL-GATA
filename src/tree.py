class Tree:
    def __init__(self,) -> None:
        ...


class TreeNode:
    def __init__(self, value, parent=None, children=None, mask=None):
        self.value = value
        self.parent = parent
        self.children = children if children else []
        self.mask = mask if mask is not None else [1.0] * 81
        self._positional_encoding = None
        if self.parent is None:
            self.branch = 0
        else:
            if self not in self.parent.children:
                self.parent.children.append(self)
            self.branch = self.parent.children.index(self)

    def num_children(self):
        return len(self.children)

    def size(self):
        return 1 + sum([child.size() for child in self.children])

    def depth(self):
        if self.is_leaf():
            return 0
        return 1 + max([child.depth() for child in self.children])

    def height(self):
        if self.parent is None:
            return 0
        return 1 + self.parent.height()

    def width(self):
        return max([self.num_children()] + [child.width() for
                                            child in self.children])

    def is_leaf(self):
        return self.num_children() == 0

    def is_first_child(self):
        return self.branch == 0

    def is_last_child(self):
        return self.branch == self.parent.num_children() - 1 if\
            self.parent else True

    def get_positional_encoding(self):
        if self._positional_encoding is None:
            if self.parent:
                self._positional_encoding = [
                    0.0 for _ in range(self.parent.num_children())]
                self._positional_encoding[self.branch] = 1.0
                self._positional_encoding += \
                    self.parent.get_positional_encoding()
            else:
                self._positional_encoding = []
        return self._positional_encoding

    def get_padded_positional_encoding(self, max_pos_len):
        padded = [x for x in self.get_positional_encoding()]
        while len(padded) < max_pos_len:
            padded.append(0.0)
        padded = padded[: max_pos_len]
        return padded

    def is_isomorphic(self, arg, struct_only=False):
        if (struct_only or self.value == arg.value) and \
                self.num_children() == arg.num_children():
            for i in range(len(self.children)):
                if not self.children[i].is_isomorphic(arg.children[i],
                                                      struct_only):
                    return False
            return True
        return False

    def prefix_traversal(self):
        def _prefix(node):
            yield node
            for child in node.children:
                yield from _prefix(child)
        yield from _prefix(self)

    def postfix_traversal(self):
        def _postfix(node):
            for child in node.children:
                yield from _postfix(child)
            yield node
        yield from _postfix(self)

    def depth_first_traversal(self):
        yield from self.prefix_traversal()

    def breadth_first_traversal(self):
        unresolved = [self]
        while unresolved:
            yield unresolved[0]
            unresolved += unresolved[0].children
            del unresolved[0]

    def choose_traversal(self, str_):
        str_to_traversal = {
            "prefix": self.prefix_traversal,
            "postfix": self.postfix_traversal,
            "depth_first": self.depth_first_traversal,
            "breadth_first": self.breadth_first_traversal
        }
        yield from str_to_traversal[str_]()

    def convert_to_sequence(self, traversal, separator=' '):
        seq = ""
        for node in traversal:
            seq += str(node.value) + separator
        return seq

    def fill(self, branch_factor=2, placeholder_token='_NULL'):
        fill_tree = {}
        for node in self.depth_first_traversal():
            value = node.value
            if node.is_leaf():
                value += "_0"
            if node is self:
                fill_tree[node] = TreeNode(value)
            else:
                fill_tree[node] = TreeNode(value, fill_tree[node.parent])
        for node in self.depth_first_traversal():
            if not node.is_leaf():
                while len(fill_tree[node].children) < branch_factor:
                    TreeNode(placeholder_token, fill_tree[node])
        return fill_tree[self]

    def left_child_right_sibling(self, placeholder_token='_NULL'):
        lcrs_tree = {}
        for node in self.depth_first_traversal():
            if node is self:
                lcrs_tree[node] = TreeNode(node.value)
            else:
                if node.is_first_child():
                    lcrs_tree[node] = TreeNode(
                        node.value, lcrs_tree[node.parent])
                    if node.parent.is_last_child():
                        TreeNode(placeholder_token, lcrs_tree[node.parent])
                else:
                    lcrs_tree[node] = TreeNode(
                        node.value, lcrs_tree[node.parent.children[
                            node.branch - 1]])
                if node.is_leaf():
                    TreeNode(placeholder_token, lcrs_tree[node])
                    if node.is_last_child():
                        TreeNode(placeholder_token, lcrs_tree[node])
        return lcrs_tree[self]

    def inverse_left_child_right_sibling(self, placeholder_token='_NULL'):
        ilcrs_tree = {}
        try:
            for node in self.depth_first_traversal():
                if node.num_children() == 1:
                    TreeNode(placeholder_token, node)
            for node in self.depth_first_traversal():
                if node is self:
                    ilcrs_tree[node] = TreeNode(node.value)
                elif node.value != placeholder_token:
                    true_first_child = node
                    while true_first_child.branch == 1:
                        true_first_child = true_first_child.parent
                    ilcrs_tree[node] = TreeNode(
                        node.value, ilcrs_tree[true_first_child.parent])
            return ilcrs_tree[self]
        except Exception:
            return TreeNode(placeholder_token)
