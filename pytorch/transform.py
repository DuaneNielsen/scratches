class MyTransform(object):

    def __init__(self, some_args):
        self.some_args = some_args

    def __call__(self, x):
        x_transform = do_something(x, self.some_args)
        return x_transform