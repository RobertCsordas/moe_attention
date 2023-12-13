class OncePerIterLayer:
    def pre_train_forward(self):
        pass

    def post_train_forward(self):
        pass

    def before_loss(self):
        pass


def call_pre_iter(model):
    for module in model.modules():
        if isinstance(module, OncePerIterLayer):
            module.pre_train_forward()


def call_post_iter(model):
    for module in model.modules():
        if isinstance(module, OncePerIterLayer):
            module.post_train_forward()


def call_before_loss(model):
    for module in model.modules():
        if isinstance(module, OncePerIterLayer):
            module.before_loss()
