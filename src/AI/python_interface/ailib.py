import _ailib

class CostFunction():
    SQUARED_ERROR = 0
    CROSSENTROPY = 1

class neuralnetwork():
    
    def __init__(self, filepath=""):
        if filepath == "": self.net = _ailib.neuralnetwork_init()
        else: self.net = _ailib.neuralnetwork_init_fromfile(filepath)
    
    def __del__(self):
        _ailib.neuralnetwork_free(self.net)

    def save(self, filepath):
        _ailib.neuralnetwork_save(self.net, filepath)
    
    def load(self, filepath):
        _ailib.neuralnetwork_load(self.net, filepath)
    
    def clear(self):
        _ailib.neuralnetwork_clear(self.net)
    
    def run(self, inputs, training=False):
        _ailib.neuralnetwork_run(self.net, inputs, training)
    
    def optimize(self, inputs, targets, optimizer):
        return _ailib.neuralnetwork_optimize(self.net, inputs, targets, optimizer.get_c_pointer())
    
    def getoutput(self, node_name):
        return _ailib.neuralnetwork_getoutput(self.net, node_name)
    
    def printstack(self):
        _ailib.neuralnetwork_printstack(self.net)

    def push_variable(self, operation_name, width, height=1, depth=1):
        _ailib.neuralnetwork_push_variable(self.net, operation_name, "", width, height, depth)
    
    def push_linear(self, operation_name, operation_inputs, size, use_bias=True,
        gradient_clipping=0.0, l1_regularization=0.0, l2_regularization=0.0):
        _ailib.neuralnetwork_push_linear(self.net, operation_name, operation_inputs,
            size, use_bias, gradient_clipping, l1_regularization, l2_regularization)
    
    def push_convolution(self, operation_name, operation_inputs, filter_width, filter_height,
        filter_count, stride, padding, gradient_clipping=0.0, l1_regularization=0.0, l2_regularization=0.0):
        _ailib.neuralnetwork_push_convolution(self.net, operation_name, operation_inputs,
            filter_width, filter_height, filter_count, stride, padding, gradient_clipping,
            l1_regularization, l2_regularization)
    
    def push_maxpooling(self, operation_name, operation_inputs, filter_size=2, padding=2):
        _ailib.neuralnetwork_push_maxpooling(self.net, operation_name, operation_inputs, filter_size, padding)
    
    def push_averagepooling(self, operation_name, operation_inputs, filter_size=2, padding=2):
        _ailib.neuralnetwork_push_averagepooling(self.net, operation_name, operation_inputs, filter_size, padding)
    
    def push_dropout(self, operation_name, operation_inputs, drop_probability):
        _ailib.neuralnetwork_push_dropout(self.net, operation_name, operation_inputs, drop_probability)
    
    def push_sigmoid(self, operation_name, operation_inputs):
        _ailib.neuralnetwork_push_sigmoid(self.net, operation_name, operation_inputs)
    
    def push_tanh(self, operation_name, operation_inputs):
        _ailib.neuralnetwork_push_tanh(self.net, operation_name, operation_inputs)
    
    def push_relu(self, operation_name, operation_inputs):
        _ailib.neuralnetwork_push_relu(self.net, operation_name, operation_inputs)
    
    def push_selu(self, operation_name, operation_inputs):
        _ailib.neuralnetwork_push_selu(self.net, operation_name, operation_inputs)
    
    def push_softmax(self, operation_name, operation_inputs, input_scale=1.0):
        _ailib.neuralnetwork_push_softmax(self.net, operation_name, operation_inputs, input_scale)
    
    def push_normalization(self, operation_name, operation_inputs, momentum=0.5):
        _ailib.neuralnetwork_push_normalization(self.net, operation_name, operation_inputs, momentum)
    
    def push_recurrent(self, operation_name, operation_inputs, size, btt_steps=3):
        _ailib.neuralnetwork_push_recurrent(self.net, operation_name, operation_inputs, size, btt_steps)
    
    def get_c_pointer(self):
        return self.net

class optimizer_sdg():
    def __init__(self, batch_size, learningrate, momentum, cost_function):
        self.optimizer = _ailib.optimizer_sdg_init(batch_size, learningrate, momentum, cost_function)
        self.batch_size = batch_size
        self.learningrate = learningrate
        self.momentum = momentum
        self.cost_function = cost_function
    
    def __del__(self):
        _ailib.optimizer_sdg_free(self.optimizer)
    
    def set_learningrate(self, learningrate):
        self.learningrate = learningrate
        _ailib.optimizer_sdg_set_learningrate(self.optimizer, learningrate)
    
    def get_learningrate(self):
        return self.learningrate

    def set_momentum(self, momentum):
        self.momentum = momentum
        _ailib.optimizer_sdg_set_momentum(self.optimizer, momentum)

    def get_momentum(self):
        return self.momentum

    def get_c_pointer(self):
        return self.optimizer

class dqn_agent():
    def __init__(self, neuralnetwork, short_term_memory_size, learningrate, curiosity, longtermexploitation):
        self.neuralnetwork = neuralnetwork
        self.short_term_memory_size = short_term_memory_size
        self.learningrate = learningrate
        self.curiosity = curiosity
        self.longtermexploitation = longtermexploitation
        self.agent = _ailib.dqn_init(neuralnetwork.get_c_pointer(), short_term_memory_size, learningrate, curiosity, longtermexploitation)
    
    def __del__(self):
        _ailib.dqn_free(self.agent)
    
    def act(self, state):
        return _ailib.dqn_act(self.agent, state)
    
    def observe(self, new_state, oldreward):
        _ailib.dqn_observe(self.agent, new_state, oldreward)

    def set_learningrate(self, learningrate):
        self.learningrate = learningrate
        _ailib.dqn_set_learningrate(self.agent, learningrate)
    
    def set_curiosity(self, curiosity):
        self.curiosity = curiosity
        _ailib.dqn_set_curiosity(self.agent, curiosity)
    
    def set_longtermexploitation(self, longtermexploitation):
        self.longtermexploitation = longtermexploitation
        _ailib.dqn_set_longtermexploitation(self.agent, longtermexploitation)
    
    def get_learningrate(self):
        return self.learningrate
    
    def get_curiosity(self):
        return self.curiosity
    
    def get_longtermexploitation(self):
        return self.longtermexploitation
    
    def get_c_pointer():
        return self.agent
    

