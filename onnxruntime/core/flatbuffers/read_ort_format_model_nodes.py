import json
import os
import sys
import ort_flatbuffers_py.experimental.fbs as fbs
from abc import ABC, abstractmethod

# script_path = os.path.dirname(os.path.realpath(__file__))
# ci_build_py_path = os.path.abspath(os.path.join(script_path, '..', 'ci_build'))
# sys.path.append(ci_build_py_path)


class FbsTypeInfo:

    tensordatatype_to_string = {
        fbs.TensorDataType.TensorDataType.FLOAT: 'float',
        fbs.TensorDataType.TensorDataType.UINT8: 'uint8_t',
        fbs.TensorDataType.TensorDataType.INT8: 'int8_t',
        fbs.TensorDataType.TensorDataType.UINT16: 'uint16_t',
        fbs.TensorDataType.TensorDataType.INT16: 'int16_t',
        fbs.TensorDataType.TensorDataType.INT32: 'int32_t',
        fbs.TensorDataType.TensorDataType.INT64: 'int64_t',
        fbs.TensorDataType.TensorDataType.STRING: 'std::string',
        fbs.TensorDataType.TensorDataType.BOOL: 'bool',
        fbs.TensorDataType.TensorDataType.FLOAT16: 'MLFloat16',
        fbs.TensorDataType.TensorDataType.DOUBLE: 'double',
        fbs.TensorDataType.TensorDataType.UINT32: 'uint32_t',
        fbs.TensorDataType.TensorDataType.UINT64: 'uint64_t',
        fbs.TensorDataType.TensorDataType.COMPLEX64: 'complex64 is not supported',
        fbs.TensorDataType.TensorDataType.COMPLEX128: 'complex128 is not supported',
        fbs.TensorDataType.TensorDataType.BFLOAT16: 'BFloat16'
    }

    @staticmethod
    def typeinfo_to_str(type: fbs.TypeInfo):
        value_type = type.ValueType()
        value = type.Value()
        type_str = 'unknown'

        if value_type == fbs.TypeInfoValue.TypeInfoValue.tensor_type:
            tensor_type_and_shape = fbs.TensorTypeAndShape.TensorTypeAndShape()
            tensor_type_and_shape.Init(value.Bytes, value.Pos)
            elem_type = tensor_type_and_shape.ElemType()
            type_str = FbsTypeInfo.tensordatatype_to_string[elem_type]

        elif value_type == fbs.TypeInfoValue.TypeInfoValue.map_type:
            # TODO: Test
            map_type = fbs.MapType.MapType()
            map_type.init(value.Bytes, value.Pos)
            key_type = map_type.KeyType()  # TensorDataType
            key_type_str = FbsTypeInfo.tensordatatype_to_string[key_type]
            value_type = map_type.ValueType()  # TypeInfo
            value_type_str = FbsTypeInfo.typeinfo_to_str(value_type)
            type_str = 'std::map<{},{}>'.format(key_type_str, value_type_str)

        elif value_type == fbs.TypeInfoValue.TypeInfoValue.sequence_type:
            # TODO: Test.
            sequence_type = fbs.SequenceType.SequenceType()
            sequence_type.Init(value.Bytes, value.Pos)
            elem_type = sequence_type.ElemType()  # TypeInfo
            elem_type_str = FbsTypeInfo.typeinfo_to_str(elem_type)
            # TODO: Decide if we need to wrap the type in a std::vector. Issue is that the element type is internal
            # to the onnxruntime::Tensor class so we're really returning the type inside the Tensor not vector<Tensor>.
            # For now, return the element type (which will be the Tensor element type, or a map<A,B>) as
            # an operator input or output will either be a sequence or a not, so we don't need to disambiguate
            # between the two.
            # type_str = 'std::vector<{}>'.format(elem_type_str)
            type_str = elem_type_str
        else:
            raise ValueError('Unknown or missing value type of {}'.format(value_type))

        return type_str


def get_nodearg(name: str, nodearg_info: dict):
    if name not in nodearg_info:
        raise RuntimeError('Missing nodearg entry for ' + name)

    return nodearg_info[name]  # TypeInfo object


def name_to_typestr(name: str, nodearg_info: dict):
    type = get_nodearg(name, nodearg_info)
    type_str = FbsTypeInfo.typeinfo_to_str(type)
    return type_str


class OperatorProcessor(ABC):
    '''
    Abstract base class for processors which implement operator specific logic to determine the type or types required.
    '''
    def __init__(self, domain: str, optype: str):
        self._domain = domain
        self._optype = optype
        self._name = '{}:{}'.format(domain, optype)

    def name(self):
        return self._name

    def cpp_name(self):
        'Return a string that can be used as a unique name in a C++ #define.'
        return self._name.upper().replace('.', '_').replace(':', '_')

    @abstractmethod
    def process(self, node: fbs.Node, nodearg_info: dict):
        pass

    def is_typed_registration_needed(self, type_in_registration):
        '''
        Given the string from a kernel registration, determine if the registration is required or not.
        :param type_in_registration: Type string from kernel registration
        :return: True is required. False if not.
        '''
        # Not all operators have typed registrations, so this is optionally implemented by derived classes
        raise RuntimeError('Did not expect processor for {} to have typed registrations.'.format(self._name))

    @abstractmethod
    def write_cpp_defines(self, outfile):
        # write the type info for this operator to the output C++ header file
        pass

    @abstractmethod
    def to_config_entry(self):
        pass

    @abstractmethod
    def from_config_entry(self, entry: str):
        pass


class DefaultOperatorProcessor(OperatorProcessor):
    '''
    Operator processor which tracks the types used for selected input/s and/or output/s.
    '''

    def __init__(self, domain: str, optype: str, inputs: [int] = [0], outputs: [int] = []):
        super().__init__(domain, optype)
        self._input_types = {}
        self._output_types = {}

        for i in inputs:
            self._input_types[i] = set()

        for o in outputs:
            self._output_types[o] = set()

    def process(self, node: fbs.Node, nodearg_info: dict):
        for i in self._input_types.keys():
            if i >= node.InputsLength():
                raise RuntimeError('Node has {} inputs. Tracker for {} incorrectly configured as it requires {}.'
                                   .format(node.InputsLength(), self.name(), i))

            type_str = name_to_typestr(node.Inputs(i), nodearg_info)
            self._input_types[i].add(type_str)

        for o in self._output_types.keys():
            if o >= node.OutputsLength():
                raise RuntimeError('Node has {} outputs. Tracker for {} incorrectly configured as it requires {}.'
                                   .format(node.OutputsLength(), self.name(), o))

            type_str = name_to_typestr(node.Outputs(o), nodearg_info)
            self._output_types[o].add(type_str)

    def is_typed_registration_needed(self, type_in_registration):
        if 0 not in self._input_types.keys():
            raise RuntimeError('Expected typed registration to be done using type from input 0.')

        return type_in_registration in self._input_types[0]

    def write_cpp_defines(self, outfile):
        for i in self._input_types.keys():
            if self._input_types[i]:
                outfile.write('#define {}_INPUT{}_TYPES std::tuple<{}>\n'
                              .format(self.cpp_name(), i, ','.join(sorted(self._input_types[i]))))

        for o in self._output_types.keys():
            if self._output_types[o]:
                outfile.write('#define {}_OUTPUT{}_TYPES std::tuple<{}>\n'
                              .format(self.cpp_name(), o, ','.join(sorted(self._output_types[o]))))

    def to_config_entry(self):
        aggregate_info = {'inputs': {}, 'outputs': {}}

        # filter out empty entries
        for i in self._input_types.keys():
            if self._input_types[i]:
                aggregate_info['inputs'][i] = sorted(self._input_types[i])

        for o in self._output_types.keys():
            if self._output_types[o]:
                aggregate_info['outputs'][o] = sorted(self._output_types[o])

        if not aggregate_info['inputs']:
            aggregate_info.pop('inputs')
        if not aggregate_info['outputs']:
            aggregate_info.pop('outputs')

        entry = json.dumps(aggregate_info) if aggregate_info else None
        return entry

    def from_config_entry(self, entry: str):
        self._input_types.clear()
        self._output_types.clear()

        aggregate_info = json.loads(entry)
        if 'inputs' in aggregate_info:
            for i_str, values in aggregate_info['inputs'].items():
                self._input_types[int(i_str)] = set(values)

        if 'outputs' in aggregate_info:
            for o_str, values in aggregate_info['outputs'].items():
                self._output_types[int(o_str)] = set(values)


class OneHotProcessor(OperatorProcessor):
    'Processor for the OneHot operator'
    def __init__(self):
        super().__init__('ai.onnx', 'OneHot')
        self._triples = set()

    def process(self, node: fbs.Node, nodearg_info: dict):
        type0 = name_to_typestr(node.Inputs(0), nodearg_info)
        type1 = name_to_typestr(node.Inputs(1), nodearg_info)
        type2 = name_to_typestr(node.Inputs(2), nodearg_info)
        key = '{}_{}_{}'.format(type0, type1, type2)
        self._triples.add(key)

    def is_typed_registration_needed(self, type_in_registration):
        # the OneHot registration creates a triple from the 3 types involved
        return type_in_registration in self._triples

    def write_cpp_defines(self, outfile):
        # exclusion via registration so don't need to write any #defines
        pass

    def to_config_entry(self):
        if not self._triples:
            return None

        aggregate_info = {'custom': sorted(self._triples)}
        entry = json.dumps(aggregate_info)

    def from_config_entry(self, entry: str):
        self._triples.clear()
        aggregate_info = json.loads(entry)
        if 'custom' in aggregate_info:
            self._triples = set(aggregate_info['custom'])


class OrtFormatModelProcessor:

    def __init__(self, model_path: os.path.realpath, required_ops: dict,  processors: dict):
        self._required_ops = required_ops  # dictionary of {domain: {opset:[operators]}}
        self._file = open(model_path, 'rb').read()
        self._buffer = bytearray(self._file)
        self._model = fbs.InferenceSession.InferenceSession.GetRootAsInferenceSession(self._buffer, 0).Model()
        self._processors = processors

    @staticmethod
    def _setup_node_args(graph: fbs.Graph, existing_node_args={}):
        '''
        Setup the node args for this level of Graph.
        We copy the current list which represents the valid outer scope values, and add the local node args to that
        to create the valid list for the current Graph.
        :param graph: Graph to create NodeArg list for
        :param existing_node_args: Outer scope NodeArgs. Empty for the top-level graph in a model.
        :return: Dictionary of NodeArg name to TypeInfo
        '''
        nodearg_info = existing_node_args.copy()
        for j in range(0, graph.NodeArgsLength()):
            n = graph.NodeArgs(j)
            nodearg_info[n.Name()] = n.Type()  # TypeInfo for this NodeArg

        return nodearg_info

    def _add_required_op(self, domain: str, opset: int, op_type: str):
        if domain not in self._required_ops:
            self._required_ops[domain] = {opset: set([op_type])}
        elif opset not in self._required_ops[domain]:
            self._required_ops[domain][opset] = set([op_type])
        else:
            self._required_ops[domain][opset].add(op_type)

    def _process_graph(self, graph: fbs.Graph, outer_scope_nodeargs: dict):
        '''
        Process one level of the Graph, descending into any subgraphs when they are found
        :param ancestor_nodeargs: Outer scope NodeArg dictionary from ancestor graphs
        '''

        # print(dir(graph.Nodes(0)))

        nodeargs = OrtFormatModelProcessor._setup_node_args(graph, outer_scope_nodeargs)

        for i in range(0, graph.NodesLength()):
            node = graph.Nodes(i)

            optype = node.OpType().decode()
            domain = node.Domain().decode()
            if not domain:
                # empty domain defaults to ai.onnx
                domain = 'ai.onnx'

            full_name = '{}:{}'.format(domain, optype)
            if full_name in self._processors:
                self._processors[full_name].process(node, nodeargs)

            self._add_required_op(domain, node.SinceVersion(), optype)

            # Read all the attributes
            for j in range(0, node.AttributesLength()):
                attr = node.Attributes(j)
                attr_type = attr.Type()
                if attr_type == fbs.AttributeType.AttributeType.GRAPH:
                    self._process_graph(attr.G(), nodeargs)
                elif attr_type == fbs.AttributeType.AttributeType.GRAPHS:
                    for k in range(0, attr.GraphsLength()):
                        self._process_graph(attr.Graphs(k), nodeargs)

    def process(self):
        graph = self._model.Graph()
        outer_scope_nodeargs = {}
        self._process_graph(graph, outer_scope_nodeargs)

    def get_required_ops(self):
        return self._required_ops


def create_operator_processors():
    operator_processors = {}

    # Office
    # # Generated from --model_path C:\Users\Joshua\AppData\Local\Temp\tmp7p5eloyh
    # com.microsoft;1;FusedConv,FusedGemm,QLinearAdd
    # ai.onnx;12;Add,BatchNormalization,Cast,Concat,Conv,DequantizeLinear,Flatten,Gemm,Max,Min,
    #            QLinearConv,QuantizeLinear,Relu,Resize,Softmax,Transpose

    # Starting with ops in the Office production models, as well as some known large kernels
    default_processor_onnx_ops = ['Add', 'BatchNormalization', 'Concat', 'Conv', 'DequantizeLinear', 'Expand'
                                  'Flatten', 'Gemm', 'Max', 'Min', 'QLinearConv',
                                  'Relu', 'Resize', 'Softmax', 'Transpose', 'Upsample']

    # ML Op notes. Also need to figure out how we'll setup the type strings for Map and Sequence
    #  CastMap: Switch on value type of input map type, and output type
    #  DictVectorizer: Templatized on key+value of input so need to handle like OneHot with custom processor
    #  LabelEncoder: Implementation switches on input and output types (only supports string and int64 in T1 and T2)
    #  LinearClassifier: Internal switch on input type and also switch on output type
    #  SVMClassifier: ditto
    #  TreeEnsembleClassifier: Templatized on input type and also switch on output type
    #  ZipMap: Switch on output type (derived from attributes)
    default_processor_onnxml_ops = []  # TODO - review and add ML ops as needed

    # FusedConv and FusedGemm are float only so can be ignored
    internal_ops = ['QLinearAdd', 'QLinearMul']

    def add(processor):
        operator_processors[processor.name()] = processor

    [add(DefaultOperatorProcessor('ai.onnx', op)) for op in default_processor_onnx_ops]
    [add(DefaultOperatorProcessor('ai.onnx.ml', op)) for op in default_processor_onnxml_ops]
    [add(DefaultOperatorProcessor('com.microsoft', op)) for op in internal_ops]

    #
    # Operators that require slightly different handling
    #
    add(DefaultOperatorProcessor('ai.onnx', 'Cast', inputs=[0], outputs=[0]))  # track input0 and output0

    # Gather and GatherElements have switching on both the data type (input0) and indices type (input1)
    add(DefaultOperatorProcessor('ai.onnx', 'Gather', inputs=[0, 1]))
    add(DefaultOperatorProcessor('ai.onnx', 'GatherElements', inputs=[0, 1]))

    # Pow dispatches on base and exponential types
    add(DefaultOperatorProcessor('ai.onnx', 'Pow', inputs=[0, 1]))

    # Random generator ops produce new data so we track the output type
    onnx_random_ops = ['RandomNormal', 'RandomNoarmalLike', 'RandomUniform', 'RandomUniformLike', 'Multinomial']
    [add(DefaultOperatorProcessor('ai.onnx', op, inputs=[], outputs=[0])) for op in onnx_random_ops]

    # we only support 'float' as input for QuantizeLinear so just track the output type
    add(DefaultOperatorProcessor('ai.onnx', 'QuantizeLinear', inputs=[], outputs=[0]))

    # OneHot concatenates types into a triple in the typed registration
    add(OneHotProcessor())

    return operator_processors


def main():

    filename = '../../test/testdata/ort_github_issue_4031.onnx.ort'
    required_ops = {}  # just processing one model currently so this isn't re-used across ModelProcessor instances yet
    operator_processors = create_operator_processors()
    model_processor = OrtFormatModelProcessor(filename, required_ops, operator_processors)
    model_processor.process()

    print(model_processor.get_required_ops())

    # TODO: At this point we have all the required ops and a collection of processors that can add type reductions
    # We need to a) use the type reduction information when excluding unused kernels; and b) generate a header
    # file with the necessary #defines.
    for key in sorted(operator_processors.keys()):
        operator_processors[key].write_cpp_defines(sys.stdout)

    # Start with the header file generation and then update/replace exclude_unused_ops to handle type reductions
    for key in sorted(operator_processors.keys()):
        entry = operator_processors[key].to_config_entry()
        if entry:
            operator_processors[key].from_config_entry(entry)


if __name__ == "__main__":
    main()
