# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.

import os
from .ort_format_model import OperatorTypeUsageManager


def parse_config(config_file: str):
    '''
    Parse the configuration file and return the required operators dictionary and an OperatorTypeUsageManager.
    The basic configuration file format is `domain;opset;op1,op2...`
    e.g. `ai.onnx;11;Add,Cast,Clip,...

    If the configuration file is generated from ORT format models it may optionally contain JSON for per-operator
    type reduction. TThe required types are generally listed per input and/or output of the operator.
    The type information is in a map, with 'inputs' and 'outputs' keys. The value for 'inputs' or 'outputs' is a map
    between the index number of the input/output and the required list of types.

    For example, both the input and output types are relevant to ai.onnx:Cast.
    Type information for input 0 and output 0 could look like this:
        `{"inputs": {"0": ["float", "int32_t"]}, "outputs": {"0": ["float", "int64_t"]}}`

    which is added directly after the operator name in the configuration file.
    e.g.
        `ai.onnx;12;Add,Cast{"inputs": {"0": ["float", "int32_t"]}, "outputs": {"0": ["float", "int64_t"]}},Concat`

    If for example the types of inputs 0 and 1 were important, the entry may look like this (e.g. ai.onnx:Gather):
        `{"inputs": {"0": ["float", "int32_t"], "1": ["int32_t"]}}`

    Finally some operators do non-standard things and store their type information under a 'custom' key.
    ai.onnx.OneHot is an example of this, where 3 type names from the inputs are combined into a string.
        `{"custom": ["float_int64_t_int64_t", "int64_t_string_int64_t"]}`

    :param config_file: Configuration file to parse
    :return: required_ops, op_type_usage_manager:
             Dictionary of domain:opset:[ops] for required operators
             OperatorTypeUsageManager manager with operator specific type usage information if available.
    '''

    if not os.path.isfile(config_file):
        raise ValueError('Configuration file {} does not exist'.format(config_file))

    required_ops = {}
    op_type_usage_manager = OperatorTypeUsageManager()

    with open(config_file, 'r') as config:
        for line in [orig_line.strip() for orig_line in config.readlines()]:
            if not line or line.startswith("#"):  # skip empty lines and comments
                continue

            domain, opset_str, operators_str = [segment.strip() for segment in line.split(';')]
            opset = int(opset_str)

            # any type reduction information is serialized json that starts/ends with { and }
            if '{' in operators_str:
                # parse individual entries in the line. type info is optional for each operator.
                operators = set()
                cur = 0
                end = len(operators_str)
                while cur < end:
                    next_comma = operators_str.find(',', cur)
                    next_open_brace = operators_str.find('{', cur)

                    # the json string starts with '{', so if that is found (next_open_brace != -1)
                    # before the next comma we have type info to parse
                    if 0 < next_open_brace < next_comma:
                        operator = operators_str[cur:next_open_brace].strip()
                        operators.add(operator)

                        # parse out the json dictionary with the type info
                        i = next_open_brace + 1
                        num_open_braces = 1
                        while num_open_braces > 0 and i < end:
                            if operators_str[i] == '{':
                                num_open_braces += 1
                            elif operators_str[i] == '}':
                                num_open_braces -= 1
                            i += 1

                        if num_open_braces != 0:
                            raise RuntimeError('Mismatched { and } in type string: ' + operators_str[next_open_brace:])

                        type_str = operators_str[next_open_brace:i]
                        op_type_usage_manager.restore_from_config_entry(domain, operator, type_str)
                        cur = i + 1
                    else:
                        # comma or end is next
                        end_str = next_comma if next_comma != -1 else end
                        operators.add(operators_str[cur:end_str].strip())
                        cur = end_str + 1

            else:
                operators = set([op.strip() for op in operators_str.split(',')])

            if domain not in required_ops:
                required_ops[domain] = {opset: operators}
            elif opset not in required_ops[domain]:
                required_ops[domain][opset] = operators
            else:
                required_ops[domain][opset].update(operators)

    return required_ops, op_type_usage_manager
