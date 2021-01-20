# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.

import os
import sys

from .ort_format_model import OperatorTypeUsageProcessors


def parse_config(config_file: str):
    '''
    Parse the configuration file and return the required operators dictionary and an OperatorTypeUsageProcessor if
    the config contains type info
    :param config_file: Configuration file to parse
    :return:
    '''

    if not os.path.isfile(config_file):
        raise ValueError('Configuration file {} does not exist'.format(config_file))

    required_ops = {}
    op_type_usage_processor = OperatorTypeUsageProcessors()

    # import map of domain string to the C++ constant name used for the kernel registrations
    script_path = os.path.dirname(os.path.realpath(__file__))
    ci_build_py_path = os.path.abspath(os.path.join(script_path, '..', '..', 'ci_build'))
    sys.path.append(ci_build_py_path)
    import op_registration_utils
    domain_to_constant = op_registration_utils.domain_map

    with open(config_file, 'r') as config:
        for line in [orig_line.strip() for orig_line in config.readlines()]:
            if not line or line.startswith("#"):  # skip empty lines and comments
                continue

            domain, opset_str, operators_str = [segment.strip() for segment in line.split(';')]

            if domain not in domain_to_constant:
                raise ValueError('Unexpected domain. Please add handling to script for ' + domain)

            domain_constant = domain_to_constant[domain]
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

                    if next_open_brace > 0 and next_open_brace < next_comma:
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
                        op_type_usage_processor.update_using_config_entry(domain, operator, type_str)
                        cur = i + 1
                    else:
                        # comma or end is next
                        end_str = next_comma if next_comma != -1 else end
                        operators.add(operators_str[cur:end_str].strip())
                        cur = end_str + 1

            else:
                operators = set([op.strip() for op in operators_str.split(',')])

            if domain_constant not in required_ops:
                required_ops[domain_constant] = {opset: operators}
            elif opset not in required_ops[domain_constant]:
                required_ops[domain_constant][opset] = operators
            else:
                required_ops[domain_constant][opset].update(operators)

    return required_ops, op_type_usage_processor
