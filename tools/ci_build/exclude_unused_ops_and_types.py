# !/usr/bin/env python3
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.

import argparse
import op_registration_utils
import os
import shutil
import sys
import typing

from logger import get_logger
log = get_logger("exclude_unused_ops_and_types")


class ExcludeOpsAndTypesRegistrationProcessor(op_registration_utils.RegistrationProcessor):
    def __init__(self, required_ops, op_type_usage_processor, output_file):
        self._required_ops = required_ops
        self._op_types_usage_processor = op_type_usage_processor
        self._output_file = output_file

    def _should_exclude_op(self, domain, operator, start_version, end_version):
        if domain not in self.required_ops:
            return True

        for opset in self.required_ops[domain]:
            if opset >= start_version and (end_version is None or opset <= end_version):
                if operator in self.required_ops[domain][opset]:
                    return False  # found a match, do not exclude

        return True

    def process_registration(self, lines: typing.List[str], domain: str, operator: str,
                             start_version: int, end_version: int = None, input_type: str = None):
        exclude = self._should_exclude_op(domain, operator, start_version, end_version)
        if exclude:
            log.info('Disabling {}:{}({})'.format(domain, operator, start_version))
            for line in lines:
                self.output_file.write('// ' + line)

            # edge case of last entry in table where we still need the terminating }; to not be commented out
            if lines[-1].rstrip().endswith('};'):
                self.output_file.write('};\n')
        else:
            for line in lines:
                self.output_file.write(line)

    def process_other_line(self, line):
        self.output_file.write(line)

    def ok(self):
        return True


def _exclude_unused_ops_and_types_in_registrations(required_operators,
                                                   op_type_usage_processor,
                                                   provider_registration_paths):
    '''rewrite provider registration file to exclude unused ops'''

    for kernel_registration_file in provider_registration_paths:
        if not os.path.isfile(kernel_registration_file):
            log.warning('Kernel registration file {} does not exist'.format(kernel_registration_file))
            return

        log.info("Processing {}".format(kernel_registration_file))

        backup_path = kernel_registration_file + '~'
        shutil.move(kernel_registration_file, backup_path)

        # read from backup and overwrite original with commented out lines for any kernels that are not required
        with open(kernel_registration_file, 'w') as file_to_write:
            processor = ExcludeOpsAndTypesRegistrationProcessor(required_operators, op_type_usage_processor,
                                                                file_to_write)

            op_registration_utils.process_kernel_registration_file(backup_path, processor)

            if not processor.ok():
                # error should have already been logged so just exit
                sys.exit(-1)


def _generate_cpp_defines(ort_root, op_type_usage_processor):

    defines = op_type_usage_processor.get_cpp_defines()
    if not defines:
        return

    # open header file to write
    type_reduction_header_path = os.path.join(ort_root, 'onnxruntime', 'core', 'framework', 'type_reductions.h')
    with open(type_reduction_header_path, 'w') as output:
        output.write('// Copyright (c) Microsoft Corporation. All rights reserved.\n')
        output.write('// Licensed under the MIT License.\n\n')
        output.write('#pragma once\n\n')

        [output.write('{}\n'.format(define) for define in defines)]

    # future: how/where will we write global type limitations?
    # should they come from the ops file or be separate? probably separate - may want to reduce types without
    # reducing operators


def exclude_unused_ops_and_types(config_path, use_cuda=True):
    script_path = os.path.dirname(os.path.realpath(__file__))
    ort_tools_py_path = os.path.abspath(os.path.join(script_path, '..', 'tools'))
    ort_root = os.path.abspath(os.path.join(script_path, '..', '..', ))
    sys.path.append(ort_tools_py_path)
    from util import parse_config

    required_ops, op_type_usage_processor = parse_config(config_path)

    registration_files = op_registration_utils.get_kernel_registration_files(ort_root, use_cuda)
    _exclude_unused_ops_and_types_in_registrations(required_ops, op_type_usage_processor, registration_files)

    _generate_cpp_defines(ort_root, op_type_usage_processor)


if __name__ == "__main__":

    parser = argparse.ArgumentParser(
        description="Script to exclude unused operator kernels by disabling their registration in ONNXRuntime. "
                    "The types supported by operator kernels may also be reduced if specified in the config file.")

    parser.add_argument("--config_path", type=str, required=True,
                        help="Path to configuration file with format of 'domain;opset;op1,op2...'")

    args = parser.parse_args()
    config_path = os.path.abspath(args.config_path)

    exclude_unused_ops_and_types(config_path, use_cuda=True)
