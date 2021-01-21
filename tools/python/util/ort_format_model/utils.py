# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.

import os

from .operator_type_usage_processors import OperatorTypeUsageManager
from .ort_model_processor import OrtFormatModelProcessor

from ..logger import get_logger
log = get_logger("ort_format_model.utils")


def _extract_ops_and_types_from_ort_models(model_path_or_dir: str, enable_type_reduction: bool):
    if not os.path.exists(model_path_or_dir):
        raise ValueError('Path to model/s does not exist: {}'.format(model_path_or_dir))

    required_ops = {}
    op_type_usage_manager = OperatorTypeUsageManager() if enable_type_reduction else None

    if os.path.isfile(model_path_or_dir):
        model_processor = OrtFormatModelProcessor(model_path_or_dir, required_ops, op_type_usage_manager)
        model_processor.process()  # this updates required_ops and op_type_processors
        log.info('Processed {}'.format(model_path_or_dir))
    else:
        for root, _, files in os.walk(model_path_or_dir):
            for file in files:
                model_path = os.path.join(root, file)
                if file.lower().endswith('.ort'):
                    model_processor = OrtFormatModelProcessor(model_path, required_ops, op_type_usage_manager)
                    model_processor.process()  # this updates required_ops and op_type_processors
                    log.info('Processed {}'.format(model_path))

    return required_ops, op_type_usage_manager


def create_config_from_models(model_path: str, output_file: str, enable_type_reduction: bool = True):

    required_ops, op_type_processors = _extract_ops_and_types_from_ort_models(model_path, enable_type_reduction)

    directory, filename = os.path.split(output_file)
    if not filename:
        raise RuntimeError("Invalid output path for configuation: {}".format(output_file))

    if not os.path.exists(directory):
        os.makedirs(directory)

    with open(output_file, 'w') as out:
        out.write("# Generated from models in {}\n".format(model_path))

        for domain in sorted(required_ops.keys()):
            for opset in sorted(required_ops[domain].keys()):
                ops = required_ops[domain][opset]
                if ops:
                    out.write("{};{};".format(domain, opset))
                    if enable_type_reduction:
                        # type string is empty if op hasn't been seen
                        entries = ['{}{}'.format(op, op_type_processors.get_config_entry(domain, op) or '')
                                   for op in sorted(ops)]
                    else:
                        entries = sorted(ops)

                    out.write("{}\n".format(','.join(entries)))

    log.info("Created config in %s", output_file)
