#!/usr/bin/env python3
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.

import argparse

from util.ort_format_model import create_config_from_models


def main():
    argparser = argparse.ArgumentParser('Script to create a build reduction config file from ORT format model/s.')
    argparser.add_argument('-t', '--enable_type_reduction', action='store_true',
                           help='Enable tracking of the specific types that individual operators require. '
                                'Operator implementations MAY support limiting the type support included in the build '
                                'to these types.')
    argparser.add_argument('model_path_or_dir', type=str,
                           help='Path to a single ORT format model, or a directory containing ORT format models '
                                'that will be recursively processed.')

    argparser.add_argument('config_path', type=str,
                           help='Path to write configuration file to.')

    args = argparser.parse_args()

    create_config_from_models(args.model_path_or_dir, args.config_path, args.enable_type_reduction)

    # temporary test
    from util import parse_config
    required_ops, op_type_usage_processor = parse_config(args.config_path)

    print(required_ops)
    op_type_usage_processor.debug_dump()


if __name__ == "__main__":
    main()
