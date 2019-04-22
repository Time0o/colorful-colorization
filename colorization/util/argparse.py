import argparse


class NiceHelpFormatter(argparse.HelpFormatter):
    def __init__(self, prog, max_help_position):
        super().__init__(prog, max_help_position=max_help_position)

    def _split_lines(self, text, width):
        return super()._split_lines(text, width) + ['']


def nice_help_formatter(max_help_position=40):
    return lambda prog: NiceHelpFormatter(prog, max_help_position)


