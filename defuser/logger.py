# SPDX-FileCopyrightText: 2026 ModelCloud.ai
# SPDX-FileCopyrightText: 2026 qubitium@modelcloud.ai
# SPDX-License-Identifier: Apache-2.0
# Contact: qubitium@modelcloud.ai, x.com/qubitium

# Adapted from intel/auto-round
# at https://github.com/intel/auto-round/blob/main/auto_round/logger.py

import logging

logger = logging.getLogger("defuser")

# Define a new logging level TRACE
TRACE_LEVEL = 5
logging.addLevelName(TRACE_LEVEL, "TRACE")


def trace(self, message, *args):
    """
    Log a message with the TRACE level.

    Args:
        message: The message format string
        *args: Variable positional arguments for message formatting

    """
    if self.isEnabledFor(TRACE_LEVEL):
        self._log(TRACE_LEVEL, message, args, stacklevel=2)


# Add the trace method to the Logger class

logging.Logger.trace = trace