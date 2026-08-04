# MIT License

# Copyright (c) 2020 Joseph Auckley, Matthew O'Kelly, Aman Sinha, Hongrui Zheng

# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:

# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.

# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.



"""
Conversion from x-axis pointing up (in map img), 0 to pi c.c.w., and 0 to -pi c.w. convention to x-axis pointing right (im map img), 0 to 2pi c.c.w. convention.
Use either on one csv file or all csv file in a directory
Author: Hongrui Zheng
"""

import numpy as np
import pandas as pd
import glob
import csv
import os
from pathlib import Path

from config import load_racetrack_config


def main():
    module = Path(__file__).resolve().parent
    config = load_racetrack_config().conversion

    all_files = glob.glob(str(module / config.pattern))
    print('Converting following files:')
    for name in all_files:
        print(name)
    if config.require_confirmation:
        input('Press ENTER to proceed, CTRL+C to stop.')

    for file in all_files:
        file_name, file_ext = os.path.splitext(file)
        new_file = file_name + config.output_suffix + file_ext
        print('Working on: ' + file)

        with open(file) as stream:
            headers = list(csv.reader(stream))[0:3]
        df = pd.read_csv(file, sep=';', header=2)
        heading_np = df[' psi_rad'].to_numpy()
        heading_np += np.pi / 2
        heading_np[heading_np > 2 * np.pi] -= 2 * np.pi
        heading_np[heading_np < 0] += 2 * np.pi
        df[' psi_rad'] = heading_np

        with open(new_file, 'w', newline='') as stream:
            csv.writer(stream).writerows(headers)
        df.to_csv(
            new_file, sep=';', header=False, index=False,
            float_format='%.7f', mode='a'
        )
        print('New convention saved to: ' + new_file)

    print('All files done.')


if __name__ == '__main__':
    main()
