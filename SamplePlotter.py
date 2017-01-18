''' 
This program plots sensor data for a single sample time series. This aids in
choosing which sensors to use to train the classification neural network
'''

#
#   Copyright (C) 2017   Ezra Erb
#
#   This program is free software: you can redistribute it and/or modify
#   it under the terms of the GNU General Public License version 3 as published
#   by the Free Software Foundation.
#
#   This program is distributed in the hope that it will be useful,
#   but WITHOUT ANY WARRANTY; without even the implied warranty of
#   MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
#   GNU General Public License for more details.
#
#   You should have received a copy of the GNU General Public License
#   along with this program.  If not, see <http://www.gnu.org/licenses/>.
#
#   I'd appreciate a note if you find this program useful or make
#   updates. Please contact me through LinkedIn or github (my profile also has
#   a link to the code depository)
#

import sys
from operator import itemgetter
import matplotlib.pyplot as plt
import inputparser

if len(sys.argv) < 5:
    print ('USAGE: SamplePlotter.py (Path to data file) (Path to results file) '
           '(Remove background ata from samples?) (Normalize values?) [samples '
           'to plot]')
    sys.exit(1)

# For the two yes/no arguments, take 0, anything stating with 'n' and anything
# starting with 'f' as False, everything else as True.
remove_background = sys.argv[3] != '0' and (sys.argv[3][0] not in "fFnN")
normalize_data = sys.argv[4] != '0' and (sys.argv[4][0] not in "fFnN")

sample_set = inputparser.load_sample_data(sys.argv[1], sys.argv[2],
                                          remove_background, normalize_data)
if len(sys.argv) > 5:
    # Convert the arguments to integers. Exceptions are passed through.
    wanted_samples = [int(value) for value in sys.argv[5:]]

    # Since sample numbers are not consecutive, need to translate the sample
    # numbers into indexes. A dictionary handles this nicely
    sample_to_index = {}
    for index, sample in enumerate(sample_set):
        sample_to_index[sample[2]] = index
    sample_list = [sample_to_index[sample] for sample in wanted_samples]
else:
    # Everything
    sample_list = range(0, len(sample_set))

for sample in itemgetter(*sample_list)(sample_set):
    plt.plot(sample[0])
    plt.title("Sample " + str(sample[2]) + ": " + sample[1])
    plt.show()
