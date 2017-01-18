'''
This program plots the time series for a single sensor across multiple sample
sets. Its highly useful for finding sensors that have high seperation ability.
Specifying samples to plot is highly recommended; showing everything will
produce a nearly unreadable plot
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
import itertools
import matplotlib.pyplot as plt
import inputparser

if len(sys.argv) < 6:
    print ('USAGE: SamplePlotter.py (Path to data file) (Path to results '
           'file) (sensor results to plot) (Remove background data from '
           'samples?) (Normalize values?) [samples to plot]')
    sys.exit(1)

# The first column in the data is a time stamp, so the sensor columns are
# numbered starting at 1. Subtract 1 to get the index into the processed data
# Extract the sensor to use early, so errors cause a quick exit
wanted_sensor = int(sys.argv[3]) - 1

# For the two yes/no arguments, take 0, anything stating with 'n' and anything
# starting with 'f' as False, everything else as True.
remove_background = sys.argv[4] != '0' and (sys.argv[4][0] not in "fFnN")
normalize_data = sys.argv[5] != '0' and (sys.argv[5][0] not in "fFnN")

sample_set = inputparser.load_sample_data(sys.argv[1], sys.argv[2],
                                          remove_background, normalize_data)

if (wanted_sensor < 0) or (wanted_sensor >= sample_set[0][0].shape[1]):
    print ('Wanted sensor {} invalid, range is 1 to {}'.
           format(wanted_sensor + 1, sample_set[0][0].shape[1]))
    sys.exit(1)

if len(sys.argv) > 6:
    # Convert the arguments to integers. Exceptions are passed through.
    wanted_samples = [int(value) for value in sys.argv[6:]]

    # Since sample numbers are not consecutive, need to translate the sample
    # numbers into indexes. A dictionary handles this nicely
    sample_to_index = {}
    for index, sample in enumerate(sample_set):
        sample_to_index[sample[2]] = index
    sample_list = [sample_to_index[sample] for sample in wanted_samples]

    # Extract. Want a list for easy processing
    sample_set = [sample_set[index] for index in sample_list]

# To do the plot, need a two dimensional array with the time series as the Y
# axis and the sample numbers as the X axis. To do this, get the transpose of
# each sample set (so the time series is the Y axis), extract the wanted sample
# row, and put these in a list. Finally convert that to a ndarray.
wanted_samples = [sample[0].T[wanted_sensor] for sample in sample_set]

# To do the labels, exploit the fact that the wanted samples and the sample set
# have the same order
for sample, labels in itertools.izip(wanted_samples, sample_set):
    plt.plot(sample, label=str(labels[2]) + ": " + labels[1])
plt.legend()
plt.title("Time series of sensor " + str(wanted_sensor + 1) + " across samples")
plt.show()
