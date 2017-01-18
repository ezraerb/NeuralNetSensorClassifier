''' 
Parse files of input sensor data and optionally normalize the data across
samples. This data is returned as a list of NDarrays paired with category
results. This data will be either plotted or used to train a neural network.
Network training requires normalized data
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

import csv
import numpy as np

def normalize_data(sample_set, remove_background=True, normalize=True):
    ''' 
    Convert raw data from the sensors into normalized data for a neural
    network by removing background data and normalize readings to a range of
    0..1  Both are required for efficient neural nets. If the data is dirty and
    some samples do not have enough readings, they will be dropped from the
    set. Background removal and normalization can be individually skipped,
    which is very useful for manual analysis of plots 
    '''

    # Data is stored in a NDArray, with time series as the first axis and
    # sensors as the second.
    if not remove_background:
        # In the existing array, the first column is the timestamps. Remove
        # this column with no other manipulation
        for sample in sample_set:
            sample[0] = np.delete(sample[0], 0, 1)
    else:
        # Adapted from: http://stackoverflow.com/questions/8312829/how-to-remove-item-from-a-python-list-if-a-condition-is-true
        keep_index = 0
        for index, sample in enumerate(sample_set):
            # The first and last hour worth of samples are background, the
            # remainder are situation sensor data. The background data is
            # resonably constant within the sensed period. Accordingly, this
            # code uses the difference to the background as input

            # Unfortunately, the sample interval was not uniform, so have to
            # calculate the split between background and situation from the time
            # stamps. This is the first column in the data
            time_stamps = sample[0].T[0]
            start_pos = np.searchsorted(time_stamps, 0.0)
            end_pos = np.searchsorted(time_stamps, time_stamps[-1] - 1.0)

            # Some samples are dirty and do not last for the required amount.
            # Ignore them in this case
            if start_pos < end_pos:
                split_data = np.split(sample[0], [start_pos, end_pos])
                background = np.concatenate((split_data[0], split_data[2]))

                # At this point the timesamps are no longer needed. Drop the
                # column
                background = np.delete(background, 0, 1)
                normalized_data = np.delete(split_data[1], 0, 1)
                adjustment = background.mean(axis=0)
                normalized_data -= adjustment

                sample[0] = normalized_data
                if keep_index < index:
                    sample_set[keep_index] = sample
                keep_index += 1

        if keep_index < len(sample_set):
            del sample_set[keep_index:]

    # Now normalize to a scale of 0..1 column by column
    if normalize:
        # Find the minimum and maximum values across the sample set
        min_values = sample_set[0][0].min(axis=0)
        max_values = sample_set[0][0].max(axis=0)
        for sample in sample_set:
            min_values = np.minimum(min_values, sample[0].min(axis=0))
            max_values = np.maximum(max_values, sample[0].max(axis=0))

        # Now nomalize
        for sample in sample_set:
            sample[0] -= min_values
            sample[0] /= (max_values - min_values)

            # If any value is NaN, it means that every value in a column was
            # the same. This is very unlikely in practice. Since the values
            # are normalized, convert these values to zero
            sample[0][np.isnan(sample[0])] = 0.0

def load_sample_data(sample_file_name, result_file_name, remove_background=True, normalize=True):
    ''' 
    Load sample data. Output is a list of tupples containing the samples
    and categories. Samples are stored in a multi-dimension array with time
    series as rows and the sensors as columns. The samples and results are
    stored in seperate files in the results data, hence the need for two file
    names 
    '''

    sample_set = []
    # First, load the results into a local array, indexed by sample ID. They
    # are contiguous starting at zero, so an array will do.

    expected_results = []
    with open(result_file_name, 'rt') as fin:
        cfin = csv.reader(fin, delimiter='\t')
        # First line is a header. Burn it
        next(cfin)
        current_sample_id = -1 # Token value
        for mrow in cfin:
            new_sample_id = int(mrow[0])
            if (current_sample_id != -1) and (current_sample_id + 1 != new_sample_id):
                print ('WARNING: Sample categories file not in inceasing '
                       'order. Have {}, expected {}. Data files likely '
                       'corrupt'.format(new_sample_id, current_sample_id + 1))
                return sample_set
            expected_results.append(mrow[2])
            current_sample_id = new_sample_id

    # Now load the actual samples
    with open(sample_file_name, 'rt') as fin:
        # Columns in the file are seperated by double spaces, requiring the
        # following
        # http://stackoverflow.com/questions/6352409/how-to-use-python-csv-module-for-splitting-double-pipe-delimited-data
        cfin = csv.reader((line.replace('  ', ' ') for line in fin), delimiter=' ')
        # first line is a header. Burn it
        next(cfin)
        current_sample_id = -1 # Token value
        sample_data = []
        for mrow in cfin:
            # Lines have extra spaces which generate bad entries. Slice them
            # away. The first entry is the sample ID. Convert it to an integer.
            # Convert the remainder to floats.
            new_sample_id = int(mrow[0])
            samples = [float(i) for i in mrow[1:12]]
            if new_sample_id != current_sample_id:
                # next batch of samples
                # The file is organized in inceasing order by sample ID. If
                # this assumption breaks, data load becomes harder. Verify it.
                if current_sample_id != -1:
                    if (current_sample_id + 1) != new_sample_id:
                        print ('WARNING: File samples are not in increasing '
                               'order. Have {}, expected {}. Data likely '
                               'invalid!'.format(new_sample_id, current_sample_id + 1))
                        sample_set = []
                        return sample_set
                    # If read a sample for which there is no result data, also
                    # have corrupted data
                    elif new_sample_id >= len(expected_results):
                        print ('WARNING: Sample id read {} for which no '
                               'expected results exist. Data likely '
                               'invalid'.format(new_sample_id))
                        sample_set = []
                        return sample_set
                    # Note the extra pair of parentheses to create a list
                    sample_set.append([np.array(sample_data),
                                       expected_results[current_sample_id],
                                       current_sample_id])
                sample_data = []
                current_sample_id = new_sample_id
            sample_data.append(samples)
        # Unless the file is empty, have a final set of samples to process
        # Note the brakcets to create a list
        sample_set.append([np.array(sample_data),
                           expected_results[current_sample_id],
                           current_sample_id])
    # Remove background data and normalize readings to a range of 0..1  Both are
    # required for efficient neural nets.
    normalize_data(sample_set, remove_background, normalize)
    return sample_set
