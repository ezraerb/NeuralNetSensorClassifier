'''
Sensor data classification using a basic neural network. The network is
explictly implemented instead of using a package to learn the algorithm. The
input is two files, one of sensor data and one of the category each belongs in.
A sample of the inputs is used to train the network and then the rest are
classified. Afterwards, the performance is analyzed using several statistics
based on a confusion matrix. The package also includes several utilities to plot
and analyze the sensor data to aid in designing the neural network.
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
from collections import defaultdict
from collections import deque
import random
import itertools
import numpy as np
import inputparser
from neuralnetwork import NeuralNetwork

def extract_wanted_sensors(sample_set, wanted_sensors):
    '''
    Given a list of sensors to use, extract data from those sensors. The sensor
    data is passed as a mutable list to avoid copying it
    '''
    wanted_sensors.sort()
    sensor_count = sample_set[0][0].shape[1]
    if (wanted_sensors[0] < 0) or (wanted_sensors[-1] >= sensor_count):
        print ('Sensors to use for classification {} invalid. All must be in '
               'range 1 to {}'.format(sys.argv[3:], sensor_count))
        raise ValueError("Sensor must be in range 1 - {}".format(sensor_count))
    # This loop returns mutable lists, so this copy works
    for sample in sample_set:
        sample[0] = sample[0][:, wanted_sensors]

def generate_expected_results(node_to_category_mapping):
    '''
    Given the mapping of categories to nodes, return the expected node values
    for samples falling in each category
    '''
    expected_results = {}
    # The heuristic for classifiers is that the ideal result has the wanted
    # node at 0.8 and the remainder at 0.2. Using 1 and 0 makes training much
    # harder
    base_result_value = np.ones((len(node_to_category_mapping))) * 0.2
    # http://stackoverflow.com/questions/522563/accessing-the-index-in-python-for-loops
    for node, category in enumerate(node_to_category_mapping):
        expected_results[category] = base_result_value.copy()
        expected_results[category][node] = 0.8
    return expected_results

def generate_sample_sets(sample_set, training_samples, validation_data,
                         test_samples, node_to_category_mapping):
    '''
    Split samples into categories needed to train and test the neural network '''

    # Want as close to an even distribution of sample categories in each type
    # as possible so the network will not be biased toward particular
    # categories. Split the samples by category, split those by type, and then
    # recombine
    samples_by_category = defaultdict(list)
    for sample in sample_set:
        samples_by_category[sample[1]].append(sample)

    # The neural network works best on randomized sample sets. Shuffle the
    # sample order so its different each time the program is run
    for sample in samples_by_category.values():
        random.shuffle(sample)

    # Set the node that should have the highest proability for each category.
    # The mapping does not order as long as it is unique. This is efficiently
    # derived from the category keys, which are in a defintive (but arbitrary)
    # order
    # NOTE: Append to list so values are returned
    node_to_category_mapping.extend(samples_by_category.keys())

    expected_results = generate_expected_results(node_to_category_mapping)

    for category_list in samples_by_category.values():
        # Find the number of samples for each sample set
        total_samples = len(category_list)
        training_count = 0
        validation_count = 0

        # If only one sample, use it for training
        if total_samples >= 4:
            # Reseve half the samples for training and another quarter for
            # validation
            training_count = total_samples / 2
            validation_count = total_samples / 4
        else:
            training_count = 1
            if total_samples > 1:
                validation_count = 1

        # Select the samples in the training set and validation sets,
        # pairing each one with the expected results
        # http://stackoverflow.com/questions/18048698/efficient-iteration-over-slice-in-python
        for sample_set in category_list[:training_count]:
            # pair every sample with the expected results so they stay
            # connected when the list is randomized later
            for sample in sample_set[0]:
                # Note the double paranthesis to create a tuple
                training_samples.append((sample, expected_results[sample_set[1]]))

        # Validation samples and results can be two seprate lists because they
        # can be an arbitrary order
        for sample in category_list[training_count:training_count + validation_count]:
            validation_data['samples'].extend(sample[0])
            # http://stackoverflow.com/questions/2785954/creating-a-list-in-python-with-multiple-copies-of-a-given-object-in-a-single-lin
            validation_data['expected_results'].extend([expected_results[sample[1]]
                                                        for _ in xrange(len(sample[0]))])

        # The remainder become the test set
        for sample in category_list[training_count + validation_count:]:
            test_samples.append(sample)

    # The samples in the lists are currently grouped by category. To get
    # the best training, those samples should be random
    random.shuffle(training_samples)

def construct_network(training_samples):
    ''' Constructs a neural network given the training samples, and retuns it '''

    # First, need the number of input and output nodes. Since the sample list
    # maps inputs to expected outputs, this is easy from the shapes
    input_node_count = training_samples[0][0].shape[0]
    output_node_count = training_samples[0][1].shape[0]

    # To size the network, need the total number of training samples
    sample_count = len(training_samples)

    # No fixed formula exists for sizing a neural network. The design below is
    # based on http://stats.stackexchange.com/questions/181/how-to-choose-the-number-of-hidden-layers-and-nodes-in-a-feedforward-neural-netw
    # This assumes fairly noisy data, which is valid for sansor data
    hidden_node_count = sample_count // ((input_node_count + output_node_count) * 5)

    if hidden_node_count < output_node_count:
        hidden_node_count = output_node_count
    elif hidden_node_count > input_node_count:
        # Heuristic for networks with iterative training
        hidden_node_count = input_node_count

    node_counts = []
    node_counts.append(input_node_count)
    node_counts.append(hidden_node_count)
    node_counts.append(output_node_count)
    return NeuralNetwork(node_counts, 0.8)

def mean_squared_error(expected_results, actual_results):
    ''' Given calculated and expected results, return the mean squared error '''
    if expected_results.shape != actual_results.shape:
        print ('ERROR: Expected results and actual results counts differ. '
               'Expected: {} Actual: {}'.format(expected_results.shape,
                                                actual_results.shape))
        return 0.0
    else:
        errors = expected_results - actual_results
        errors = np.square(errors)
        return np.sum(errors, -1)

def is_increasing_values(value_queue):
    ''' Given a queue of values, return true if the values are non-decreasing '''

    # Queues require using the built in iterator for efficient access
    prev_value = float('-inf')
    for value in value_queue:
        if prev_value > value:
            return False
        prev_value = value
    return True

def train_network(node_network, training_samples, validation_data, patience_period):
    ''' Train the network given the training and validation data '''

    # Record the lowest validation error seen, and the weights at that point
    lowest_error = 0.0
    lowest_error_weights = []

    # Record the validations errors seen within the patience period. Need an
    # extra slot to accomodate the value immediately after the patience period
    # used in the test
    validation_error_queue = deque([], patience_period + 1)

    # Some networks never converge. Put a hard limit on the number of epochs to
    # prevent it from running forever. This value is somewhat arbitrary
    epoch_limit = 3000
    epoch_count = 0

    while True:
        # Each set of training sample processing is called an epoch. These are
        # grouped into chunks, and validated after each chunk. The use of
        # chunks trades off between training speed and minimizing error
        for _ in range(5):
            for sample in training_samples:
                node_network.process_sample_data(sample[0])
                node_network.backpropagate_errors(sample[1])
            epoch_count += 1

        # Calculate validation error
        validation_results = []
        for sample in validation_data['samples']:
            validation_results.append(node_network.process_sample_data(sample))
        validation_results = np.asarray(validation_results)
        validation_error = np.sum(mean_squared_error(validation_results,
                                                     validation_data['expected_results']))
        print ('Epoch: {} Validation error: '
               '{}'.format(epoch_count, validation_error))
        if not lowest_error_weights:
            # No weights implies first validation test
            lowest_error = validation_error
            lowest_error_weights = node_network.get_weights()
        elif validation_error < lowest_error:
            lowest_error = validation_error
            lowest_error_weights = node_network.get_weights()
        validation_error_queue.append(validation_error)

        # Stop on increasing error through the patience period or hitting the
        # training epoch limit
        if (((len(validation_error_queue) > patience_period) and
             is_increasing_values(validation_error_queue)) or
                (epoch_count > epoch_limit)):
            break

    # Reset network to weights with lowest error
    node_network.set_weights(lowest_error_weights)

def print_confusion_matrix(confusion_matrix, labels):
    ''' Pretty prints a confusion matrix '''


    # Want to print the matrix with columns and rows labeled. This requires
    # knowing the widths of each column. Assume there won't be many of them so
    # word wrap is not an issue. Also assume that the sample count is low enough
    # that the length of a printed value will not exceed the longest column
    # label
    max_word_length = 9 # "predicted"
    for label in labels:
        word_length = len(label)
        if word_length > max_word_length:
            max_word_length = word_length
    # Add two spaces between columns
    max_word_length += 2

    # Need the total number of columns. This is the width of the confusion
    # matrix plus 1
    output_columns = confusion_matrix.shape[1] + 1

    # Top label. The second label must line up in the center of the data
    # columns. Need to add two columns to the width so the label is offset by
    # one column to account for the label column on the left
    print "predicted".rjust(((output_columns + 2) * max_word_length) / 2)
    print "actual".rjust(max_word_length) \
          + "".join([label.rjust(max_word_length) for label in labels])

    # To print the matrix need to iteate rows and labels and join the column
    # contents. See http://stackoverflow.com/questions/17870612/printing-a-two-dimensional-array-in-python
    for label, values in itertools.izip(labels, confusion_matrix):
        print label.rjust(max_word_length) \
              + "".join([str(value).rjust(max_word_length) for value in values])

def report_classification_stats(confusion_matrix, categories):
    '''
    Calculate precision, recall, and balanced F statistic for each category
    given the classifier confusion matrix
    '''

    # The important statistics for classification are precision and recall.
    # The former is the percentage of a predicted category items that actually
    # fall in that category. The latter is the precentaage of items in a
    # category that were predicted to be that category. They are combined into
    # an onverall statistic called the balanced F statistic.
    #
    # This code calculates them category by category and then averages those
    # statistics to get overall values for the classifier. This gives equal
    # preference to the perfomance in each category

    # In a confusion matrix, the actual number in a category is the sum of each
    # row, the predicted number in a category is the sum of each column, and
    # the number correctly classified is the diagonal
    correctly_classified = np.diagonal(confusion_matrix).astype(np.float64)
    total_actual = np.sum(confusion_matrix, 1).astype(np.float64)
    total_predicted = np.sum(confusion_matrix, 0).astype(np.float64)

    # Handle the rare case of no test samples in a given category. This implies
    # the number correctly classified is zero, so the denonimator in the
    # calculations below does not matter. Note the less than one test to handle
    # float imprecision
    total_actual[total_actual < 1.0] = 1.0
    total_predicted[total_predicted < 1.0] = 1.0

    precision = correctly_classified / total_predicted
    recall = correctly_classified / total_actual

    # Avoid a divide by zero in the Fstatistic calculation by testing for it
    # up front. Since both precision and recall are non-negative, the only way
    # to get a zero value is for both to be zero, which implies the numerator
    # of the calculation will be zero. The denominator can be anything in that
    # case
    precision_recall_sum = precision + recall
    precision_recall_sum[precision_recall_sum < 1.0] = 1.0
    f_statistic = 2 * precision * recall / precision_recall_sum
    for cat, prec, rec, fstat in itertools.izip(categories, precision, recall, f_statistic):
        print ('Category: {} Precision: {} Recall: {} Balanced F Statistic:'
               ' {}'.format(cat, prec, rec, fstat))

    print ('Overall: Precision: {}  Recall: {} Balanced F Statistic:'
           ' {}'.format(precision.mean(), recall.mean(), f_statistic.mean()))

def main():
    '''
    Main Classification driver. Read in data files, classify the sensor data
    they contain, and evaluate the peformance of the classifier
    '''

    if len(sys.argv) < 3:
        print ('USAGE: SensorClassifier.py (Path to data file) (Path to '
               'results file) [sensors to use for classification by number]')
        sys.exit(1)
    sample_set = inputparser.load_sample_data(sys.argv[1], sys.argv[2])

    # If the data set is empty, likely have a file read error
    if len(sample_set) == 0:
        print "Input data corrupted, no data read"
        sys.exit(1)

    if len(sys.argv) > 3:
        # The first column in the data is a time stamp, so the sensor number is
        # one higher than the column index
        sensors_to_use = [int(value) - 1 for value in sys.argv[3:]]
        extract_wanted_sensors(sample_set, sensors_to_use)

    # Split the samples into groups for training, validation, and final test.
    # Also geneate expected results as needed. Note carefully how everything
    # is passed as a mutable list to avoid copies
    training_samples = []
    # Validation data is best processed in bulk. Keep the data and results
    # together for consistency
    validation_data = dict((('samples', []), ('expected_results', [])))
    test_samples = []
    node_to_category_mapping = []

    generate_sample_sets(sample_set, training_samples, validation_data,
                         test_samples, node_to_category_mapping)

    # The expected results are currently a list of ndarrays. Convert to a
    # single NDarray
    validation_data['expected_results'] = np.asarray(validation_data['expected_results'])

    # At this point, can build the network. The optimum layout depends on
    # the number of training samples available
    node_network = construct_network(training_samples)

    # Number of periods for which valiation error must increase to stop training
    patience_period = 3

    train_network(node_network, training_samples, validation_data, patience_period)

    # Test the network on each of the test samples and build the confusion
    # matrix Doing so requres a mapping between categories and the nodes
    # that signal them, which allows an ndarray to be used for the matrix
    category_to_node_mapping = {}
    for node_index, category in enumerate(node_to_category_mapping):
        category_to_node_mapping[category] = node_index
    confusion_matrix = np.zeros((len(node_to_category_mapping), len(node_to_category_mapping)))

    for sample in test_samples:
        predicted_result = node_network.find_category(sample[0])
        confusion_matrix[category_to_node_mapping[sample[1]]][predicted_result] += 1
    print_confusion_matrix(confusion_matrix, node_to_category_mapping)
    report_classification_stats(confusion_matrix, node_to_category_mapping)

if __name__ == '__main__':
    main()
