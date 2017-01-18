'''
Neural network implementation. One class represents a layer, which can be
processed very efficiently using ndarrays. The second class represents a group
of layers to form the entire network. The network was implemented by hand 
instead of using a package in order to learn the algogithm. It uses the classic
gradient-descent method for training
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
import itertools
import numpy as np

class NeuralLayer(object):
    '''
    One layer of a neural network. Initialize with number of nodes in the
    previous layer, the number in this layer, and velocity damper value
    '''
    def __init__(self, input_nodes, nodes, velocity_damper):
        # Weight matrix converts data from input nodes into that for the
        # layer's nodes. Initialize it with random data. If it used consistent
        # data, the learning algorithm would not work properly
        self.weights = np.random.rand(input_nodes, nodes)

        # The last input and output values need to be cached for error
        # correction
        self.last_output = np.zeros(nodes)
        self.last_input = np.zeros(input_nodes)

        self.velocity_damper = velocity_damper

        # Finally, need the initial velocities. This is a zeroed version of
        # the weights. Note the extra set of paraentheses:
        # http://stackoverflow.com/questions/5446522/data-type-not-understood
        self.velocities = np.zeros((input_nodes, nodes))

    def process_layer(self, input_values):
        '''
        Feed forward through this layer of the neural network. Makes a cached
        copy of the output values for backpropagation
        '''
        # If the number of inputs does not match the number of nodes, have
        # a configuration error. Return all zeros and don't update the
        # cached values
        if (input_values.ndim != 1) or (input_values.shape[0] != self.weights.shape[0]):
            print ('Node input invalid. Expected {} values, '
                   'have {}'.format(self.weights.shape[0], input_values.shape[0]))
            return np.zeros(self.weights.shape[1])

        # To find the output values, multiply the inputs by the weight matrix,
        # in effect combining the values differently for every mode in the
        # layer. The sums get processed through the sigmoid function to get the
        # final results. The sigmoid function is often used in classification
        # networks because the output can be interpretated as probabilities,
        # and it has a well-behaved easy to calculate derivative
        self.last_input = input_values
        total_net_input = input_values.dot(self.weights)
        # The sigmoid is 1/(1 + e^-x)
        self.last_output = 1 / (1 + np.exp(-total_net_input))
        return self.last_output

    def adjust_weights(self, errors):
        '''
        Given the difference between the expected and actual outputs for this
        layer, adjust the weight matrix and find the errors to pass to the next
        layer. Note that this method is sensitive to sign; errors must be
        expected - actual, not the other way around!
        '''

        # If the number of errors does not match the number of nodes in the
        # next layer, have a configuration error. Return all zeros and don't
        # update the weights
        if (errors.ndim != 1) or (errors.shape[0] != self.weights.shape[1]):
            print ('Error values for weight adjustment invalid. Expected {} '
                   'values, have {}'.format(self.weights.shape[1], errors.shape[0]))
            return np.zeros(self.weights.shape[0])

        # The wieght adjustment gets done via the classic back propagation
        # gradient descent algorithm. It requires knowing the derivative of the
        # input calculation at the point values were calculated. For the
        # sigmoid function this is incredibly easy. Define Y = sigmoid(X).
        # df(X)/dX = Y(1-Y) ! The last values calculated were cached, making
        # this easy to determine. That gets multiplied by the errors to get
        # the adjustment inputs
        # See https://mattmazur.com/2015/03/17/a-step-by-step-backpropagation-example/
        # Also, note the sign flip on the error, an effect of the derivative
        # of the mean squared error function
        delta = self.last_output * (1 - self.last_output) * (-errors)

        # Calculate the error to pass on to the next layer, which is the
        # negative delta times the transpose of the CURRENT weights
        next_errors = -delta.dot(self.weights.T)

        # Calculate the new velocity, which is the damped current velocity plus
        # the delta values times the last input.
        self.velocities = ((self.velocities * self.velocity_damper)
                           + np.outer(self.last_input, delta.T))

        self.weights -= self.velocities
        return next_errors

    def __str__(self):
        # Use join to get spacing right. Note double parenthses to create a tupple
        return ' '.join(("Weights:", str(self.weights), "Last inputs:",
                         str(self.last_input), "Last output:",
                         str(self.last_output)))

    def __repr__(self):
        return self.__str__()

class NeuralNetwork(object):
    '''
    An entire neural network
    '''

    def __init__(self, node_counts, velocity_damper):
        '''
        Constuct the network given a list of the number of nodes per layer. If
        the list has less than two layers in it, or any value is zero, the
        network does nothing (this is partly for the prev_node test, but also
        because a network with no nodes in a layer really is meaningless)
        '''
        self.nodes = []
        if (len(node_counts) > 1) and (0 not in node_counts):
            prev_node = None
            for node in node_counts:
                if prev_node:
                    self.nodes.append(NeuralLayer(prev_node, node, velocity_damper))
                prev_node = node

        # The last result returned. Used for error propagation
        self.last_result = None

    def process_sample_data(self, sample_data):
        '''
        Feed forward a set of inputs through the network
        '''

        if not self.nodes:
            # If network does nothing, return samples unchanged
            return sample_data
        else:
            current_result = sample_data
            for node in self.nodes:
                current_result = node.process_layer(current_result)
            self.last_result = current_result
            return self.last_result

    def find_category(self, sample_set):
        '''
        Given a time series of inputs, predict the category of item that
        generated those inputs in the sensor network
        '''

        # Predicting a category has to deal with two types of issues:
        # uncertainty in the network and outliers in the data. The former is
        # mostly delt with by the classic technique of assigning the category
        # to whichever output node has the highest value. The latter is delt
        # with by classifying every input in the series individually and taking
        # whichever category a plurality fall into. Note that this process can
        # be misled by very noisy data
        results = np.zeros(self.nodes[-1].weights.shape[1])
        for sample in sample_set:
            category_probability = self.process_sample_data(sample)
            results[np.argmax(category_probability)] += 1
        return np.argmax(results)

    def backpropagate_errors(self, expected_results):
        '''
        Update the network given the expected outputs for the last set of
        inputs. The update is done by the classic back-propagation algorithm
        '''

        # Errors only have context in terms of the last processed inputs. If
        # these are not defined, have nothing to do. Also ignore if the network
        # is malformed
        if (self.nodes) and (self.last_result.size):
            if expected_results.shape != self.last_result.shape:
                print ('ERROR: Wrong number of expected results given retuned '
                       'values (expected {} '
                       'got {})'.format(self.last_result.shape, expected_results.shape))
            else:
                # Sign is very important, this order is specific
                last_errors = expected_results - self.last_result
                for node in reversed(self.nodes):
                    last_errors = node.adjust_weights(last_errors)

    def get_weights(self):
        '''
        Return a vector of the current weights for this network. Used to cache
        weights at a paticular moment in network evolution
        '''

        weights = []
        for node in self.nodes:
            # Need an explicit copy so future weight updates won't overwrite
            # the returned weight data
            weights.append(node.weights.copy())
        return weights

    def set_weights(self, weights):
        '''
        Set the network to a given set of weights, likely ones previously
        cached. The wieghts must be consistent with the network topolology
        '''

        # Validate the weights. They must have the same shape as those in the
        # nodes, in the same order
        valid = 1
        if len(weights) != len(self.nodes):
            print ('ERROR on weight update, supplied {} but network has {} '
                   'sets of weights'.format(len(weights), len(self.nodes)))
            valid = 0
        else:
            for new, node in itertools.izip(weights, self.nodes):
                if new.shape != node.weights.shape:
                    print ('ERROR on weight update. node expects {} '
                           'supplied {}'.format(node.weights.shape, new.shape))
                    valid = 0
        # Update after all validation to preserve integrity
        if valid:
            for new, node in itertools.izip(weights, self.nodes):
                node.weights = new

    def __str__(self):
        return str(self.nodes)

    def __repr__(self):
        return self.__str__()
