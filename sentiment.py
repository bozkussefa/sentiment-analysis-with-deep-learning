import numpy as np
import random
import string
import tensorflow as tf
import sys

def create_vectors(vector):
    vec = {}
    with open(vector, 'r', encoding="utf8") as f:
        contents = f.readlines()
        for l in contents[:len(contents)]:
            (key, val) = l.split(":")
            vec[key] = [val]
    return vec

def create_lexicon(vec):
    lexicon = []
    with open(vec, 'r', encoding="utf8") as f:
        contents = f.readlines()
        for l in contents[:len(contents)]:
            all_words = l.split(":")
            all_words = all_words[0]
            lexicon.append(all_words)

    return lexicon

def sample_handling(sample, lexicon, classification, vectors):
    featureset = []
    with open(sample, 'r', encoding="utf8") as f:
        contents = f.readlines()
        for l in contents[:len(contents)]:
            table = str.maketrans({key: None for key in string.punctuation})
            new_s = l.translate(table)
            current_words = new_s.split(" ")
            features = np.zeros(200, dtype=float)
            for word in current_words:
                if word in lexicon:
                    vector_values = vectors.get(word)[0].split(" ")
                    for c in range(0, 200):
                        a = np.array(vector_values, dtype=float)
                        features[c] += a[c]

            features = list(features)
            featureset.append([features, classification])
    return featureset

def create_feature_sets_and_labels(pos, neg, vec, test_size):
    vectors = create_vectors(vec)
    lexicon = create_lexicon(vec)
    features = []
    features += sample_handling(pos, lexicon, [1, 0], vectors)
    features += sample_handling(neg, lexicon, [0, 1], vectors)
    random.shuffle(features)
    features = np.array(features)
    testing_size = int(test_size * len(features))
    train_x = list(features[:, 0][:-testing_size])
    train_y = list(features[:, 1][:-testing_size])
    test_x = list(features[:, 0][-testing_size:])
    test_y = list(features[:, 1][-testing_size:])

    return train_x, train_y, test_x, test_y

def neural_network_model(data):
    l1 = tf.add(tf.matmul(data, hidden_1_layer['weight']), hidden_1_layer['bias'])
    l1 = tf.nn.relu(l1)

    l2 = tf.add(tf.matmul(l1, hidden_2_layer['weight']), hidden_2_layer['bias'])
    l2 = tf.nn.relu(l2)

    output = tf.matmul(l2, output_layer['weight']) + output_layer['bias']
    return output

def train_neural_network(x):
    prediction = neural_network_model(x)

    cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=prediction, labels=y))
    optimizer = tf.train.GradientDescentOptimizer(learning_rate=0.001).minimize(cost)

    with tf.Session() as sess:
        sess.run(tf.initialize_all_variables())

        for epoch in range(hm_epochs):
            epoch_loss = 0
            i = 0
            while i < len(train_x):
                start = i
                end = i + batch_size
                batch_x = np.array(train_x[start:end])
                batch_y = np.array(train_y[start:end])

                _, c = sess.run([optimizer, cost], feed_dict={x: batch_x,
                                                              y: batch_y})

                epoch_loss += c
                i += batch_size

            print('Epoch', epoch + 1, 'completed out of', hm_epochs, 'loss:', epoch_loss)
        correct = tf.equal(tf.argmax(prediction, 1), tf.argmax(y, 1))
        accuracy = tf.reduce_mean(tf.cast(correct, 'float'))

        print('Accuracy:', accuracy.eval({x: test_x, y: test_y}))

if __name__ == '__main__':
    args = sys.argv

    train_x, train_y, test_x, test_y = create_feature_sets_and_labels(args[1], args[2], args[3],1-int(args[4])/100)

    n_nodes_hl1 = 100
    n_nodes_hl2 = 100

    n_classes = 2
    batch_size = 10
    hm_epochs = 10

    x = tf.placeholder('float')
    y = tf.placeholder('float')

    hidden_1_layer = {'f_fum': n_nodes_hl1,
                      'weight': tf.Variable(tf.random_normal([len(train_x[0]), n_nodes_hl1])),
                      'bias': tf.Variable(tf.random_normal([n_nodes_hl1]))}

    hidden_2_layer = {'f_fum': n_nodes_hl2,
                      'weight': tf.Variable(tf.random_normal([n_nodes_hl1, n_nodes_hl2])),
                      'bias': tf.Variable(tf.random_normal([n_nodes_hl2]))}

    output_layer = {'f_fum': None,
                    'weight': tf.Variable(tf.random_normal([n_nodes_hl2, n_classes])),
                    'bias': tf.Variable(tf.random_normal([n_classes])), }

    train_neural_network(x)