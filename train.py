import tensorflow as tf
from model import *
from preprocessing import *

train_images, train_labels = get_data(training=True)
test_images, test_labels = get_data(training=False)

with tf.Session() as sess:
    sess.run(init)
    train_loss = []
    test_loss = []
    train_accuracy = []
    test_accuracy = []

    summary_writer = tf.summary.FileWriter('./Output', sess.graph)
    for i in range(training_iters):
        for batch in range(len(train_images)//batch_size):
            batch_x = train_images[batch*batch_size:min((batch+1)*batch_size, len(train_images))]
            batch_y = train_labels[batch*batch_size:min((batch+1)*batch_size, len(train_labels))]
            # Run optimization op (backprop).
                # Calculate batch loss and accuracy
            opt = sess.run(optimizer, feed_dict={X: batch_x,
                                                              Y: batch_y})
            loss, acc = sess.run([cost, accuracy], feed_dict={X: batch_x,
                                                              Y: batch_y})
        print("Iter " + str(i) + ", Loss= " + \
                      "{:.6f}".format(loss) + ", Training Accuracy= " + \
                      "{:.5f}".format(acc))

        # Calculate accuracy for all 10000 mnist test images
        test_acc,valid_loss = sess.run([accuracy, cost], feed_dict={X: test_images,Y : test_labels})
        train_loss.append(loss)
        test_loss.append(valid_loss)
        train_accuracy.append(acc)
        test_accuracy.append(test_acc)
        print("Testing Accuracy:","{:.5f}".format(test_acc))
    summary_writer.close()