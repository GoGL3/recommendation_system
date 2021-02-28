import tensorflow as tf
from time import time

from get_log import *
from validation import *


def train_model(model, train_dataset, valid_dataset, criterion, optimizer, args, logger):

    total_step = len(train_dataset)

    for epoch in range(args['EPOCHS']):

        print("\nStart of epoch %d" % (epoch,))
        running_loss = 0

        # Iterate over the batches of the dataset.
        for step, (x_batch_train, y_batch_train) in enumerate(train_dataset):

            # Open a GradientTape to record the operations run
            # during the forward pass, which enables auto-differentiation.
            with tf.GradientTape() as tape:

                # Run the forward pass of the layer.
                # The operations that the layer applies
                # to its inputs are going to be recorded
                # on the GradientTape.
                logits = model(x_batch_train, training=True)  # Logits for this minibatch

                # Compute the loss value for this minibatch.
                loss_value = criterion(y_batch_train, logits)

            # Use the gradient tape to automatically retrieve
            # the gradients of the trainable variables with respect to the loss.
            grads = tape.gradient(loss_value, model.trainable_weights)

            # Run one step of gradient descent by updating
            # the value of the variables to minimize the loss.
            optimizer.apply_gradients(zip(grads, model.trainable_weights))
            running_loss += loss_value

            if step % total_step == (total_step - 1):

                # Run a validation loop at the end of each epoch.
                valid_loss = validation(model, valid_dataset, criterion, args)
                valid_loss_average = valid_loss / len(valid_dataset)

                logger.info("Epoch: {}/{}.. ".format(epoch + 1, args['EPOCHS']) +
                            "Training Loss: {:.6f}.. ".format(running_loss / total_step) +
                            "Validation Loss: {:.6f}.. ".format(valid_loss_average))
                model.save(args['SAVE_PATH']+'checkpoint_%s_%d.h5' % (args['LAYERS'], time()))

    return

