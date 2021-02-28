
def validation(model, valid_dataset, criterion):
    valid_loss = 0

    # Run a validation loop at the end of each epoch.
    for user_batch_val, item_batch_val, y_batch_val in valid_dataset:

        val_logits = model([user_batch_val, item_batch_val], training=False)
        loss = criterion(y_batch_val, val_logits)
        valid_loss += loss

    return valid_loss


