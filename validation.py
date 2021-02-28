
def validation(model, valid_dataset, criterion, args):
    valid_loss = 0

    # Run a validation loop at the end of each epoch.
    for x_batch_val, y_batch_val in valid_dataset:
        val_logits = model(x_batch_val, training=False)
        loss = criterion(y_batch_val, val_logits)
        valid_loss += loss

    return valid_loss


