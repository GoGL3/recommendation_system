# def train_model(model, gmf_model, mlp_model, num_layers):
#
#     # GMF
#     gmf_user_embeddings = gmf_model.get_layer('user_embedding').get_weights()
#     gmf_item_embeddings = gmf_model.get_layer('item_embedding').get_weights()
#     model.get_layer('mf_embedding_user').set_weights(gmf_user_embeddings)
#     model.get_layer('mf_embedding_item').set_weights(gmf_item_embeddings)
#
#     # MLP
#     mlp_user_embeddings = mlp_model.get_layer('user_embedding').get_weights()
#     mlp_item_embeddings = mlp_model.get_layer('item_embedding').get_weights()
#     mlp_context_embeddings = mlp_model.get_layer('context_embedding').get_weights()
#     model.get_layer('mlp_embedding_user').set_weights(mlp_user_embeddings)
#     model.get_layer('mlp_embedding_item').set_weights(mlp_item_embeddings)
#     model.get_layer('mlp_embedding_context').set_weights(mlp_context_embeddings)
#
#     # MLP layers
#     for i in range(1, num_layers):
#         mlp_layer_weights = mlp_model.get_layer('layer%d' %i).get_weights()
#         model.get_layer('layer%d' %i).set_weights(mlp_layer_weights)
#
#     # Prediction weights
#     gmf_prediction = gmf_model.get_layer('output_layer').get_weights()
#     mlp_prediction = mlp_model.get_layer('output_layer').get_weights()
#     new_weights = np.concatenate((gmf_prediction[0], mlp_prediction[0]), axis=0)
#     new_b = gmf_prediction[1] + mlp_prediction[1]
#     model.get_layer('prediction').set_weights([0.5*new_weights, 0.5*new_b])
#     return model
