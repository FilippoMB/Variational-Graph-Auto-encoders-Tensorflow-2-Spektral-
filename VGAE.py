import tensorflow as tf
import numpy as np
import scipy.sparse as sp

from tensorflow.keras.layers import Dropout, Input
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.regularizers import l2

from spektral.datasets import citation
from spektral.layers import GCNConv
from spektral.utils.sparse import sp_matrix_to_sp_tensor

from utils import mask_test_edges, get_roc_score

#################################
########### Load data ###########
#################################

dataset = citation.Cora()
graph = dataset[0]

# Node features
X = graph.x 

# Target graph to reconstruct
A_label = graph.a + sp.eye(graph.a.shape[0], dtype=np.float32)
A_label = A_label.toarray().reshape([-1])

# Remove edges randomly from training set and put them in the validation/test sets
adj_train, _, val_edges, val_edges_false, test_edges, test_edges_false = mask_test_edges(graph.a)

# Normalize the adj matrix and convert it to sparse tensor
A = GCNConv.preprocess(adj_train)
A = sp_matrix_to_sp_tensor(A)

# Compute the class weights (necessary due to imbalanceness in the number of non-zero edges)
pos_weight = float(adj_train.shape[0] * adj_train.shape[0] - adj_train.sum()) / adj_train.sum()
norm = adj_train.shape[0] * adj_train.shape[0] / float((adj_train.shape[0] * adj_train.shape[0] - adj_train.sum()) * 2)


#################################
####### Model definition ########
#################################

# Parameters
hidden_dim1, hidden_dim2 = 32, 16 # Units in the GCN layers
dropout = 0.0                     # Dropout rate 
l2_reg = 0e-5                     # L2 regularization rate
learning_rate = 1e-2              # Learning rate
epochs = 20000                    # Max number of training epochs
val_epochs = 20                   # After how many epochs should check the validation set

N = dataset.n_nodes               # Number of nodes in the graph
F = dataset.n_node_features       # Original size of node features

# GNN architecture
x_in = Input(shape=(F,))
a_in = Input((N,), sparse=True)

gc_1 = GCNConv(
    hidden_dim1,
    activation="relu",
    kernel_regularizer=l2(l2_reg),
)([x_in, a_in])
gc_1 = Dropout(dropout)(gc_1)

z_mean = GCNConv(
    hidden_dim2,
    activation=None,
    kernel_regularizer=l2(l2_reg),
)([gc_1, a_in])
z_log_std = GCNConv(
    hidden_dim2,
    activation=None,
    kernel_regularizer=l2(l2_reg),
)([gc_1, a_in])
z = z_mean + tf.random.normal([N, hidden_dim2]) * tf.exp(z_log_std)

out = tf.matmul(z, tf.transpose(z))
out = tf.reshape(out, [-1])

# This is not used for training and we make it deterministic
out_det = tf.matmul(z_mean, tf.transpose(z_mean))
A_rec = tf.keras.layers.Activation('sigmoid')(out_det)

# Build model
model = Model(inputs=[x_in, a_in], outputs=[out, A_rec, z_mean, z_log_std])
optimizer = Adam(learning_rate=learning_rate)


#################################
######### Train & Test ##########
#################################

# Define training step
@tf.function
def train():
    with tf.GradientTape() as tape:
        
        predictions, _, model_z_mean, model_z_log_std = model([X, A], training=True)
        
        # reconstruction loss
        rec_loss = norm*tf.reduce_mean(tf.nn.weighted_cross_entropy_with_logits(logits=predictions, labels=A_label, pos_weight=pos_weight))
        
        # latent loss
        kl_loss = (0.5 / N) * tf.reduce_mean(tf.reduce_sum(
            1 + 2 * model_z_log_std - tf.square(model_z_mean) - tf.square(tf.exp(model_z_log_std)), 1
            ))
        
        # total loss
        loss = rec_loss - kl_loss + sum(model.losses)
    gradients = tape.gradient(loss, model.trainable_variables)
    optimizer.apply_gradients(zip(gradients, model.trainable_variables))
    return loss

# Training
best_val_roc = 0
for epoch in range(1, epochs):
    loss = train()
    print(f"epoch: {epoch:d} -- loss: {loss:.3f}")
    
    # Check performance on Validation
    if epoch%val_epochs==0:
        _, adj_rec, _, _ = model([X,A])
        adj_rec = adj_rec.numpy()
        val_roc, _ = get_roc_score(val_edges, val_edges_false, adj_rec)
        
        if val_roc <= best_val_roc:
            break
        else:
            best_val_roc = val_roc
            acc = np.mean(np.round(adj_rec) == graph.a.toarray())
            print(f"Val AUC: {best_val_roc*100:.1f}, Accuracy: {acc*100:.1f}")
        
# Testing
_, adj_rec, node_emb, _ = model([X,A])
adj_rec = adj_rec.numpy()
roc_score, ap_score = get_roc_score(test_edges, test_edges_false, adj_rec)
print(f"AUC: {roc_score*100:.1f}, AP: {ap_score*100:.1f}")
test_acc = np.mean(np.round(adj_rec.ravel()) == A_label)
print(f"Test accuracy: {test_acc*100:.1f}")