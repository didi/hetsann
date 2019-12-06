import numpy as np
import tensorflow as tf

conv1d = tf.layers.conv1d

def attn_head(seq, out_sz, bias_mat, activation, in_drop=0.0, coef_drop=0.0, residual=False):
    with tf.name_scope('my_attn'):
        if in_drop != 0.0:
            seq = tf.nn.dropout(seq, 1.0 - in_drop)

        seq_fts = tf.layers.conv1d(seq, out_sz, 1, use_bias=False)

        # simplest self-attention possible
        f_1 = tf.layers.conv1d(seq_fts, 1, 1)
        f_2 = tf.layers.conv1d(seq_fts, 1, 1)
        logits = f_1 + tf.transpose(f_2, [0, 2, 1])
        coefs = tf.nn.softmax(tf.nn.leaky_relu(logits) + bias_mat)

        if coef_drop != 0.0:
            coefs = tf.nn.dropout(coefs, 1.0 - coef_drop)
        if in_drop != 0.0:
            seq_fts = tf.nn.dropout(seq_fts, 1.0 - in_drop)

        vals = tf.matmul(coefs, seq_fts)
        ret = tf.contrib.layers.bias_add(vals)

        # residual connection
        if residual:
            if seq.shape[-1] != ret.shape[-1]:
                ret = ret + conv1d(seq, ret.shape[-1], 1) # activation
            else:
                ret = ret + seq

        return activation(ret)  # activation

# Experimental sparse attention head (for running on datasets such as Pubmed)
# N.B. Because of limitations of current TF implementation, will work _only_ if batch_size = 1!
def sp_attn_head(seq, out_sz, adj_mat, activation, nb_nodes, in_drop=0.0, coef_drop=0.0, residual=False):
    with tf.name_scope('sp_attn'):
        if in_drop != 0.0:
            seq = tf.nn.dropout(seq, 1.0 - in_drop)

        seq_fts = tf.layers.conv1d(seq, out_sz, 1, use_bias=False)

        # simplest self-attention possible
        f_1 = tf.layers.conv1d(seq_fts, 1, 1)
        f_2 = tf.layers.conv1d(seq_fts, 1, 1)
        
        f_1 = tf.reshape(f_1, (nb_nodes, 1))
        f_2 = tf.reshape(f_2, (nb_nodes, 1))

        f_1 = adj_mat*f_1
        f_2 = adj_mat * tf.transpose(f_2, [1,0])

        logits = tf.sparse_add(f_1, f_2)
        lrelu = tf.SparseTensor(indices=logits.indices, 
                values=tf.nn.leaky_relu(logits.values), 
                dense_shape=logits.dense_shape)
        coefs = tf.sparse_softmax(lrelu)

        if coef_drop != 0.0:
            coefs = tf.SparseTensor(indices=coefs.indices,
                    values=tf.nn.dropout(coefs.values, 1.0 - coef_drop),
                    dense_shape=coefs.dense_shape)
        if in_drop != 0.0:
            seq_fts = tf.nn.dropout(seq_fts, 1.0 - in_drop)

        # As tf.sparse_tensor_dense_matmul expects its arguments to have rank-2,
        # here we make an assumption that our input is of batch size 1, and reshape appropriately.
        # The method will fail in all other cases!
        coefs = tf.sparse_reshape(coefs, [nb_nodes, nb_nodes])
        seq_fts = tf.squeeze(seq_fts)
        vals = tf.sparse_tensor_dense_matmul(coefs, seq_fts)
        vals = tf.expand_dims(vals, axis=0)
        vals.set_shape([1, nb_nodes, out_sz])
        ret = tf.contrib.layers.bias_add(vals)

        # residual connection
        if residual:
            if seq.shape[-1] != ret.shape[-1]:
                ret = ret + conv1d(seq, ret.shape[-1], 1) # activation
            else:
                ret = ret + seq

        return activation(ret)  # activation


def sp_hete_attn_head(seq, out_sz, adj_mat, adj_type, activation, nb_nodes, in_drop=0.0, coef_drop=0.0, residual=False):
    # input adjacency matrices are TRANSPOSED before feeding!
    with tf.name_scope('sp_hete_attn'):
        if in_drop != 0.0:
            seq = [tf.nn.dropout(seq_i, 1.0 - in_drop) for seq_i in seq]
        
        # seq_fts[j][i]: hidden features from group i to group j, center node is j
        # 1 * nb_nodes_i * out_sz_j
        seq_fts = [[tf.layers.conv1d(seq_i,
                                     out_sz, # out_sz_j
                                     1,
                                     use_bias=False) for seq_i in seq] for _ in seq]
        # for out_sz_j in out_sz
        coefs_lists = [[] for _ in range(len(seq))]
        seq_fts_lists = [[] for _ in range(len(seq))]
        
        # simplest self-attention possible
        for adj_ij, type_ij in zip(adj_mat, adj_type):
            # transposed, # nb_nodes_j * nb_nodes_i
            i, j = type_ij
            
            f_1 = tf.layers.conv1d(seq_fts[j][j], 1, 1)
            f_2 = tf.layers.conv1d(seq_fts[j][i], 1, 1)
            
            f_1 = tf.reshape(f_1, (nb_nodes[j], 1))
            f_2 = tf.reshape(f_2, (nb_nodes[i], 1))
    
            f_1 = adj_ij*f_1
            f_2 = adj_ij * tf.transpose(f_2, [1,0])
    
            logits = tf.sparse_add(f_1, f_2) # nb_nodes_j * nb_nodes_i
            coefs = tf.SparseTensor(indices=logits.indices, 
                    values=tf.nn.leaky_relu(logits.values), 
                    dense_shape=logits.dense_shape)
            # coefs = tf.sparse_softmax(lrelu)
    
            if coef_drop != 0.0:
                coefs = tf.SparseTensor(indices=coefs.indices,
                        values=tf.nn.dropout(coefs.values, 1.0 - coef_drop),
                        dense_shape=coefs.dense_shape)
            coefs_lists[j].append(coefs) # transposed, nb_nodes_j * nb_nodes_i
            if in_drop != 0.0:
                seq_fts_ij = tf.nn.dropout(seq_fts[j][i], 1.0 - in_drop)
            seq_fts_lists[j].append(tf.squeeze(seq_fts_ij)) # nb_nodes_i * out_sz_j
            
    
        # As tf.sparse_tensor_dense_matmul expects its arguments to have rank-2,
        # here we make an assumption that our input is of batch size 1, and reshape appropriately.
        # The method will fail in all other cases!
        
        # coefs = tf.sparse_reshape(coefs, [nb_nodes, nb_nodes])
        coefs = [tf.sparse_concat(1, coefs_list) for coefs_list in coefs_lists]
        coefs = [tf.sparse_softmax(coef) for coef in coefs]
        # seq_fts = tf.squeeze(seq_fts)
        seq_fts = [tf.concat(seq_fts_list, 0) for seq_fts_list in seq_fts_lists]
        vals = [tf.sparse_tensor_dense_matmul(coef, seq_ft) for coef, seq_ft in zip(coefs, seq_fts)]
        # nb_nodes_j * out_sz_j
        vals = [tf.expand_dims(val, axis=0) for val in vals]
        for i, val in enumerate(vals):
            val.set_shape([1, nb_nodes[i], out_sz])
        ret = [tf.contrib.layers.bias_add(val) for val in vals]
    
        # residual connection
        if residual:
            ret2 = []
            for r, s in zip(ret, seq):
                if s.shape[-1] != r.shape[-1]:
                    ret2.append(r + tf.layers.conv1d(s, r.shape[-1], 1)) 
                else:
                    ret2.append(r + s)
            ret = ret2
        ret = [activation(r) for r in ret]
        return ret  # activation

def full_connection(seq, out_sz, target_node, activation, in_drop=0.0, use_bias=True):
    with tf.name_scope('full_connection_layer'):
        if in_drop != 0.0:
            seq = [tf.nn.dropout(seq_i, 1.0 - in_drop) for seq_i in seq]
        
        seq_fc = [tf.layers.conv1d(seq[target_node[i]], out_sz[i], 1, use_bias=use_bias) for i in range(len(target_node))]
        seq_fc = [tf.squeeze(seq_i) for seq_i in seq_fc] # remove the bach_size which is set as 1
        ret = [activation(s) for s in seq_fc]
        return ret