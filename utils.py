
# Utility function to create a tensorflow dataset (iterator on batches, with sequences of features and labels).
# Labels are in the future of features.
# Length of feature sequences is input_sequence_length
# Length of labels  sequences is output_sequence_length

def timeseries_dataset_multistep(features, labels, input_sequence_length, output_sequence_length, batch_size):
    def extract_output(l):
        return l[:output_sequence_length]
    
        feature_ds = tf.keras.preprocessing.timeseries_dataset_from_array(features, None, input_sequence_length, batch_size=1).unbatch()
        label_ds = tf.keras.preprocessing.timeseries_dataset_from_array(labels, None, input_sequence_length, batch_size=1) \
            .skip(input_sequence_length) \
            .unbatch() \
            .map(extract_output)
    
    return tf.data.Dataset.zip((feature_ds, label_ds)).batch(batch_size)

