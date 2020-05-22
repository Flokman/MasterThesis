import tensorflow as tf
from tensorflow.python.ops import confusion_matrix


class CohenKappa:
    def __init__(self, num_classes):
        self.num_classes = num_classes

    def cohen_kappa(self,
                    labels,
                    predictions_idx):
        labels = tf.argmax(labels, axis=-1)
        predictions_idx = tf.argmax(predictions_idx, axis=-1)
        conf_matrix = confusion_matrix.confusion_matrix(labels, predictions_idx, num_classes=self.num_classes, dtype=tf.float64)
        total_correct = tf.reduce_sum(tf.linalg.tensor_diag_part(conf_matrix))
        total = tf.reduce_sum(tf.reduce_sum(conf_matrix))
        accuracy = total_correct / total
        expected = 1 / (total * total) * tf.reduce_sum(
            [tf.reduce_sum(conf_matrix[:, c]) * tf.reduce_sum(conf_matrix[c, :]) for c in range(self.num_classes)])
        # if expected is 1, return 0 (otherwise would result in nan)
        kappa = tf.cond(expected == 1, lambda: tf.cast(0.0, dtype=tf.float64), lambda: (accuracy - expected) / (1 - expected))
        return kappa