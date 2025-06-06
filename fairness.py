import pandas as pd
import tensorflow as tf
from tensorflow_model_remediation import min_diff

LABEL_KEY = 'PINCP'

# Helper function to convert a pandas DataFrame into a tf.Data.dataset object
# necessary for the purposes of this exercise.
def dataframe_to_dataset(dataframe):
  dataframe = dataframe.copy()
  labels = dataframe.pop(LABEL_KEY)
  dataset = tf.data.Dataset.from_tensor_slices(
      ((dict(dataframe), labels)))
  return dataset

# Stack the inputs as a dictionary and preprocess them.
def stack_dict(inputs, fun=tf.stack):
  values = []
  for key in sorted(inputs.keys()):
    values.append(tf.cast(inputs[key], tf.float64))

  return fun(values, axis=-1)

def main():
    """Main is entry point for the script."""

    # Import the dataset
    acs_df = pd.read_csv(filepath_or_buffer="https://download.mlcc.google.com/mledu-datasets/acsincome_raw_2018.csv")

    # Print five random rows of the pandas DataFrame.
    print(acs_df.sample(5))

    LABEL_THRESHOLD = 50000.0

    acs_df[LABEL_KEY] = acs_df[LABEL_KEY].apply(
        lambda income: 1 if income > LABEL_THRESHOLD else 0)

    print(acs_df.sample(10))

    inputs = {}
    features = acs_df.copy()
    features.pop(LABEL_KEY)

    # Instantiate a Keras input node for each column in the dataset.
    for name, column in features.items():
        if name != LABEL_KEY:
            inputs[name] = tf.keras.Input(
                shape=(1,), name=name, dtype=tf.float64)

    x = stack_dict(inputs, fun=tf.concat)

    # Collect the features from the DataFrame, stack them together and normalize
    # their values by passing them to the normalization layer.
    normalizer = tf.keras.layers.Normalization(axis=-1)
    normalizer.adapt(stack_dict(dict(features)))

    # Build the main body of the model using a normalization layer, two dense
    # rectified-linear layers, and a single output node for classification.
    x = normalizer(x)
    x = tf.keras.layers.Dense(64, activation='relu')(x)
    x = tf.keras.layers.Dense(32, activation='relu')(x)
    outputs = tf.keras.layers.Dense(1, activation='sigmoid')(x)

    # Put it all together using the Keras Functional API
    base_model = tf.keras.Model(inputs, outputs)

    # Define the metrics used to monitor model performance while training.
    METRICS = [
    tf.keras.metrics.BinaryAccuracy(name='accuracy'),
    tf.keras.metrics.AUC(name='auc'),
    ]

    # Configure the model for training using a stochastic gradient descent
    # optimizer, cross-entropy loss between true labels and predicted labels, and
    # the metrics defined above to evaluate the base model during training.
    base_model.compile(
        optimizer='adam',
        loss=tf.keras.losses.BinaryCrossentropy(),
        metrics=METRICS)

    RANDOM_STATE = 200
    BATCH_SIZE = 100
    EPOCHS = 10

    # Use the sample() method in pandas to split the dataset into a training set
    # that represents 80% of the original dataset, then convert it to a
    # tf.data.Dataset object, and finally train the model using the
    # converted training set.
    acs_train_df = acs_df.sample(frac=0.8, random_state=RANDOM_STATE)
    acs_train_ds = dataframe_to_dataset(acs_train_df)
    acs_train_batches = acs_train_ds.batch(BATCH_SIZE)

    base_model.fit(acs_train_batches, epochs=EPOCHS)

    # Use the indices from the training set to create the test set, which represents
    # 20% of the original dataset; then convert it to a tf.data.Dataset object, and
    # evaluate the base model using the converted test set.
    acs_test_df = acs_df.drop(acs_train_df.index).sample(frac=1.0)
    acs_test_ds = dataframe_to_dataset(acs_test_df)
    acs_test_batches = acs_test_ds.batch(BATCH_SIZE)

    base_model.evaluate(acs_test_batches, batch_size=BATCH_SIZE)

    SENSITIVE_ATTRIBUTE_VALUES = {1.0: "Male", 2.0: "Female"}
    SENSITIVE_ATTRIBUTE_KEY = 'SEX'
    PREDICTION_KEY = 'PRED'

    # Make a copy of the test set, replace sensitive attribute values with
    # categorial strings (for ease of visualization), and add predictions
    # from the test set to the copied DataFrame as a separate column.
    base_model_analysis = acs_test_df.copy()
    base_model_analysis[SENSITIVE_ATTRIBUTE_KEY].replace(
        SENSITIVE_ATTRIBUTE_VALUES, inplace=True)
    base_model_predictions = base_model.predict(acs_test_batches, batch_size=BATCH_SIZE)
    base_model_analysis[PREDICTION_KEY] = base_model_predictions

    # Show five random examples to ensure that it looks correct.
    base_model_analysis.sample(5)

    #@title Double-Click to View a Possible Answer { display-mode: "form" }

    # A pandas DataFrame offers many approaches when it comes to indexing and
    # selecting rows. One approach is by using boolean indexing
    # (e.g., df[df['col'] == value]) as demonstrated in the following code:
    sensitive_group_pos = acs_train_df[
        (acs_train_df[SENSITIVE_ATTRIBUTE_KEY] == 2.0) & (acs_train_df[LABEL_KEY] == 1)]
    non_sensitive_group_pos = acs_train_df[
        (acs_train_df[SENSITIVE_ATTRIBUTE_KEY] == 1.0) & (acs_train_df[LABEL_KEY] == 1)]

    # To learn more, visit: https://pandas.pydata.org/docs/user_guide/indexing.html

    print(len(sensitive_group_pos),
        'positively labeled sensitive group examples')
    print(len(non_sensitive_group_pos),
        'positively labeled non-sensitive group examples')

    # Convert sensitive and non-sensitive subsets into tf.data.Dataset.
    MIN_DIFF_BATCH_SIZE = 50
    sensitive_group_ds = dataframe_to_dataset(sensitive_group_pos)
    non_sensitive_group_ds = dataframe_to_dataset(non_sensitive_group_pos)

    # Batch the subsets.
    sensitive_group_batches = sensitive_group_ds.batch(
        MIN_DIFF_BATCH_SIZE, drop_remainder=True)
    non_sensitive_group_batches = non_sensitive_group_ds.batch(
        MIN_DIFF_BATCH_SIZE, drop_remainder=True)

    # All you have to do is include each of the three batched tf.data.Datasets into
    # the arguments: the training set, the sensitive subset (female) and
    # the non-sensitive subset (male).
    acs_train_min_diff_ds = min_diff.keras.utils.pack_min_diff_data(
        original_dataset = acs_train_batches,
        sensitive_group_dataset = sensitive_group_batches,
        nonsensitive_group_dataset = non_sensitive_group_batches)

    # Wrap the original model in a MinDiffModel.
    min_diff_model = min_diff.keras.MinDiffModel(
        original_model=base_model,
        loss=min_diff.losses.MMDLoss(),
        loss_weight=1)

    # Compile the model after wrapping the original model.
    min_diff_model.compile(
        optimizer='adam',
        loss=tf.keras.losses.BinaryCrossentropy(from_logits=False),
        metrics=METRICS)

    # Train MinDiff model using the packed dataset instead of the training set.
    min_diff_model.fit(acs_train_min_diff_ds, epochs=EPOCHS)

    min_diff_model.evaluate(acs_test_batches, batch_size=BATCH_SIZE)

    # Generate MinDiff output predictions using the test set.
    min_diff_model_predictions = min_diff_model.predict(
        acs_test_batches, batch_size=BATCH_SIZE)

if __name__ == "__main__":
    main()
