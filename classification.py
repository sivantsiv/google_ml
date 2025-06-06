#@title Binary Classification

import keras
import ml_edu.experiment
import ml_edu.results
import numpy as np
import pandas as pd
import plotly.express as px

# The following lines adjust the granularity of reporting.
pd.options.display.max_rows = 10
pd.options.display.float_format = "{:.1f}".format


# Define the functions that create and train a model.
def create_model(
    settings: ml_edu.experiment.ExperimentSettings,
    metrics: list[keras.metrics.Metric],
) -> keras.Model:
  """Create and compile a simple classification model."""
  model_inputs = [
      keras.Input(name=feature, shape=(1,))
      for feature in settings.input_features
  ]
  # Use a Concatenate layer to assemble the different inputs into a single
  # tensor which will be given as input to the Dense layer.
  # For example: [input_1[0][0], input_2[0][0]]

  concatenated_inputs = keras.layers.Concatenate()(model_inputs)
  model_output = keras.layers.Dense(
      units=1, name='dense_layer', activation=keras.activations.sigmoid
  )(concatenated_inputs)
  model = keras.Model(inputs=model_inputs, outputs=model_output)
  # Call the compile method to transform the layers into a model that
  # Keras can execute.  Notice that we're using a different loss
  # function for classification than for regression.
  model.compile(
      optimizer=keras.optimizers.RMSprop(
          settings.learning_rate
      ),
      loss=keras.losses.BinaryCrossentropy(),
      metrics=metrics,
  )
  return model

def train_model(
    experiment_name: str,
    model: keras.Model,
    dataset: pd.DataFrame,
    labels: np.ndarray,
    settings: ml_edu.experiment.ExperimentSettings,
) -> ml_edu.experiment.Experiment:
  """Feed a dataset into the model in order to train it."""

  # The x parameter of keras.Model.fit can be a list of arrays, where
  # each array contains the data for one feature.
  features = {
      feature_name: np.array(dataset[feature_name])
      for feature_name in settings.input_features
  }

  history = model.fit(
      x=features,
      y=labels,
      batch_size=settings.batch_size,
      epochs=settings.number_epochs,
  )

  return ml_edu.experiment.Experiment(
      name=experiment_name,
      settings=settings,
      model=model,
      epochs=history.epoch,
      metrics_history=pd.DataFrame(history.history),
  )

def compare_train_test(experiment: ml_edu.experiment.Experiment, test_metrics: dict[str, float]):
  print('Comparing metrics between train and test:')
  for metric, test_value in test_metrics.items():
    print('------')
    print(f'Train {metric}: {experiment.get_final_metric_value(metric):.4f}')
    print(f'Test {metric}:  {test_value:.4f}')

def main():
    """Main is entry point for the script."""

    # Load the dataset
    rice_dataset_raw = pd.read_csv("https://download.mlcc.google.com/mledu-datasets/Rice_Cammeo_Osmancik.csv")

    # Read and provide statistics on the dataset.
    rice_dataset = rice_dataset_raw[[
        'Area',
        'Perimeter',
        'Major_Axis_Length',
        'Minor_Axis_Length',
        'Eccentricity',
        'Convex_Area',
        'Extent',
        'Class',
    ]]

    rice_dataset.describe()

    # Solutions (run the cell to get the answers)
    print(
        f'The shortest grain is {rice_dataset.Major_Axis_Length.min():.1f}px long,'
        f' while the longest is {rice_dataset.Major_Axis_Length.max():.1f}px.'
    )
    print(
        f'The smallest rice grain has an area of {rice_dataset.Area.min()}px, while'
        f' the largest has an area of {rice_dataset.Area.max()}px.'
    )
    print(
        'The largest rice grain, with a perimeter of'
        f' {rice_dataset.Perimeter.max():.1f}px, is'
        f' ~{(rice_dataset.Perimeter.max() - rice_dataset.Perimeter.mean())/rice_dataset.Perimeter.std():.1f} standard'
        f' deviations ({rice_dataset.Perimeter.std():.1f}) from the mean'
        f' ({rice_dataset.Perimeter.mean():.1f}px).'
    )
    print(
        f'This is calculated as: ({rice_dataset.Perimeter.max():.1f} -'
        f' {rice_dataset.Perimeter.mean():.1f})/{rice_dataset.Perimeter.std():.1f} ='
        f' {(rice_dataset.Perimeter.max() - rice_dataset.Perimeter.mean())/rice_dataset.Perimeter.std():.1f}'
    )

    # Create five 2D plots of the features against each other, color-coded by class.
    for x_axis_data, y_axis_data in [
        ('Area', 'Eccentricity'),
        ('Convex_Area', 'Perimeter'),
        ('Major_Axis_Length', 'Minor_Axis_Length'),
        ('Perimeter', 'Extent'),
        ('Eccentricity', 'Major_Axis_Length'),
    ]:
        px.scatter(rice_dataset, x=x_axis_data, y=y_axis_data, color='Class').show()

    # One possible solution

    # Plot major and minor axis length and eccentricity, with observations
    # color-coded by class.
    px.scatter_3d(
        rice_dataset,
        x='Eccentricity',
        y='Area',
        z='Major_Axis_Length',
        color='Class',
    ).show()

    # Calculate the Z-scores of each numerical column in the raw data and write
    # them into a new DataFrame named df_norm.

    feature_mean = rice_dataset.mean(numeric_only=True)
    feature_std = rice_dataset.std(numeric_only=True)
    numerical_features = rice_dataset.select_dtypes('number').columns
    normalized_dataset = (
        rice_dataset[numerical_features] - feature_mean
    ) / feature_std

    # Copy the class to the new dataframe
    normalized_dataset['Class'] = rice_dataset['Class']

    # Examine some of the values of the normalized training set. Notice that most
    # Z-scores fall between -2 and +2.
    normalized_dataset.head()

    keras.utils.set_random_seed(42)

    # Create a column setting the Cammeo label to '1' and the Osmancik label to '0'
    # then show 10 randomly selected rows.
    normalized_dataset['Class_Bool'] = (
        # Returns true if class is Cammeo, and false if class is Osmancik
        normalized_dataset['Class'] == 'Cammeo'
    ).astype(int)
    normalized_dataset.sample(10)

    # Create indices at the 80th and 90th percentiles
    number_samples = len(normalized_dataset)
    index_80th = round(number_samples * 0.8)
    index_90th = index_80th + round(number_samples * 0.1)

    # Randomize order and split into train, validation, and test with a .8, .1, .1 split
    shuffled_dataset = normalized_dataset.sample(frac=1, random_state=100)
    train_data = shuffled_dataset.iloc[0:index_80th]
    validation_data = shuffled_dataset.iloc[index_80th:index_90th]
    test_data = shuffled_dataset.iloc[index_90th:]

    # Show the first five rows of the last split
    test_data.head()

    label_columns = ['Class', 'Class_Bool']

    train_features = train_data.drop(columns=label_columns)
    train_labels = train_data['Class_Bool'].to_numpy()
    validation_features = validation_data.drop(columns=label_columns)
    validation_labels = validation_data['Class_Bool'].to_numpy()
    test_features = test_data.drop(columns=label_columns)
    test_labels = test_data['Class_Bool'].to_numpy()

    # Name of the features we'll train our model on.
    input_features = [
        'Eccentricity',
        'Major_Axis_Length',
        'Area',
    ]

    # Let's define our first experiment settings.
    settings = ml_edu.experiment.ExperimentSettings(
        learning_rate=0.001,
        number_epochs=60,
        batch_size=100,
        classification_threshold=0.35,
        input_features=input_features,
    )

    metrics = [
        keras.metrics.BinaryAccuracy(
            name='accuracy', threshold=settings.classification_threshold
        ),
        keras.metrics.Precision(
            name='precision', thresholds=settings.classification_threshold
        ),
        keras.metrics.Recall(
            name='recall', thresholds=settings.classification_threshold
        ),
        keras.metrics.AUC(num_thresholds=100, name='auc'),
    ]

    # Establish the model's topography.
    model = create_model(settings, metrics)

    # Train the model on the training set.
    experiment = train_model(
        'baseline', model, train_features, train_labels, settings
    )

    # Plot metrics vs. epochs
    ml_edu.results.plot_experiment_metrics(experiment, ['accuracy', 'precision', 'recall'])
    ml_edu.results.plot_experiment_metrics(experiment, ['auc'])

    # Evaluate test metrics
    test_metrics = experiment.evaluate(test_features, test_labels)
    compare_train_test(experiment, test_metrics)

    # Features used to train the model on.
    # Specify all features.
    all_input_features = [
    'Eccentricity',
    'Major_Axis_Length',
    'Minor_Axis_Length',
    'Area',
    'Convex_Area',
    'Perimeter',
    'Extent',
    ]

    settings_all_features = ml_edu.experiment.ExperimentSettings(
        learning_rate=0.001,
        number_epochs=60,
        batch_size=100,
        classification_threshold=0.5,
        input_features=all_input_features,
    )

    # Modify the following definition of METRICS to generate
    # not only accuracy and precision, but also recall:
    metrics = [
        keras.metrics.BinaryAccuracy(
            name='accuracy',
            threshold=settings_all_features.classification_threshold,
        ),
        keras.metrics.Precision(
            name='precision',
            thresholds=settings_all_features.classification_threshold,
        ),
        keras.metrics.Recall(
            name='recall', thresholds=settings_all_features.classification_threshold
        ),
        keras.metrics.AUC(num_thresholds=100, name='auc'),
    ]

    # Establish the model's topography.
    model_all_features = create_model(settings_all_features, metrics)

    # Train the model on the training set.
    experiment_all_features = train_model(
        'all features',
        model_all_features,
        train_features,
        train_labels,
        settings_all_features,
    )

    # Plot metrics vs. epochs
    ml_edu.results.plot_experiment_metrics(
        experiment_all_features, ['accuracy', 'precision', 'recall']
    )
    ml_edu.results.plot_experiment_metrics(experiment_all_features, ['auc'])

    settings_all_features = ml_edu.experiment.ExperimentSettings(
        learning_rate=0.001,
        number_epochs=60,
        batch_size=100,
        classification_threshold=0.5,
        input_features=all_input_features,
    )

    # Modify the following definition of METRICS to generate
    # not only accuracy and precision, but also recall:
    metrics = [
        keras.metrics.BinaryAccuracy(
            name='accuracy',
            threshold=settings_all_features.classification_threshold,
        ),
        keras.metrics.Precision(
            name='precision',
            thresholds=settings_all_features.classification_threshold,
        ),
        keras.metrics.Recall(
            name='recall', thresholds=settings_all_features.classification_threshold
        ),
        keras.metrics.AUC(num_thresholds=100, name='auc'),
    ]

    # Establish the model's topography.
    model_all_features = create_model(settings_all_features, metrics)

    # Train the model on the training set.
    experiment_all_features = train_model(
        'all features',
        model_all_features,
        train_features,
        train_labels,
        settings_all_features,
    )

    # Plot metrics vs. epochs
    ml_edu.results.plot_experiment_metrics(
        experiment_all_features, ['accuracy', 'precision', 'recall']
    )
    ml_edu.results.plot_experiment_metrics(experiment_all_features, ['auc'])

    ml_edu.results.compare_experiment([experiment, experiment_all_features],
                                    ['accuracy', 'auc'],
                                    test_features, test_labels)

if __name__ == "__main__":
    main()
