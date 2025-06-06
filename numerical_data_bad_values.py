# @title Setup - Import relevant modules
import pandas as pd
from matplotlib import pyplot as plt
import io

# The following lines adjust the granularity of reporting.
pd.options.display.max_rows = 10
pd.options.display.float_format = "{:.1f}".format

#@title Define the plotting functions { display-mode: "form" }

# The following code defines the plotting functions that can be used to
# visualize the data.

def plot_the_dataset(training_df, feature, label, number_of_points_to_plot):
  """Plot N random points of the dataset."""

  # Label the axes.
  plt.xlabel(feature)
  plt.ylabel(label)

  # Create a scatter plot from n random points of the dataset.
  random_examples = training_df.sample(n=number_of_points_to_plot)
  plt.scatter(random_examples[feature], random_examples[label])

  # Render the scatter plot.
  plt.show()

def plot_a_contiguous_portion_of_dataset(training_df, feature, label, start, end):
  """Plot the data points from start to end."""

  # Label the axes.
  plt.xlabel(feature + "Day")
  plt.ylabel(label)

  # Create a scatter plot.
  plt.scatter(training_df[feature][start:end], training_df[label][start:end])

  # Render the scatter plot.
  plt.show()


training_df = pd.read_csv('dataset.csv', on_bad_lines='warn')

def main():
    """Main is entry point for the script."""

    # The following code returns basic statistics about the data in the dataframe.
    print("The basic data statistics are:", training_df.describe())

    plot_the_dataset(training_df,"calories", "test_score", number_of_points_to_plot=50)

    # Get statistics on Week 0
    print("The 0 week statistics are:", training_df[0:350].describe())
    # Get statistics on Week 1
    print("The 1 week statistics are:", training_df[350:700].describe())
    # Get statistics on Week 2
    print("The 2 week statistics are:", training_df[700:1050].describe())
    # Get statistics on Week 3
    print("The 3 week statistics are:", training_df[1050:1400].describe())

    for i in range(0,7):
        start = i * 50
        end = start + 49
        print("\nDay %d" % i)
        plot_a_contiguous_portion_of_dataset(training_df, "calories", "test_score", start, end)

    running_total_of_thursday_calories = 0
    running_total_of_non_thursday_calories = 0
    count = 0
    for week in range(0,4):
      for day in range(0,7):
        for subject in range(0,50):
          position = (week * 350) + (day * 50) + subject
          if (day == 4):  # Thursday
            running_total_of_thursday_calories += training_df['calories'][position]
          else:  # Any day except Thursday
            count += 1
            running_total_of_non_thursday_calories += training_df['calories'][position]

    mean_of_thursday_calories = running_total_of_thursday_calories / (position - count + 1)   
    mean_of_non_thursday_calories = running_total_of_non_thursday_calories / count

    print("The mean of Thursday calories is %.0f" % (mean_of_thursday_calories))
    print("The mean of calories on days other than Thursday is %.0f" % (mean_of_non_thursday_calories))

if __name__ == "__main__":
    main()
