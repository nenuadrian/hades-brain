#include <iostream>
#include <vector>
#include <cmath>
#include <algorithm>
#include <random>
#include <cassert>

#include <iostream>
#include <fstream>
#include <vector>
#include <cassert>

#include "argparse.hpp"

// Function to read integers from MNIST file (big-endian to little-endian).
int read_int(std::ifstream &file)
{
  unsigned char buffer[4];
  file.read(reinterpret_cast<char *>(buffer), 4);
  return (buffer[0] << 24) | (buffer[1] << 16) | (buffer[2] << 8) | buffer[3];
}

// Function to read MNIST image file.
std::vector<std::vector<double>> read_mnist_images(const std::string &filename)
{
  std::ifstream file(filename, std::ios::binary);
  assert(file.is_open() && "Failed to open file");

  int magic_number = read_int(file);
  assert(magic_number == 2051 && "Invalid image file");

  int num_images = read_int(file);
  int num_rows = read_int(file);
  int num_columns = read_int(file);

  std::vector<std::vector<double>> images(num_images, std::vector<double>(num_rows * num_columns));

  for (int i = 0; i < num_images; ++i)
  {
    for (int j = 0; j < num_rows * num_columns; ++j)
    {
      unsigned char pixel = file.get();
      images[i][j] = pixel / 255.0; // Normalize to [0, 1].
    }
  }

  file.close();
  return images;
}

// Function to read MNIST label file.
std::vector<int> read_mnist_labels(const std::string &filename)
{
  std::ifstream file(filename, std::ios::binary);
  assert(file.is_open() && "Failed to open file");

  int magic_number = read_int(file);
  assert(magic_number == 2049 && "Invalid label file");

  int num_labels = read_int(file);

  std::vector<int> labels(num_labels);
  for (int i = 0; i < num_labels; ++i)
  {
    unsigned char label = file.get();
    labels[i] = label;
  }

  file.close();
  return labels;
}

// Define sigmoid activation function and its derivative.
double sigmoid(double x)
{
  return 1.0 / (1.0 + std::exp(-x));
}

double sigmoid_derivative(double x)
{
  double sig = sigmoid(x);
  return sig * (1 - sig);
}

// Define the Neural Network class.
class NeuralNetwork
{
private:
  std::vector<std::vector<double>> weights1, weights2;
  std::vector<double> biases1, biases2;

  // Initialize weights and biases randomly.
  void initialize_weights()
  {
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_real_distribution<> dis(-1.0, 1.0);

    for (auto &row : weights1)
      for (auto &val : row)
        val = dis(gen);
    for (auto &val : biases1)
      val = dis(gen);

    for (auto &row : weights2)
      for (auto &val : row)
        val = dis(gen);
    for (auto &val : biases2)
      val = dis(gen);
  }

public:
  NeuralNetwork(size_t input_size, size_t hidden_size, size_t output_size)
      : weights1(hidden_size, std::vector<double>(input_size)),
        weights2(output_size, std::vector<double>(hidden_size)),
        biases1(hidden_size),
        biases2(output_size)
  {
    initialize_weights();
  }

  void load_from_file(const std::string &filename)
  {
    std::ifstream file(filename, std::ios::binary);
    if (!file.is_open())
    {
      throw std::runtime_error("Failed to open file for loading.");
    }

    // Load the dimensions of weights and biases
    size_t input_size, hidden_size, output_size;

    file.read(reinterpret_cast<char *>(&input_size), sizeof(size_t));
    file.read(reinterpret_cast<char *>(&hidden_size), sizeof(size_t));
    file.read(reinterpret_cast<char *>(&output_size), sizeof(size_t));

    // Resize the weights and biases to match the loaded dimensions
    weights1.resize(hidden_size, std::vector<double>(input_size));
    biases1.resize(hidden_size);
    weights2.resize(output_size, std::vector<double>(hidden_size));
    biases2.resize(output_size);

    // Load weights1
    for (auto &row : weights1)
    {
      file.read(reinterpret_cast<char *>(row.data()), row.size() * sizeof(double));
    }

    // Load biases1
    file.read(reinterpret_cast<char *>(biases1.data()), biases1.size() * sizeof(double));

    // Load weights2
    for (auto &row : weights2)
    {
      file.read(reinterpret_cast<char *>(row.data()), row.size() * sizeof(double));
    }

    // Load biases2
    file.read(reinterpret_cast<char *>(biases2.data()), biases2.size() * sizeof(double));

    file.close();
    std::cout << "Model loaded from " << filename << std::endl;
  }

#include <fstream>

  void save_to_file(const std::string &filename) const
  {
    std::ofstream file(filename, std::ios::binary);
    if (!file.is_open())
    {
      throw std::runtime_error("Failed to open file for saving.");
    }

    // Save the dimensions of weights and biases
    size_t input_size = weights1[0].size();
    size_t hidden_size = weights1.size();
    size_t output_size = weights2.size();

    file.write(reinterpret_cast<const char *>(&input_size), sizeof(size_t));
    file.write(reinterpret_cast<const char *>(&hidden_size), sizeof(size_t));
    file.write(reinterpret_cast<const char *>(&output_size), sizeof(size_t));

    // Save weights1
    for (const auto &row : weights1)
    {
      file.write(reinterpret_cast<const char *>(row.data()), row.size() * sizeof(double));
    }

    // Save biases1
    file.write(reinterpret_cast<const char *>(biases1.data()), biases1.size() * sizeof(double));

    // Save weights2
    for (const auto &row : weights2)
    {
      file.write(reinterpret_cast<const char *>(row.data()), row.size() * sizeof(double));
    }

    // Save biases2
    file.write(reinterpret_cast<const char *>(biases2.data()), biases2.size() * sizeof(double));

    file.close();
    std::cout << "Model saved to " << filename << std::endl;
  }

  // Forward pass.
  std::pair<std::vector<double>, std::vector<double>> forward(const std::vector<double> &inputs)
  {
    std::vector<double> hidden_layer(weights1.size());
    std::vector<double> output_layer(weights2.size());

    for (size_t i = 0; i < weights1.size(); ++i)
    {
      hidden_layer[i] = biases1[i];
      for (size_t j = 0; j < inputs.size(); ++j)
        hidden_layer[i] += weights1[i][j] * inputs[j];
      hidden_layer[i] = sigmoid(hidden_layer[i]);
    }

    for (size_t i = 0; i < weights2.size(); ++i)
    {
      output_layer[i] = biases2[i];
      for (size_t j = 0; j < hidden_layer.size(); ++j)
        output_layer[i] += weights2[i][j] * hidden_layer[j];
      output_layer[i] = sigmoid(output_layer[i]);
    }

    return {hidden_layer, output_layer};
  }

  // Backward pass and weights update.
  void train(const std::vector<double> &inputs, const std::vector<double> &targets, double learning_rate)
  {
    auto [hidden_layer, outputs] = forward(inputs);

    std::vector<double> output_errors(outputs.size());
    std::vector<double> output_deltas(outputs.size());

    for (size_t i = 0; i < outputs.size(); ++i)
    {
      output_errors[i] = targets[i] - outputs[i];
      output_deltas[i] = output_errors[i] * sigmoid_derivative(outputs[i]);
    }

    std::vector<double> hidden_errors(hidden_layer.size());
    std::vector<double> hidden_deltas(hidden_layer.size());

    for (size_t i = 0; i < hidden_layer.size(); ++i)
    {
      hidden_errors[i] = 0;
      for (size_t j = 0; j < output_deltas.size(); ++j)
        hidden_errors[i] += output_deltas[j] * weights2[j][i];
      hidden_deltas[i] = hidden_errors[i] * sigmoid_derivative(hidden_layer[i]);
    }

    // Update weights and biases for weights2 and biases2.
    for (size_t i = 0; i < weights2.size(); ++i)
    {
      for (size_t j = 0; j < hidden_layer.size(); ++j)
        weights2[i][j] += learning_rate * output_deltas[i] * hidden_layer[j];
      biases2[i] += learning_rate * output_deltas[i];
    }

    // Update weights and biases for weights1 and biases1.
    for (size_t i = 0; i < weights1.size(); ++i)
    {
      for (size_t j = 0; j < inputs.size(); ++j)
        weights1[i][j] += learning_rate * hidden_deltas[i] * inputs[j];
      biases1[i] += learning_rate * hidden_deltas[i];
    }
  }

  // Predict function.
  std::vector<double> predict(const std::vector<double> &inputs)
  {
    return forward(inputs).second;
  }

  void print_table_view() const
  {
    std::cout << "Neural Network Table View\n";
    std::cout << "--------------------------\n";

    // Print weights and biases of the first layer
    std::cout << "\nLayer 1: Input to Hidden\n";
    std::cout << "Weights:\n";
    for (size_t i = 0; i < weights1.size(); ++i)
    {
      std::cout << "Neuron " << i + 1 << ": ";
      for (size_t j = 0; j < weights1[i].size(); ++j)
      {
        std::cout << std::fixed << std::setprecision(2) << weights1[i][j] << " ";
      }
      std::cout << "\n";
    }
    std::cout << "Biases:\n";
    for (size_t i = 0; i < biases1.size(); ++i)
    {
      std::cout << "Neuron " << i + 1 << ": " << std::fixed << std::setprecision(2) << biases1[i] << "\n";
    }

    // Print weights and biases of the second layer
    std::cout << "\nLayer 2: Hidden to Output\n";
    std::cout << "Weights:\n";
    for (size_t i = 0; i < weights2.size(); ++i)
    {
      std::cout << "Neuron " << i + 1 << ": ";
      for (size_t j = 0; j < weights2[i].size(); ++j)
      {
        std::cout << std::fixed << std::setprecision(2) << weights2[i][j] << " ";
      }
      std::cout << "\n";
    }
    std::cout << "Biases:\n";
    for (size_t i = 0; i < biases2.size(); ++i)
    {
      std::cout << "Neuron " << i + 1 << ": " << std::fixed << std::setprecision(2) << biases2[i] << "\n";
    }
  }
};

std::vector<double> one_hot_encode(int label, int num_classes = 10)
{
  std::vector<double> encoded(num_classes, 0.0);
  encoded[label] = 1.0; // Set the correct class to 1.0
  return encoded;
}

void train()
{
  // Replace these with the paths to your MNIST files.
  const std::string train_images_file = "MNIST_ORG/train-images.idx3-ubyte";
  const std::string train_labels_file = "MNIST_ORG/train-labels.idx1-ubyte";

  // Read the MNIST training data.
  auto train_images = read_mnist_images(train_images_file);
  auto train_labels = read_mnist_labels(train_labels_file);

  // Print an example image and its label.
  std::cout << "First image label: " << train_labels[0] << std::endl;
  std::cout << "First image pixels (normalized): ";
  for (int i = 0; i < 28; ++i)
  { // Assuming 28x28 images.
    for (int j = 0; j < 28; ++j)
    {
      std::cout << (train_images[0][i * 28 + j] > 0.5 ? '#' : '.');
    }
    std::cout << std::endl;
  }

  // Define network parameters.
  const size_t input_size = 784;  // MNIST image size: 28x28 pixels.
  const size_t hidden_size = 128; // Hidden layer size.
  const size_t output_size = 10;  // Output size: 10 digits.

  NeuralNetwork nn(input_size, hidden_size, output_size);

  // Train the network.
  const size_t epochs = 5;
  const double learning_rate = 0.01;

  for (size_t epoch = 0; epoch < epochs; ++epoch)
  {
    for (size_t i = 0; i < train_images.size(); ++i)
    {
      // Get the input image and its corresponding target label
      std::vector<double> input = train_images[i];
      std::vector<double> target = one_hot_encode(train_labels[i]);

      // Train the network
      nn.train(input, target, learning_rate);
    }

    std::cout << "Epoch " << epoch + 1 << "/" << epochs << " completed." << std::endl;
    nn.save_to_file("mnist_model.dat");
    nn.print_table_view();
  }
}

void predict()
{
  // Replace these with the paths to your MNIST files.
  const std::string test_images_file = "MNIST_ORG/t10k-images.idx3-ubyte";
  const std::string test_labels_file = "MNIST_ORG/t10k-labels.idx1-ubyte";

  // Read the MNIST test data.
  auto test_images = read_mnist_images(test_images_file);
  auto test_labels = read_mnist_labels(test_labels_file);

  // Define network parameters.
  const size_t input_size = 784;  // MNIST image size: 28x28 pixels.
  const size_t hidden_size = 128; // Hidden layer size.
  const size_t output_size = 10;  // Output size: 10 digits.

  NeuralNetwork nn(input_size, hidden_size, output_size);
  nn.load_from_file("mnist_model.dat");
  nn.print_table_view();

  size_t correct_predictions = 0;

  // Iterate over the test dataset
  for (size_t i = 0; i < test_images.size(); ++i)
  {
    // Get the input image
    std::vector<double> input = test_images[i];

    // Get the true label
    int true_label = test_labels[i];

    // Predict the label
    std::vector<double> output = nn.predict(input);

    // Find the index of the maximum value in the output (predicted label)
    int predicted_label = std::distance(output.begin(), std::max_element(output.begin(), output.end()));

    // Check if the prediction is correct
    if (predicted_label == true_label)
    {
      ++correct_predictions;
    }
  }

  // Calculate accuracy
  double accuracy = (static_cast<double>(correct_predictions) / test_images.size()) * 100.0;

  std::cout << "Accuracy: " << accuracy << "%" << std::endl;
}

int main(int argc, char **argv)
{
  argparse::ArgumentParser program("nn");

  program.add_argument("--train")
      .help("train")
      .default_value(false)
      .implicit_value(true);

  try
  {
    program.parse_args(argc, argv);
  }
  catch (const std::exception &err)
  {
    std::cerr << err.what() << std::endl;
    std::cerr << program;
    return 1;
  }

  auto is_train = program.get<bool>("train");

  if (is_train)
  {
    train();
  }
  else
  {
    predict();
  }

  return 0;
}
