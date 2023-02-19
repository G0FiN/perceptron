using Newtonsoft.Json;

namespace Perceptron
{
    class Data
    {
        public List<List<double>>? Values { get; set; }
        public List<double>? Weights { get; set; }
    }

    class Program
    {
        static void Main()
        {
            Console.WriteLine("Perceptron is running...");
            Data? data = JsonConvert.DeserializeObject<Data>(File.ReadAllText("data.json"));
            List<double> weights = data.Weights ?? new List<double> { 0, 0, 0 };

            for (int epoch = 1; epoch <= 10; epoch++)
            {
                Console.WriteLine($"Epoch {epoch}");

                foreach (var item in data.Values)
                {
                    Console.WriteLine($"Training on input: {string.Join(",", item)}");
                    Correction(item, weights);
                }

                Console.WriteLine($"Weights after epoch {epoch}: A={weights[0]:0.00}, B={weights[1]:0.00}, C={weights[2]:0.00}");
            }

            Console.WriteLine("\nTraining completed.\nWeights: \n\tA={0:0.000}, \n\tB={1:0.000}, \n\tC={2:0.000}", weights[0], weights[1], weights[2]);
            Console.WriteLine("Press any key to exit...");
            Console.ReadKey();
        }

        static void Correction(List<double> item, List<double> weights)
        {
            double output = ActivationFunc(item, weights);
            double targetOutput = item[item.Count - 1];
            double error = targetOutput - output;
            double learningRate = 0.15;

            List<double> deltaWeights = new List<double>();
            for (int i = 0; i < weights.Count - 1; i++)
            {
                double deltaWeight = learningRate * item[i] * error;
                deltaWeights.Add(deltaWeight);
                weights[i] += deltaWeight;
            }

            double deltaBias = learningRate * error;
            deltaWeights.Add(deltaBias);
            weights[weights.Count - 1] += deltaBias;
        }

        static int ActivationFunc(List<double> input, List<double> weights)
        {
            double net = weights[weights.Count - 1];
            for (int i = 0; i < input.Count - 1; i++) {
                net += input[i] * weights[i];
            }
            return net >= 0 ? 1 : -1;
        }
    }
}
