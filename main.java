/**
 *
 * Multi-Layer Perceptron program (mlp class and calling main routine)
 *
 * @author Jan Hendriks
 * @date 2009-11-08
 * @copyright GPLv3 2010 Jan Hendriks
 * @url http://gawe-design.de
 *
 * Compiled and tested with a recent Sun Microsystems JDK: Java(TM) SE Runtime Environment (build 1.6.0_16-b01) and JRE 6u17 under linux x64
 * 
 * All function and variable names should be self-explanatory
 *
 * Usage:
 * Compile with e.g.
 * $ javac main.java
 * Make sure the 'train.dat' file resides in the same directory as the main.class
 * Also, the directory has to be writable for the 'epocheerror.dat'
 * Execute the program with e.g.
 * $ java main
 * Or alternatively, if it generates too much output
 * $ java main > /dev/null
 *
 */

// Imports
import java.io.*;
import java.util.Random.*;
import java.lang.Math.*;

class mlp {

    // Using nested inner classes to avoid some inheritance issues
    public class layer {

        // Using nested inner classes to avoid some inheritance issues
        public class neuron {

            private int neuronId; // unique id, e.g. for convenience or PRNG order
            private double weightedSum; // Store the calculated net for each neuron here
            private double output; // Contains the output of the neuron object
            private double delta; // The delta of the delta rule, not the weight change, assigned to every neuron

            // Constructor for neuron object, it only assigns an identifier to each neuron
            private neuron (int id) {
                System.out.println( (id == 0) ? " > Creating BIAS neuron in current layer" : " > Creating neuron " + id);
                this.neuronId = id; // Set the current last neuron number in MLP
            }

            // This is the call to the calculations in the neuron object, to make sure the calculations are only issued one per neuron (mlp) input
            private void calculateWeightedSumAndOutput() {
                this.calculateWeightedSum();
                this.calculateOutput();
            }

            // The calculates the weighted sum for the current neuron object
            private void calculateWeightedSum() {
                if (this.neuronId == 0) {
                    this.weightedSum = 1; // If BIAS neuron
                } else {
                    if (layerId == 0) { // If input layer, the weighted sum and output is merely the input itself, since the input layer acts only as buffer
                        this.weightedSum = inputValues[this.neuronId-1];
                    } else {
                        // Prepare to walk through all neurons in the above layer
                        neuron[] aboveLayerNeurons = layers[(layerId-1)].layerNeurons;
                        // Initialize the weighted sum with the bias weight from the bias to the current neuron
                        double weightedSumAccumulator = getWeightFromTo(layers[(layerId-1)].BIAS,this); // negative threshold, init net with Bias weight
                        System.out.format(" 1      * W(  0>%3d) = %+3.5f * %+3.5f%n", this.neuronId, (double)1, weightedSumAccumulator);
                        for (int i = 0; i < aboveLayerNeurons.length; i++) {
                            // This is the weight from one above neuron to the current neuron object
                            double weightToCurrentNeuronFromAboveNeuron = getWeightFromTo(aboveLayerNeurons[i],this);
                            // This is the output of one neuron in the above layer of the current layer
                            double outputOfNeuronInAboveLayer = aboveLayerNeurons[i].getOutput();
                            System.out.format(" O(%3d) * W(%3d>%3d) = %+3.5f * %+3.5f%n", aboveLayerNeurons[i].neuronId , aboveLayerNeurons[i].neuronId, this.neuronId, weightToCurrentNeuronFromAboveNeuron, outputOfNeuronInAboveLayer);
                            // This calculates one summand (w_ij * o_i) for the weighted sum and adds it to the previous summands
                            weightedSumAccumulator += weightToCurrentNeuronFromAboveNeuron * outputOfNeuronInAboveLayer;
                        }
                        // After walking through all neurons in the above layer (and the bias) and adding up the summands, we have the weighted sum
                        this.weightedSum = weightedSumAccumulator;
                    }
                    System.out.format("Calculated net for neuron %6d: %+3.5f%n", this.neuronId, this.weightedSum);
                }
            }

            // This calculates what the current neuron object outputs
            private void calculateOutput() {
                if (layerId == 0) {
                    this.output = this.weightedSum; // If input layer, the weighted sum and output is merely the input itself (same for BIAS)
                } else {
                    // Otherwise calculate the value of the specified transferfunction at the weighted sum, e.g. output = transferfunction(weightedsum)
                    switch(transferFunction) {
                        case tanh:
                            this.output = Math.tanh(this.weightedSum); // Output is the tanh function at the spot of the weighted sum
                            break;
                        case fermi:
                            this.output = 1 / (1 + Math.exp( -(this.weightedSum))); // Output is the fermi function at the spot of the weighted sum
                            break;
                        default:
                        case id:
                            this.output = this.weightedSum; // In case of the identity, the output is identical to the weighted sum
                    }
                }
                System.out.format("Calculated output for neuron %3d: %+3.5f%n", this.neuronId, this.output);
            }

            // Returns the value of the derivations of the given layer transfer functions at z
            private double getDerivationAt(double z) {
                switch(transferFunction) {
                    case tanh:
                        double tanhTempResult = Math.tanh(z); // Do the calculations in a half-intelligent way to avoid the same calculations more than once
                        return 1 - Math.pow(tanhTempResult,2); // 1-tanh^2(z)
                    case fermi:
                        double fermiTempResult = 1 / (1 + Math.exp( -(z))); // Do the calculations in a half-intelligent way to avoid the same calculations more than once
                        return fermiTempResult*(1-fermiTempResult); // (f(1-f))(z)
                    default:
                    case id:
                        return 1; // The derivation of the identity function is always 1: id(x) = x => id'(x) = 1
                }
            }

            // Calculates the delta of the delta rule, not the weight change
            private void calculateDelta() {
                // Test for output layer
                if (layerId == (layerCount-1)) {
                    // Exclude the bias neuron in the output layer
                    if (this.neuronId != 0) {
                        // Calculate delta for the output neurons (teacher y(m) - output y(m)) * f'(net)
                        this.delta = (teacherOutput[(this.neuronId-getFirstNeuronId())] - this.getOutput()) * this.getDerivationAt(this.getWeightedSum());
                    }
                } else if (layerId != 0) { // failsafe to exclude the input layer
                    // Prepare to walk through the neurons in the below layer
                    layer belowLayer = layers[layerId+1];
                    double deltaAccumulator = 0;
                    // Walk through neurons in the below layer
                    for (int i = 0; i < belowLayer.getLayerSize(); i++) {
                        // Sums up the (weights*deltas) from this neuron to the neurons in the below layers
                        deltaAccumulator += getWeightFromTo(this,belowLayer.layerNeurons[i]) * belowLayer.layerNeurons[i].getDelta();
                        System.out.format(" delta += W(%3d>%3d) * delta (%3d) = %+3.5f * %+3.5f%n",this.neuronId, belowLayer.layerNeurons[i].neuronId, belowLayer.layerNeurons[i].neuronId, getWeightFromTo(this, belowLayer.layerNeurons[i]), belowLayer.layerNeurons[i].getDelta());
                    }
                    this.delta = deltaAccumulator * this.getDerivationAt(this.getWeightedSum());
                }
                System.out.format(" f'(%+3.5f) = %+3.5f%n", this.getWeightedSum(), this.getDerivationAt(this.getWeightedSum()));
                System.out.format("Delta of neuron %3d in layer %3d: %+3.5f%n", this.neuronId, layerId, this.delta);
            }

            // Returns the id of the layer in which the neuron resides
            public short getLayerId() {
                return layerId;
            }

            // Returns the weighted sum (=net) for the neuron
            public double getWeightedSum() {
                return this.weightedSum;
            }

            // Returns the output for the neuron
            public double getOutput() {
                return this.output;
            }

            // Returns the delta for the neuron
            public double getDelta() {
                return this.delta;
            }
        }

        private neuron[] layerNeurons; // The neurons per layer excluding the bias neuron
        private neuron BIAS; // The bias neuron per layer
        private short layerId; // layerId == 0 means input (=first) layer
        private Transferfunction transferFunction; // The transfer function per layer
        private int weightRowOffset; // row offset in the weight matrix in current layer, used internally
        private int firstNeuronIdInLayer; // The id of the first neuron in the layer (excluding the bias neuron with id always=0 in every layer)
        private double learningRate; // The learning rate per layer, also denoted eta
        
        // Constructor for first layer (no above layer exists) -> not allowed to be called directly, use mlp.addLayer() instead!
        private layer(int layerSize, Transferfunction transf, double eta, short layerId) {
            System.out.println( ((layerId == 0) ? " > Creating input layer of size " + layerSize : " > Creating layer " + layerId + " of size " + layerSize)+ " with " + transf + " transfer function");
            this.layerId = layerId;
            this.transferFunction = transf;
            this.learningRate = eta;
            this.BIAS = new neuron(0); // BIAS neuron (each layer gets one, b/c neurons always affiliate to a layer)
            this.layerNeurons = new neuron[layerSize];
            this.firstNeuronIdInLayer = lastNeuronNumber;
            for (int i = 0; i < layerSize; i++) {
                this.layerNeurons[i] = new neuron(lastNeuronNumber++);
            }
            this.calculateRowOffsetInWeightMatrix();

        }

        // Converts w_ij to corresponding matrix position and returns x,y in an array
        private int[] getWeightMatrixPositionForNeuronWeightFromTo(neuron fromNeuron, neuron toNeuron) {
            int[] weightMatrixPosition = new int[2];
            if (fromNeuron.neuronId == 0) { // because from neuron 0 = from BIAS neuron, ambigious for each layer
                weightMatrixPosition[0] = layers[(toNeuron.getLayerId() - 1)].getRowOffsetInWeightMatrix();
            } else {
                weightMatrixPosition[0] = layers[fromNeuron.getLayerId()].getRowOffsetInWeightMatrix() + 1 + (fromNeuron.neuronId - layers[fromNeuron.getLayerId()].getFirstNeuronId());
            }
            weightMatrixPosition[1] = (toNeuron.neuronId - layers[toNeuron.getLayerId()].getFirstNeuronId());
            return weightMatrixPosition;
        }

        // Convert w_i_j to matrix row & column and return weight w_i_j from weightMatrix
        private double getWeightFromTo(neuron fromNeuron, neuron toNeuron) {
            // First calculate the position of the weight w_i_j in the weight matrix
            int[] weightMatrixPosition = this.getWeightMatrixPositionForNeuronWeightFromTo(fromNeuron, toNeuron);
            // Now return the weight
            return weightMatrix[weightMatrixPosition[0]][weightMatrixPosition[1]];
        }

        // Internal helping routine, calculates the row offset for the neuron weights in the weight matrix for each layer (recursive)
        private void calculateRowOffsetInWeightMatrix() {
            if (this.layerId < 1) {
                this.weightRowOffset = 0;
            } else {
                this.weightRowOffset = layers[(this.layerId-1)].getLayerSize() + 1 + layers[(this.layerId-1)].getRowOffsetInWeightMatrix();
           }
        }

        // Internal helping routine to help convert the neuron weights to the row-column entry in the matrix
        private int getRowOffsetInWeightMatrix() {
            return this.weightRowOffset;
        }

        // Returns the id of the first neuron in the layer (excluding bias neuron with id 0 in every layer)
        private int getFirstNeuronId() {
            return this.firstNeuronIdInLayer;
        }

        // Returns the size excluding the bias neuron
        public int getLayerSize() {
            return this.layerNeurons.length;
        }

        // Collects and returns the output of all neurons in the layer
        private double[] getLayerOutput() {
            double[] outputCollector = new double[this.getLayerSize()];
            for (int j = 0; j < this.getLayerSize(); j++) {
                outputCollector[j] = this.layerNeurons[j].getOutput();
            }
            return outputCollector;
        }

    }

    // This represents the available choices of the transfer function per layer
    public enum Transferfunction {
        id,
        fermi,
        tanh
    }

    // 2-dim array = matrix (nearly in hinton diagram style without the leading 0's in the rows)
    private double[][] weightMatrix; // This matrix contains the weights in the mlp
    private double[][] differenceWeightMatrix; // For calculated weight changes
    private double[] inputValues; // x_1 .. x_N
    private double[] teacherOutput; // The teacher output teacher y_1 .. teacher y_M
    
    private int lastNeuronNumber = 1; // neurons of all layers have unique consecutive numbers (for id and PRNG)

    private short layerCount = 0;
    private layer[] layers = new layer[255]; // Assuming <255 layers (short)

    // Init with an input pattern and amount of input and output neurons
    public mlp (double[][] weightMatrixFromFile) {
        System.out.println(" > Initializing MLP");
        this.weightMatrix = weightMatrixFromFile;
        this.initDifferenceWeightMatrix();
    }

    // Sets input and initializes the MLP calculations (no lazy evaluation b/c all values will be referenced at some time)
    public void propagateInput(double[] inputVector) {
        this.inputValues = inputVector;
        // Init calculation here, propagate input (pass input forward through MLP)
        // Walk through all layers from input to output layer
        for (int currentLayer = 0; currentLayer < this.layerCount; currentLayer++) {
            // Walk through the neurons in the layer (except the bias neuron)
            for (int n = 0; n < this.layers[currentLayer].getLayerSize(); n++) {
                // This calls the actual calculation of the weighted sum and the output per neuron in the layer
                this.layers[currentLayer].layerNeurons[n].calculateWeightedSumAndOutput();
            }
        }
    }

    // Create one(!) input layer (has to be the first layer to be created)
    public void addLayer(int layerSize) {
        if (this.layerCount > 0) {
            System.err.println(" > Only one input layer allowed!");
        } else {
            // Init the input layer in a convenient way
            this.addLayer(layerSize,Transferfunction.id,0);
        }
    }

    // Create a hidden or output layer
    public void addLayer(int layerSize, Transferfunction trans, double eta) {
        this.layers[this.layerCount] = new layer(layerSize,trans,eta, this.layerCount++);
    }

    // Init the difference weight matrix (dimensions, the values are calculated anyhow and do not need explicit init)
    private void initDifferenceWeightMatrix() {
        this.differenceWeightMatrix = new double[this.weightMatrix.length][this.weightMatrix[0].length]; // Set same dimensions
    }

    // This is the backpropagation of delta learning algorithm
    public void backpropagationOfDelta() {
        // Walk through layers backwards, starting at output layer
        for (int b = this.layerCount-1; b > 0; b--) { // There are layerCount-1 layers in the mlp
            // Walk through all neurons in the layer
            for (int n = 0; n < this.layers[b].getLayerSize(); n++) {
                // Let each neuron calculate its delta
                this.layers[b].layerNeurons[n].calculateDelta();
                // Prepare to walk through the above layer neurons
                layer aboveLayer = this.layers[b-1];

                // First consider the bias neuron in the above layer
                // This only converts the w_0j to the corresponding weight entry in the weight matrix (row,column)=(weightPosition[0]][weightPosition[1])
                int[] weightPosition = aboveLayer.getWeightMatrixPositionForNeuronWeightFromTo(aboveLayer.BIAS,this.layers[b].layerNeurons[n]);
                // This is the actual weight change calculation (diff w_0j = eta_j * delta_j * o_0 = eta_j * delta_j)
                this.differenceWeightMatrix[weightPosition[0]][weightPosition[1]] = this.layers[b].learningRate * this.layers[b].layerNeurons[n].getDelta();
                System.out.format(" > Weight change for weight W(%3d>%3d): eta(%3d) * delta(%3d) * o(%3d) = %+3.5f * %+3.5f * %+3.5f = %+3.5f%n",0, this.layers[b].layerNeurons[n].neuronId, this.layers[b].layerNeurons[n].neuronId, this.layers[b].layerNeurons[n].neuronId, 0, this.layers[b].learningRate, this.layers[b].layerNeurons[n].getDelta(), (double)1, this.differenceWeightMatrix[weightPosition[0]][weightPosition[1]]);

                // Consider all other neurons in the above layer
                for (int aboveLayerNeurons = 0; aboveLayerNeurons < aboveLayer.getLayerSize(); aboveLayerNeurons++) {
                    // This only converts the w_ij to the corresponding weight entry in the weight matrix (row,column)=(weightPosition[0]][weightPosition[1])
                    weightPosition = aboveLayer.getWeightMatrixPositionForNeuronWeightFromTo(aboveLayer.layerNeurons[aboveLayerNeurons],this.layers[b].layerNeurons[n]);
                    // This is the actual weight change calculation (diff w_ij = eta_j * delta_j * o_i)
                    this.differenceWeightMatrix[weightPosition[0]][weightPosition[1]] = this.layers[b].learningRate * this.layers[b].layerNeurons[n].getDelta() * aboveLayer.layerNeurons[aboveLayerNeurons].getOutput();
                    System.out.format(" > Weight change for weight W(%3d>%3d): eta(%3d) * delta(%3d) * o(%3d) = %+3.5f * %+3.5f * %+3.5f = %+3.5f%n",aboveLayer.layerNeurons[aboveLayerNeurons].neuronId, this.layers[b].layerNeurons[n].neuronId, this.layers[b].layerNeurons[n].neuronId, this.layers[b].layerNeurons[n].neuronId, aboveLayer.layerNeurons[aboveLayerNeurons].neuronId, this.layers[b].learningRate, this.layers[b].layerNeurons[n].getDelta(), aboveLayer.layerNeurons[aboveLayerNeurons].getOutput(), this.differenceWeightMatrix[weightPosition[0]][weightPosition[1]]);
                }
            }
        }
    }

    // This is the update step, it adds the calculated weight changes (=difference weights) to the current weights
    public void updateWeights() {
        // Just walk through both matrices (they both have the same dimensions)
        for (int x = 0; x < this.weightMatrix.length; x++) {
            for (int y = 0; y < this.weightMatrix[0].length; y++) {
                // Add the weights from the weight change matrix to the weights in the weight matrix at the same positions
                this.weightMatrix[x][y] += this.differenceWeightMatrix[x][y];
            }
        }
    }

    // Sets the teacher output vector for the mlp
    public void setTeacherOutputVector(double[] teacherOutput) {
        this.teacherOutput = teacherOutput;
    }

    // Calculates and returns the individual error = 1/2 sum (teacher y - net y)^2 for one pattern
    public double getErrorForOnePattern() {
        double error = 0;
        if (this.layerCount > 0) { // Failsafe if e.g. only input layer exists
            double[] mlpOutput = this.getOutput();
            for (int y = 0; y < mlpOutput.length; y++) {
                error +=
                        Math.pow(this.teacherOutput[y] - mlpOutput[y],2); // sum (teacher y-y)^2
            }
        }
        return (error/2); // Error = 1/2 sum (^y-y)^2
    }

    // Returns the output of the mlp in an array of size M
    public double[] getOutput() {
        if (this.layerCount > 0)
            return this.layers[this.layerCount-1].getLayerOutput(); // The last (=output) layer has the id layerCount-1
        else
            return new double[0]; // Failsafe if e.g. only input layer exists
    }

    // Prints the mlp's output
    public void printOutput() {
        // Inits with the size of the output layer (M)
        double[] myNeuralNetOutput = this.getOutput();
        System.out.println("Output:");
        for (int j = 0; j < myNeuralNetOutput.length; j++) {
            System.out.format(" %+3.5f",myNeuralNetOutput[j]);
        }
        System.out.println();
    }

    // Outputs the pattern input
    public void printPatternInput() {
        System.out.println("Input:");
        for (int i = 0; i < this.inputValues.length; i++) {
            System.out.format(" %+3.5f",this.inputValues[i]);
        }
        System.out.println();
    }

    // Outputs the weights in the mlp with neuron numbers
    public void showMlpNeuronWeights() {
        System.out.println("Weights:");
        // Walk through layers
        for (int layer = 0; layer < this.layerCount-1; layer++) {
            layer currentLayer = layers[layer];
            layer belowLayer = layers[layer+1];
            // Walk through neurons
            for (int belowNeuron = 0; belowNeuron < belowLayer.getLayerSize(); belowNeuron++) {
                for (int neuron = -1; neuron < currentLayer.getLayerSize(); neuron++) {
                    if (neuron == -1) // This is for the bias neuron
                        System.out.format(" W(%3d>%3d) = %+3.5f%n", 0, belowLayer.layerNeurons[belowNeuron].neuronId, currentLayer.getWeightFromTo(currentLayer.BIAS, belowLayer.layerNeurons[belowNeuron]));
                    else // for all other neurons in the layer
                        System.out.format(" W(%3d>%3d) = %+3.5f%n", currentLayer.layerNeurons[neuron].neuronId, belowLayer.layerNeurons[belowNeuron].neuronId, currentLayer.getWeightFromTo(currentLayer.layerNeurons[neuron], belowLayer.layerNeurons[belowNeuron]));
                }
            }
        }
    }

    // Outputs the weight changes in the mlp with neuron numbers
    public void showMlpNeuronWeightChanges() {
        System.out.println("Weight Changes:");
        int[] weightPosition;
        // Walk through layers
        for (int layer = 0; layer < this.layerCount-1; layer++) {
            layer currentLayer = layers[layer];
            layer belowLayer = layers[layer+1];
            // Walk through neurons
            for (int belowNeuron = 0; belowNeuron < belowLayer.getLayerSize(); belowNeuron++) {
                for (int neuron = -1; neuron < currentLayer.getLayerSize(); neuron++) {
                    if (neuron == -1) {// This is for the bias neuron
                        weightPosition = currentLayer.getWeightMatrixPositionForNeuronWeightFromTo(currentLayer.BIAS, belowLayer.layerNeurons[belowNeuron]);
                        System.out.format(" Diff W(%3d>%3d) = %+3.5f%n", 0, belowLayer.layerNeurons[belowNeuron].neuronId, this.differenceWeightMatrix[weightPosition[0]][weightPosition[1]]);
                    } else {// for all other neurons in the layer
                        weightPosition = currentLayer.getWeightMatrixPositionForNeuronWeightFromTo(currentLayer.layerNeurons[neuron], belowLayer.layerNeurons[belowNeuron]);
                        System.out.format(" Diff W(%3d>%3d) = %+3.5f%n", currentLayer.layerNeurons[neuron].neuronId, belowLayer.layerNeurons[belowNeuron].neuronId, this.differenceWeightMatrix[weightPosition[0]][weightPosition[1]]);
                    }
                }
            }
        }
    }

    // Outputs the weights in the mlp
    public void printWeightMatrix(int X, int Y) {
        System.out.println("WeightMatrix:");
        for (int i = 0; i < X; i++) {
            for (int j = 0; j < Y; j++) {
                System.out.format(" %+3.5f",this.weightMatrix[i][j]);
            }
            System.out.println();
        }
    }

    // Outputs the weight differences calculated by backprop
    public void printDifferencesWeightMatrix(int X, int Y) {
        System.out.println("Differences WeightMatrix:");
        for (int i = 0; i < X; i++) {
            for (int j = 0; j < Y; j++) {
                System.out.format(" %+3.5f",this.differenceWeightMatrix[i][j]);
            }
            System.out.println();
        }
    }

} // end of class mlp


// This is the surrounding class that manages file I/O, the setup of weights & patterns and calls the mlp's functions
class main {

    // The patterns from the text file are stored in here, each line corresponds to 1 pattern
    private static int[] teacherPatternsOrder; // This contains the order in which the patterns are presented to the mlp
    private static double[][] teacherInputPatterns;
    private static double[][] teacherOutputPatterns;    
    private static double[][] weightMatrix; // This matrix stores the weights, row i=from neuron i, column j=to neuron j
    private static int N = 4; // Number of input neurons, can also be set explicitly when adding the layer
    private static int H = 2; // Number of hidden neurons, can also be set explicitly when adding the layer
    private static int M = 1; // Number of output neurons, can also be set explicitly when adding the layer
    private static int P; // Number of patterns
    private static int WX; // width of weightmatrix
    private static int WY; // height of weightmatrix
    private static java.util.Random prng; // The pseudo-random number generator (PRNG) object
    private static double[] netProgress; // This contains the accumulated mlp error per epoche
    private static String separator = ","; // Separator for the read files (e.g. "train.dat"), e.g. "," for CSV
    private static final long seed = 3015; // Seed for the pseudo-random number generator
    private static final int epocheCount = 1000; // Number of epoches the net calculates over, this is also the break condition for the mlp

    // Main program entry point
    public static void main(String[] args) {

        // This file contains the net topology as well as the weights, but both can be set in this main function alternatively
        // Uncomment the following line in order to read out the net topology and weights from the below file
//        if (!readValuesFromFile("perzeptronWeights.dat", true)) System.err.println(" > Error reading weights file");

        WX = N + H + 2; // rows=N+H+2 (+1 BIAS for each layer)
        WY = Math.max(H,M); // The biggest layer of the two determines the count of columns

        // Read in the training patterns, assume the dimensions of the patterns match those in the perceptron.net or the specified ones
        if (!readValuesFromFile("train.dat", false)) System.err.println(" > Error reading training file!");

        // Call and initialize the PRNG with seed
        prng = new java.util.Random();
        prng.setSeed(seed);

        // Init the pattern feed order once with some value (default identity order is as good as any)
        teacherPatternsOrder = new int[P];
        for (int i = 0; i < P; i++) {
            teacherPatternsOrder[i] = i;
        }

        // Set the weights randomly to values in the range -0.5 .. +0.5
        weightMatrix = generateRandomWeightMatrix(WX,WY,-0.5,0.5);

        // Generate a new instance of the neural net
        mlp myNeuralNet = new mlp(weightMatrix); // Init with weights

        // Add input layer of size N (N can also be set to an int value at this point) with default (=identity) transfer function and default (=no) learning rate
        myNeuralNet.addLayer(N);
        // Add a hidden layer of size H with transferfunction tanh and learning rate eta = 0.5
        // Available transferfunctions:
        //  identity: mlp.Transferfunction.id
        //  tanh: mlp.Transferfunction.tanh
        //  fermi: mlp.Transferfunction.fermi (also called 'sigmoid function')
        myNeuralNet.addLayer(H,mlp.Transferfunction.tanh, 0.5);
        // Add an output layer of size M with fermi transferfunction and learning rate 0.2
        myNeuralNet.addLayer(M,mlp.Transferfunction.fermi, 0.2);
        // Init the learning progress variable
        netProgress = new double[epocheCount];
        // Each call in this loop iterates over all patterns (=1 epoche)
        for (int epoche = 0; epoche < epocheCount; epoche++) {
            netProgress[epoche] = 0; // The initial error of this epoche is 0
            // Shuffle the patterns so that the net does not memorize the order of the patterns (to avoid overfitting)
            shuffle(teacherPatternsOrder);
            // Walk through teacher patterns
            for (int pattern = 0; pattern < P; pattern++) {
                System.out.println();
                System.out.format(" > Feeding net with input pattern number %3d%n", teacherPatternsOrder[pattern]);
                // Set the current pattern teacher output
                myNeuralNet.setTeacherOutputVector(teacherOutputPatterns[teacherPatternsOrder[pattern]]);
                // Feed the current pattern as input
                myNeuralNet.propagateInput(teacherInputPatterns[teacherPatternsOrder[pattern]]);
//                myNeuralNet.printPatternInput(); // Output the input pattern
//                myNeuralNet.printOutput(); // Show the mlp output
                System.out.format(" > Overall pattern net output error: %+3.5f%n", myNeuralNet.getErrorForOnePattern());
                netProgress[epoche] += myNeuralNet.getErrorForOnePattern(); // F = sum E(p), Overall Error F is the sum over all individual errors of pattern p
                myNeuralNet.backpropagationOfDelta(); // Call the backpropagation of Delta learning algorithm for the currently fed pattern
//                myNeuralNet.showMlpNeuronWeights(); // Output the weights of the mlp with neuron ids in a human readable way
                myNeuralNet.updateWeights(); // The is the update step, here the weight changes returned by backpropagation of delta are applied
//                myNeuralNet.showMlpNeuronWeightChanges(); // Output the weight changes calculated by the backprop learning algorithm
//                myNeuralNet.showMlpNeuronWeights(); // Output the updated weights
            }
            System.out.format(" > Overall accumulated epoche error: %+3.5f%n", netProgress[epoche]); // Outputs F, the overall accumulated epoche error
        }

		// Write out net progress (the overall epoche error) for use with gnuplot
        writeGnuPlotFile("epocheerror.dat");
//        myNeuralNet.showMlpNeuronWeights(); // Output the updated weights
        // Save the weights to a file
//        writeValuesToFile("perzeptronOut.net");
    }

    // Initializes the weights randomly in a given range, e.g. -0.5,0.5
    // It has the form
    // First row: Bias weights to hidden layer
    // Following N rows: weights from layer N to layer H, w_n_h with row=n, column=h
    // Following (N+1)th row: Bias weights to output layer
    // Following H rows: weights from layer H to layer M, w_h_m with row=h, column=m
    // You can check the calculated weights with showMlpNeuronWeights()
    public static double[][] generateRandomWeightMatrix(int WX, int WY, double min, double max) {
        double range = max-min; // First get the range
        double[][] randomWeightMatrix = new double[WX][WY];
        for (int i = 0; i < WX; i++) {
            for (int j = 0; j < WY; j++) {
                // nextDouble() returns numbers in [0..1), to scale it correctly do the following
                randomWeightMatrix[i][j] = (prng.nextDouble()*range)+min;
            }
        }
        return randomWeightMatrix;
    }

    // Read in the values from file, weightMatrixMode = for weightmatrix, else e.g. for patterns
    public static boolean readValuesFromFile(String FileName, boolean weightMatrixMode) {
        File inputTextFile = new File(FileName);
        if (inputTextFile.exists()) {
            System.out.println("> Reading " + FileName);
            FileInputStream fis = null;
            BufferedInputStream bis = null;
//            DataInputStream dis = null;
            BufferedReader dis = null;
            try {
                fis = new FileInputStream(inputTextFile);
                bis = new BufferedInputStream(fis);
//                dis = new DataInputStream(bis);
                dis = new BufferedReader(new InputStreamReader(bis));

                if (weightMatrixMode) {
                    String Lines = dis.readLine();
                    // Omit commented lines
                    while (Lines.startsWith("#")) { // This is a comment line, ignore it
                        Lines = dis.readLine();
                    }
                    // In the first line, N and M are defined
                    String[] layerDefs = Lines.split("-");
                    // Set dimensions of weightmatrix according to N (+1 b/c BIAS) and M
                    N = Integer.parseInt(layerDefs[0].trim());
                    H = Integer.parseInt(layerDefs[1].trim());
                    M = Integer.parseInt(layerDefs[2].trim());
                    WX = N + H + 2; // rows=N+H+2 (+1 BIAS for each layer)
                    WY = Math.max(H,M); // The biggest layer of the two determines the count of columns
                    weightMatrix = new double[WX][WY];
                    System.out.println(" (" + (N) + "-" + H + "-" + M + ")");
                } else {
                    // # Patterns <= 1000, first teacher input (#N) + then teacher output of pattern (#M)
                    teacherInputPatterns = new double[1000][N];
                    teacherOutputPatterns = new double[1000][M];
                }

                // Iterate over lines (i) in the file
                int i = 0;
                String line;
                String singleValue;
                double readValue;
                // Filter out irrelevant and unwanted cases (e.g. comments and lines with only a line break in it)
                while (((line = dis.readLine()) != null) && ((!weightMatrixMode) || (i < WX))) { // if there is stuff (e.g. line breaks) after the specified dimensions, break
                    if (!line.startsWith("#")) {
                        String[] singleValues = line.split(separator);
                        // Iterate over single values (j)
                        for (int j = 0; j < singleValues.length; j++) {
                            singleValue = singleValues[j].trim(); // Cut out blanks
                            // Catch if a weight is not set, set it to 0 (it won't be referenced anyway, just in case s.t. goes wrong)
                            if (singleValue.isEmpty()) {
                                readValue = (double)0;
                            } else {
                                // Convert to double
                                readValue = Double.valueOf(singleValue).doubleValue();
                            }
                            System.out.format(" %3d x %3d: %+3.5f", i, j, readValue);
                            if (weightMatrixMode) {
                                weightMatrix[i][j] = readValue;
                                System.out.println();
                            } else {
                                if (j < N) { // The first N values are teacher input
                                    System.out.println(" in");
                                    teacherInputPatterns[i][j] = readValue;
                                } else { // The last M values are teacher output
                                    System.out.println(" out");
                                    teacherOutputPatterns[i][j-N] = readValue;
                                }
                            }
                        }
                        i++;
                    }
                }
                if (!weightMatrixMode) {
                    P = i;
                }
                fis.close();
                bis.close();
                dis.close();
            } catch (FileNotFoundException e) {
                e.printStackTrace();
                return false;
            } catch (IOException e) {
                e.printStackTrace();
                return false;
            }
        }
        return true; // all ok
    }

    // This writes the overall error data per epoche to a specified output file for use with gnuplot, e.g.
    // gnuplot> set log y; unset log x; set xlabel "Epoches"; set ylabel "ln(Error)"; plot "lernkurve.dat" using 1:2 title 'Overall Epoche Error' with lines
    public static boolean writeGnuPlotFile(String FileName) {
        // Create file
        try {
            FileWriter fstream = new FileWriter(FileName);
            BufferedWriter out = new BufferedWriter(fstream);
            out.write("# Usage example:\n# gnuplot> set xlabel \"Epoches\"; set ylabel \"Error\"; plot \""+FileName+"\" using 1:2 title 'Overall Epoche Error' \n");
            for (int i = 0; i < epocheCount; i++) {
                out.write(String.valueOf(i) + "\t" + String.valueOf(netProgress[i]) + "\n");
            }
            out.close();
        } catch (Exception e) {
            e.printStackTrace();
            return false;
        }
        return true;
    }

    // Writes the weight matrix to a file using the given separator
    public static boolean writeValuesToFile(String FileName) {
        System.out.println(" > Writing to file " + FileName);
        // Create file
        try {
            FileWriter fstream = new FileWriter(FileName);
            BufferedWriter out = new BufferedWriter(fstream);
            out.write("(" + N + " - " + H + "-" + M + ")\n");
            for (int i = 0; i < WX; i++) {
                for (int j = 0; j < WY; j++) {
                    double writeValue = weightMatrix[i][j];
                    out.write(String.valueOf(writeValue));
                    if (j != WX - 1) {
                        out.write(separator);
                    }
                }
                out.write("\n");
            }
            out.close();
        } catch (Exception e) {
            e.printStackTrace();
            return false;
        }
        return true;
    }

    // Simple modified Fisher-Yates shuffle for use with teacher patterns
    public static void shuffle(int[] array) {
        int n = array.length;            // The number of items left to shuffle (loop invariant).
        while (n > 1) {
            n--;                         // n is now the last pertinent index
            int k = prng.nextInt(n + 1);  // 0 <= k <= n.
            // Simple swap of variables
            int tmp = array[k];
            array[k] = array[n];
            array[n] = tmp;
        }
    }


} // end of class main
 
