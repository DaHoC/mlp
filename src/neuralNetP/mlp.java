/**
 *
 * @author Jan Hendriks
 */

package neuralNetP;

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
//                        System.out.format(" 1      * W(  0>%3d) = %+3.5f * %+3.5f%n", this.neuronId, (double)1, weightedSumAccumulator);
                        for (int i = 0; i < aboveLayerNeurons.length; i++) {
                            // This is the weight from one above neuron to the current neuron object
                            double weightToCurrentNeuronFromAboveNeuron = getWeightFromTo(aboveLayerNeurons[i],this);
                            // This is the output of one neuron in the above layer of the current layer
                            double outputOfNeuronInAboveLayer = aboveLayerNeurons[i].getOutput();
//                            System.out.format(" O(%3d) * W(%3d>%3d) = %+3.5f * %+3.5f%n", aboveLayerNeurons[i].neuronId , aboveLayerNeurons[i].neuronId, this.neuronId, weightToCurrentNeuronFromAboveNeuron, outputOfNeuronInAboveLayer);
                            // This calculates one summand (w_ij * o_i) for the weighted sum and adds it to the previous summands
                            weightedSumAccumulator += weightToCurrentNeuronFromAboveNeuron * outputOfNeuronInAboveLayer;
                        }
                        // After walking through all neurons in the above layer (and the bias) and adding up the summands, we have the weighted sum
                        this.weightedSum = weightedSumAccumulator;
                    }
//                    System.out.format("Calculated net for neuron %6d: %+3.5f%n", this.neuronId, this.weightedSum);
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
//                System.out.format("Calculated output for neuron %3d: %+3.5f%n", this.neuronId, this.output);
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
//                        System.out.format(" delta += W(%3d>%3d) * delta (%3d) = %+3.5f * %+3.5f%n",this.neuronId, belowLayer.layerNeurons[i].neuronId, belowLayer.layerNeurons[i].neuronId, getWeightFromTo(this, belowLayer.layerNeurons[i]), belowLayer.layerNeurons[i].getDelta());
                    }
                    this.delta = deltaAccumulator * this.getDerivationAt(this.getWeightedSum());
                }

//                System.out.format(" Delta %+3.5f at layer %d%n", this.delta, this.getLayerId());
//                System.out.format(" f'(%+3.5f) = %+3.5f%n", this.getWeightedSum(), this.getDerivationAt(this.getWeightedSum()));
//                System.out.format("Delta of neuron %3d in layer %3d: %+3.5f%n", this.neuronId, layerId, this.delta);
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

/**
 * @TODO: Enhance: According to the profiler, the mlp spends too much time in this rather unimportant routine,
 *  suggest using (&building at runtime) a caching data structure or lookup table instead of calculating the position each time
 * @param fromNeuron
 * @param toNeuron
 * @return
 */
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

/**
 * @TODO: alternative to BP-algorithm below: 
 *  Genetic algorithm (aka. evolutionary algorithm) using random weights for use with e.g. heaviside function (BP useless with this transferfunction because of inconsistent / useless derivation):
 *  evaluate the best weights by calculating (and continue mating) the weights that returned the lowest error
 */

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
//                System.out.format(" > Weight change for weight W(%3d>%3d): eta(%3d) * delta(%3d) * o(%3d) = %+3.5f * %+3.5f * %+3.5f = %+3.5f%n",0, this.layers[b].layerNeurons[n].neuronId, this.layers[b].layerNeurons[n].neuronId, this.layers[b].layerNeurons[n].neuronId, 0, this.layers[b].learningRate, this.layers[b].layerNeurons[n].getDelta(), (double)1, this.differenceWeightMatrix[weightPosition[0]][weightPosition[1]]);

                // Consider all other neurons in the above layer
                for (int aboveLayerNeurons = 0; aboveLayerNeurons < aboveLayer.getLayerSize(); aboveLayerNeurons++) {
                    // This only converts the w_ij to the corresponding weight entry in the weight matrix (row,column)=(weightPosition[0]][weightPosition[1])
                    weightPosition = aboveLayer.getWeightMatrixPositionForNeuronWeightFromTo(aboveLayer.layerNeurons[aboveLayerNeurons],this.layers[b].layerNeurons[n]);
                    // This is the actual weight change calculation (diff w_ij = eta_j * delta_j * o_i)
                    this.differenceWeightMatrix[weightPosition[0]][weightPosition[1]] = this.layers[b].learningRate * this.layers[b].layerNeurons[n].getDelta() * aboveLayer.layerNeurons[aboveLayerNeurons].getOutput();
//                    System.out.format(" > Weight change for weight W(%3d>%3d): eta(%3d) * delta(%3d) * o(%3d) = %+3.5f * %+3.5f * %+3.5f = %+3.5f%n",aboveLayer.layerNeurons[aboveLayerNeurons].neuronId, this.layers[b].layerNeurons[n].neuronId, this.layers[b].layerNeurons[n].neuronId, this.layers[b].layerNeurons[n].neuronId, aboveLayer.layerNeurons[aboveLayerNeurons].neuronId, this.layers[b].learningRate, this.layers[b].layerNeurons[n].getDelta(), aboveLayer.layerNeurons[aboveLayerNeurons].getOutput(), this.differenceWeightMatrix[weightPosition[0]][weightPosition[1]]);
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
