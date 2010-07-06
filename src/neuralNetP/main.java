/**
 *
 * @author Jan Hendriks
 * Compiled and tested with a recent Sun Microsystems JDK: Java(TM) SE Runtime Environment (build 1.6.0_16-b01) and JRE 6u17
 * All function and variable names should be self-explanatory
 */

package neuralNetP;

// Imports
import java.io.*;
import java.util.Random.*;

// This is the surrounding class that manages file I/O, the setup of weights & patterns and calls the mlp's functions
class main {

    // The patterns from the text file are stored in here, each line corresponds to 1 pattern
    private static int[] teacherPatternsOrder; // This contains the order in which the patterns are presented to the mlp
    private static double[][] teacherInputPatterns;
    private static double[][] teacherOutputPatterns;    
    private static double[][] weightMatrix; // This matrix stores the weights, row i=from neuron i, column j=to neuron j
    private static int N = 4; // Number of input neurons, can also be set explicitly when adding the layer
    private static int H = 12; // Number of hidden neurons, can also be set explicitly when adding the layer
    private static int M = 3; // Number of output neurons, can also be set explicitly when adding the layer
    private static int P; // Number of patterns
    private static int WX; // width of weightmatrix
    private static int WY; // height of weightmatrix
    private static java.util.Random prng; // The pseudo-random number generator (PRNG) object
    private static double[] netProgress; // This contains the accumulated mlp error per epoche
    private static String separator = " "; // Separator for the read files, e.g. "," for CSV
    private static final long seed = 3015; // Seed for the pseudo-random number generator
    private static final int epocheCount = 10000; // Number of epoches the net calculates over, this is also the break condition for the mlp

    // Main program entry point
    public static void main(String[] args) {

        // This file contains the net topology as well as the weights, but both can be set in this main function alternatively
        // Uncomment the following line in order to read out the net topology and weights from the below file
//        if (!readValuesFromFile("perzeptron.net", true)) System.err.println(" > Fehler beim Einlesen der Datei");

        WX = N + H + 2; // rows=N+H+2 (+1 BIAS for each layer)
        WY = Math.max(H,M); // The biggest layer of the two determines the count of columns

        // Read in the training patterns, assume the dimensions of the patterns match those in the perceptron.net or the specified ones
        if (!readValuesFromFile("add.pat", false)) System.err.println(" > Fehler beim Einlesen der Datei");

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
        // Add a hidden layer of size H with transferfunction fermi and learning rate eta = 0.5
        // Available transferfunctions:
        // identity: neuralNetP.mlp.Transferfunction.id
        // tanh: neuralNetP.mlp.Transferfunction.tanh
        // fermi: neuralNetP.mlp.Transferfunction.fermi
        myNeuralNet.addLayer(H,neuralNetP.mlp.Transferfunction.tanh, 0.2);
        // Add an output layer of size M with fermi transferfunction and learning rate 0.5
        myNeuralNet.addLayer(M,neuralNetP.mlp.Transferfunction.fermi, 0.5);
        // Init the learning progress variable
        netProgress = new double[epocheCount];
        // Each call in this loop iterates over all patterns (=1 epoche)
        for (int epoche = 0; epoche < epocheCount; epoche++) {
            System.out.format(" > Entering epoche %3d of %3d %n", epoche, epocheCount);
            netProgress[epoche] = 0; // The initial error of this epoche is 0
            // Shuffle the patterns so that the net does not memorize the order of the patterns
            shuffle(teacherPatternsOrder);
            // Walk through teacher patterns
            for (int pattern = 0; pattern < P; pattern++) {
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
                myNeuralNet.updateWeights(); // The update step, here the weight changes returned by backpropagation of delta are applied
//                myNeuralNet.showMlpNeuronWeightChanges(); // Output the weight changes calculated by the backprop learning algorithm
//                myNeuralNet.showMlpNeuronWeights(); // Output the updated weights
            }
            System.out.format(" > Overall accumulated epoche error: %+3.5f%n", netProgress[epoche]); // Outputs F, the overall accumulated epoche error
            System.out.println();
        }

        myNeuralNet.showMlpNeuronWeights(); // Output the neuron weights

        writeGnuPlotFile("lernkurve.dat");
        // Save the weights to a file
//        writeValuesToFile("perzeptronWeights.net");
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
            out.write("# Usage example:\n# gnuplot> set log y; unset log x; set xlabel \"Epoches\"; set ylabel \"ln(Error)\"; plot \""+FileName+"\" using 1:2 title 'Overall Epoche Error' with lines \n");
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
