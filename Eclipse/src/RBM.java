import java.io.BufferedReader;
import java.io.BufferedWriter;
import java.io.FileInputStream;
import java.io.FileWriter;
import java.io.IOException;
import java.io.InputStream;
import java.io.InputStreamReader;
import java.nio.charset.Charset;
import java.text.SimpleDateFormat;
import java.util.Date;
import java.util.Random;

public class RBM {

	public double LRATE;
	public double T;

	public double LRATE_INITIAL = 0.02;
	public double T_INITIAL = 1;
	public double K = 0.001;

	public double[][][] weights;
	public double[][] ratingBias;
	public double[] hiddenBias;
	public int[] userStart;

	public int NUM_HIDDEN_NODES = 10;
	public int TEST_PARAM;

	public byte[] hiddenStates = new byte[NUM_HIDDEN_NODES];

	public Random rand = new Random(0);

	public static final int NUM_EPOCHS_SPAN = 5;
	public static final double MIN_ERROR_DIFF = 0.00001;

	public double GLOBAL_MEAN = 3.6033;
	public static final double CINEMATCH_BASELINE = 0.9514;

	public static final int NUM_USERS = 458293;
	public static final int NUM_MOVIES = 17770;
	public static final int NUM_DATES = 2243;
	public static final int NUM_POINTS = 102416306;
	public static final int NUM_1_POINTS = 94362233;
	public static final int NUM_2_POINTS = 1965045;
	public static final int NUM_3_POINTS = 1964391;
	public static final int NUM_4_POINTS = 1374739;
	public static final int NUM_5_POINTS = 2749898;
	public static final int NUM_TRAINING_POINTS = NUM_1_POINTS + NUM_2_POINTS
			+ NUM_3_POINTS;
	public static final int NUM_TRAINING_PROBE_POINTS = NUM_1_POINTS
			+ NUM_2_POINTS + NUM_3_POINTS + NUM_4_POINTS;

	public static final String INPUT_DATA = "all.dta";
	public static final String INPUT_INDEX = "all.idx";
	public static final String INPUT_QUAL = "qual.dta";
	public static final String LOGFILE = "log.txt";

	// Data that is stored in memory
	public int[] userEnd = new int[NUM_USERS];
	public int[] users = new int[NUM_TRAINING_PROBE_POINTS];
	public short[] movies = new short[NUM_TRAINING_PROBE_POINTS];
	public short[] dates = new short[NUM_TRAINING_PROBE_POINTS];
	public byte[] ratings = new byte[NUM_TRAINING_PROBE_POINTS];
	public int[][] probeData = new int[NUM_4_POINTS][4];
	public int[][] qualData = new int[NUM_5_POINTS][3];

	public RBM(int NUM_HIDDEN_NODES) {
		this.NUM_HIDDEN_NODES = NUM_HIDDEN_NODES;

		// Initialize things that are specific to the training session.
		initializeVars();
	}

	public RBM(int NUM_HIDDEN_NODES, double testParam) {
		this.NUM_HIDDEN_NODES = NUM_HIDDEN_NODES;
		this.K = testParam;

		// Initialize things that are specific to the training session.
		initializeVars();
	}

	private void initializeVars() {
		// Hidden / movie bias will be the last node in the indices
		weights = new double[NUM_HIDDEN_NODES][NUM_MOVIES][5];
		ratingBias = new double[NUM_MOVIES][5];
		hiddenBias = new double[NUM_HIDDEN_NODES];
		userStart = new int[NUM_USERS];

		LRATE = LRATE_INITIAL;
		T = T_INITIAL;

		// Initialize weights.
		for (int i = 0; i < NUM_HIDDEN_NODES; i++) {
			for (int j = 0; j < NUM_MOVIES; j++) {
				for (int k = 0; k < 5; k++) {
					weights[i][j][k] = rand.nextGaussian() / 100;
				}
			}
		}
	}

	private void setVarsToNull() {
		weights = null;
		ratingBias = null;
		hiddenBias = null;
		userStart = null;
	}

	public void train() throws NumberFormatException, IOException {
		System.out.println(timestampLine(String.format("Training %d nodes.",
				NUM_HIDDEN_NODES)));

		// Read in input
		readInput();

		// Set up logfile.
		BufferedWriter logWriter = new BufferedWriter(new FileWriter(LOGFILE, true));
		logWriter.write("\n");

		// TRAIN WITH TRAINING SET ONLY (no probe)
		precompute(NUM_TRAINING_POINTS);
		double previousRmse = calcProbeRmse();
		logRmse(logWriter, previousRmse, 0);
		int numEpochsToTrain = 0;
		for (int i = 1; true; i++) {
			double rmse = trainWithNumPoints(NUM_TRAINING_POINTS);
			logRmse(logWriter, rmse, i);

			// Slow down learning rate as we're getting close to the answer.
			LRATE *= .9;
			if (i == 17)
				T = 3;
			else if (i == 27)
				T = 5;
			else if (i == 40)
				T = 9;

			// If probe error has been going up, we should stop.
			double rmseDiff = previousRmse - rmse;
			if (rmseDiff < MIN_ERROR_DIFF) {
				System.out
						.println(timestampLine("Probe error has started"
								+ " to go up significantly; memorizing number of epochs to train."));
				generateProbeOutput();
				numEpochsToTrain = i;
				break;
			}

			previousRmse = rmse;
		}

		// TRAIN WITH PROBE.
		setVarsToNull();
		initializeVars();
		precompute(NUM_TRAINING_PROBE_POINTS);
		logEpoch(logWriter, 0);
		for (int i = 1; i <= numEpochsToTrain + NUM_EPOCHS_SPAN; i++) {
			// Train with training set AND probe.
			trainWithNumPoints(NUM_TRAINING_PROBE_POINTS);
			logEpoch(logWriter, i);

			// Slow down learning rate as we're getting close to the answer.
			LRATE *= .9;
			if (i == 17)
				T = 3;
			else if (i == 27)
				T = 5;
			else if (i == 40)
				T = 9;

			if (i == numEpochsToTrain + NUM_EPOCHS_SPAN) {
				generateOutput();
			}
		}

		logWriter.close();
		System.out.println("Done!");
	}

	public double trainWithNumPoints(int numPoints) throws IOException {
		int user, user_start, user_end, user_num_points;
		double[] numer = new double[5];

		int i = 0;
		while (i < numPoints) {
			user = users[i];
			user_start = i;
			user_end = userEnd[user];
			user_num_points = user_end - user_start + 1;

			System.out.println(user);

			// Compute states of hidden units: for each hidden node,
			// compute the activation energy and then randomly activate.
			for (int k = 0; k < NUM_HIDDEN_NODES; k++) {
				if (rand.nextDouble() < logistic(activationEnergyHidden(k, user_start,
						user_end))) {
					hiddenStates[k] = 1;
				} else {
					hiddenStates[k] = 0;
				}
			}
			// Compute positive(e_ij) for all edges
			for (int l = 0; l < NUM_HIDDEN_NODES; l++) {
				for (int m = 0; m < NUM_MOVIES; m++) {
					// Calculate numerator / denominator for multinomial distribution.
					double denom = 0;
					for (int n = 0; n < 5; n++) {
						double edgeSum = 0;
						double term = 0;
						for (int o = 0; o < NUM_HIDDEN_NODES; o++) {
							edgeSum += hiddenStates[o] * weights[o][m][n];
						}
						term = Math.exp(ratingBias[m][n] + edgeSum);
						denom += term;
						numer[n] = term;
					}

					// Calculate error for all 5 rating nodes
					for (int n = 0; n < 5; n++) {
						// For all rated movies for this user, positive is
						// hiddenStates, otherwise it's 0.
						double positive = 0;
						if (userMovieToRatingContainsKey(m, user_start, user_end)) {
							positive = hiddenStates[l] * 1;
						}

						// Calculate negative by flipping visibleStates
						double negative = 0;
						for (int k = 0; k < NUM_MOVIES; k++) {
							double activationEnergyVisible = numer[n] / denom;
							if (rand.nextDouble() < activationEnergyVisible) {
								negative = hiddenStates[l];
							} else {
								negative = 0;
							}
						}

						// Update weights.
						weights[l][m][n] += LRATE
								* ((positive - negative) - K * weights[l][m][n]);
					}
				}
			}

			// // Update weights after every 1000 users.
			// if (user % 1000 == 999 || user == NUM_USERS - 1) {
			// }

			i += user_num_points;
		}

		// Test the model in probe set.
		return calcProbeRmse();
	}

	private boolean userMovieToRatingContainsKey(int movie, int userStart,
			int userEnd) {
		int m;
		for (int i = userStart; i <= userEnd; i++) {
			m = movies[i];
			if (m == movie)
				return true;
		}
		return false;
	}

	private double activationEnergyHidden(int hiddenNode, int userStart,
			int userEnd) {
		double activation = hiddenBias[hiddenNode];
		int movie;
		int rating;
		for (int i = userStart; i <= userEnd; i++) {
			movie = movies[i];
			rating = ratings[i];
			activation += weights[hiddenNode][movie][rating - 1];
		}
		return activation;
	}

	public double logistic(double x) {
		return 1 / (1 + (Math.exp(-1 * x)));
	}

	public double predictRating(int movie, int user, int date) {
		// Set hidden states
		int user_start = userStart[user];
		int user_end = userEnd[user];
		double[] tempProbs = new double[NUM_HIDDEN_NODES];
		for (int k = 0; k < NUM_HIDDEN_NODES; k++) {
			tempProbs[k] = logistic(activationEnergyHidden(k, user_start, user_end));
		}
		// Calculate multinomial probs
		double[] numer = new double[5];
		// Calculate numerator / denominator for multinomial distribution.
		double denom = 0;
		for (int n = 0; n < 5; n++) {
			double edgeSum = 0;
			double term = 0;
			for (int o = 0; o < NUM_HIDDEN_NODES; o++) {
				edgeSum += tempProbs[o] * weights[o][movie][n];
			}
			term = Math.exp(ratingBias[movie][n] + edgeSum);
			denom += term;
			numer[n] = term;
		}
		// Normalize probabilities
		double probNorm = 0;
		double[] probs = new double[5];
		for (int i = 0; i < 5; i++) {
			probNorm += probs[i];
		}
		for (int i = 0; i < 5; i++) {
			probs[i] /= probNorm;
		}
		double ratingSum = 0;
		for (int i = 0; i < 5; i++) {
			ratingSum += probs[i] * (i + 1);
		}
		return ratingSum;
	}

	public void precompute(int numPoints) throws IOException {
		// If we are precomputing with probe, we need to re-read the data in the
		// correct order.
		if (numPoints == NUM_TRAINING_PROBE_POINTS) {
			// Read input into memory
			InputStream fis = new FileInputStream(INPUT_DATA);
			BufferedReader br = new BufferedReader(new InputStreamReader(fis,
					Charset.forName("UTF-8")));
			InputStream fisIdx = new FileInputStream(INPUT_INDEX);
			BufferedReader brIdx = new BufferedReader(new InputStreamReader(fisIdx,
					Charset.forName("UTF-8")));

			// Read INPUT_INDEX
			System.out.println(timestampLine("Loading data index..."));
			byte[] dataIndices = new byte[NUM_POINTS];
			String line;
			byte index;
			int lineNum = 0;
			while ((line = brIdx.readLine()) != null) {
				index = Byte.parseByte(line);
				dataIndices[lineNum] = index;
				lineNum++;
			}

			// Read INPUT_DATA
			System.out.println(timestampLine("Loading data..."));
			String[] parts;
			int user;
			short movie, date;
			byte rating;
			lineNum = 0;
			int trainingDataIndex = 0;
			while ((line = br.readLine()) != null) {
				parts = line.split(" ");
				user = Integer.parseInt(parts[0]) - 1;
				movie = (short) (Short.parseShort(parts[1]) - 1);
				date = (short) (Short.parseShort(parts[2]) - 1);
				rating = (byte) (Byte.parseByte(parts[3]));
				if (dataIndices[lineNum] == 1 || dataIndices[lineNum] == 2
						|| dataIndices[lineNum] == 3 || dataIndices[lineNum] == 4) {

					users[trainingDataIndex] = user;
					movies[trainingDataIndex] = movie;
					dates[trainingDataIndex] = date;
					ratings[trainingDataIndex] = rating;
					trainingDataIndex++;
				}
				lineNum++;
				if (lineNum % 10000000 == 0) {
					System.out.println(timestampLine(lineNum + " / " + NUM_POINTS));
				}
			}
		}

		// Compute mean.
		long ratingSum = 0;
		for (int i = 0; i < numPoints; i++) {
			ratingSum += ratings[i];
		}
		GLOBAL_MEAN = ((double) ratingSum) / numPoints;

		// Compute user end.
		int prevUser = -1;
		int user = 0;
		// Index the beginning of data for each user
		for (int i = 0; i < numPoints; i++) {
			user = users[i];
			if (user != prevUser && prevUser != -1) {
				userEnd[prevUser] = i;
			}
			prevUser = user;
		}
		userEnd[prevUser] = numPoints - 1;

		// Compute userMovieToRatings
		prevUser = -1;
		for (int i = 0; i < numPoints; i++) {
			user = users[i];
			if (user != prevUser && prevUser != -1) {
				userStart[user] = i;
			}
			prevUser = user;
		}
	}

	public double outputRating(int movie, int user, int date) {
		double rating = predictRating(movie, user, date);
		rating = addAndClip(rating, 0);
		return rating;
	}

	public String timestampLine(String logline) {
		String currentDate = new SimpleDateFormat("h:mm:ss a").format(new Date());
		return currentDate + ": " + logline;
	}

	private double addAndClip(double n, double addThis) {
		n += addThis;
		if (n > 5) {
			return 5;
		} else if (n < 1) {
			return 1;
		}
		return n;
	}

	private double calcProbeRmse() throws IOException {
		int user;
		short movie, date;
		byte rating;

		// Test the model in probe set.
		double rmse = 0;
		for (int j = 0; j < probeData.length; j++) {
			user = probeData[j][0];
			movie = (short) probeData[j][1];
			date = (short) probeData[j][2];
			rating = (byte) probeData[j][3];

			if (j % 10000 == 0)
				System.out.println(j);
			rmse += Math.pow(rating - predictRating(movie, user, date), 2);
		}
		rmse = Math.sqrt(rmse / NUM_4_POINTS);

		return rmse;
	}

	private void logRmse(BufferedWriter logWriter, double rmse, int i)
			throws IOException {
		// Print + log some stats.
		double predictedPercent = (1 - rmse / CINEMATCH_BASELINE) * 100;
		String currentDate = new SimpleDateFormat("h:mm:ss a").format(new Date());
		String logline = currentDate
				+ String.format(": epoch %d probe RMSE %.5f (%.2f%%) ", i, rmse,
						predictedPercent);
		System.out.println(logline);
		logWriter.write(logline + "\n");
	}

	private void logEpoch(BufferedWriter logWriter, int i) throws IOException {
		// Print + log some stats.
		String currentDate = new SimpleDateFormat("h:mm:ss a").format(new Date());
		String logline = currentDate + String.format(": epoch %d", i);
		System.out.println(logline);
		logWriter.write(logline + "\n");
	}

	private void readInput() throws NumberFormatException, IOException {
		// Read input into memory
		InputStream fis = new FileInputStream(INPUT_DATA);
		BufferedReader br = new BufferedReader(new InputStreamReader(fis,
				Charset.forName("UTF-8")));
		InputStream fisIdx = new FileInputStream(INPUT_INDEX);
		BufferedReader brIdx = new BufferedReader(new InputStreamReader(fisIdx,
				Charset.forName("UTF-8")));

		// Read INPUT_INDEX
		System.out.println(timestampLine("Loading data index..."));
		byte[] dataIndices = new byte[NUM_POINTS];
		String line;
		byte index;
		int lineNum = 0;
		while ((line = brIdx.readLine()) != null) {
			index = Byte.parseByte(line);
			dataIndices[lineNum] = index;
			lineNum++;
		}

		// Read INPUT_DATA
		System.out.println(timestampLine("Loading data..."));
		String[] parts;
		int user;
		short movie, date;
		byte rating;
		lineNum = 0;
		int trainingDataIndex = 0, probeDataIndex = 0, qualDataIndex = 0;
		while ((line = br.readLine()) != null) {
			parts = line.split(" ");
			user = Integer.parseInt(parts[0]) - 1;
			movie = (short) (Short.parseShort(parts[1]) - 1);
			date = (short) (Short.parseShort(parts[2]) - 1);
			rating = (byte) (Byte.parseByte(parts[3]));
			if (dataIndices[lineNum] == 1 || dataIndices[lineNum] == 2
					|| dataIndices[lineNum] == 3) {

				users[trainingDataIndex] = user;
				movies[trainingDataIndex] = movie;
				dates[trainingDataIndex] = date;
				ratings[trainingDataIndex] = rating;
				trainingDataIndex++;
			} else if (dataIndices[lineNum] == 4) {
				probeData[probeDataIndex][0] = user;
				probeData[probeDataIndex][1] = movie;
				probeData[probeDataIndex][2] = date;
				probeData[probeDataIndex][3] = rating;

				probeDataIndex++;
			} else if (dataIndices[lineNum] == 5) {
				qualData[qualDataIndex][0] = user;
				qualData[qualDataIndex][1] = movie;
				qualData[qualDataIndex][2] = date;

				qualDataIndex++;
			}
			lineNum++;
			if (lineNum % 10000000 == 0) {
				System.out.println(timestampLine(lineNum + " / " + NUM_POINTS));
			}
		}

		// Now add probe data onto the end of the four arrays.
		for (int i = 0; i < probeData.length; i++) {
			user = probeData[i][0];
			movie = (short) probeData[i][1];
			date = (short) probeData[i][2];
			rating = (byte) probeData[i][3];

			users[trainingDataIndex] = user;
			movies[trainingDataIndex] = movie;
			dates[trainingDataIndex] = date;
			ratings[trainingDataIndex] = rating;
			trainingDataIndex++;
		}

		System.out.println(timestampLine("Done loading data."));
	}

	private void generateOutput() throws IOException {
		FileWriter fstream = new FileWriter("RBM_1234_with_probe_training");
		BufferedWriter out = new BufferedWriter(fstream);
		int movie, user, date;
		double predictedRating;
		for (int i = 0; i < NUM_TRAINING_PROBE_POINTS; i++) {
			user = users[i];
			movie = movies[i];
			date = dates[i];

			predictedRating = outputRating(movie, user, date);
			out.write(String.format("%.4f\n", predictedRating));
		}
		out.close();

		fstream = new FileWriter("RBM_5_with_probe_training");
		out = new BufferedWriter(fstream);
		for (int i = 0; i < qualData.length; i++) {
			user = qualData[i][0];
			movie = qualData[i][1];
			date = qualData[i][2];

			predictedRating = outputRating(movie, user, date);
			out.write(String.format("%.4f\n", predictedRating));
		}
		out.close();
	}

	private void generateProbeOutput() throws IOException {
		FileWriter fstream = new FileWriter("RBM_123_no_probe_training");
		BufferedWriter out = new BufferedWriter(fstream);
		int user;
		short movie, date;
		double predictedRating;
		for (int j = 0; j < NUM_TRAINING_POINTS; j++) {
			user = users[j];
			movie = (short) movies[j];
			date = (short) dates[j];

			predictedRating = outputRating(movie, user, date);
			out.write(String.format("%.4f\n", predictedRating));
		}
		out.close();

		fstream = new FileWriter("RBM_4_no_probe_training");
		out = new BufferedWriter(fstream);
		// Test the model in probe set.
		for (int j = 0; j < probeData.length; j++) {
			user = probeData[j][0];
			movie = (short) probeData[j][1];
			date = (short) probeData[j][2];

			predictedRating = outputRating(movie, user, date);
			out.write(String.format("%.4f\n", predictedRating));
		}
		out.close();
	}

	public static void main(String[] args) throws NumberFormatException,
			IOException {

		RBM trainer;
		if (args.length == 1) {
			trainer = new RBM(Integer.parseInt(args[0]));
		} else if (args.length == 2) {
			trainer = new RBM(Integer.parseInt(args[0]), Double.parseDouble(args[1]));
		} else {
			System.exit(1);
			return;
		}
		trainer.train();
	}
}
