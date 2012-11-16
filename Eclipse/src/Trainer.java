import java.io.BufferedReader;
import java.io.BufferedWriter;
import java.io.FileInputStream;
import java.io.FileOutputStream;
import java.io.FileWriter;
import java.io.IOException;
import java.io.InputStream;
import java.io.InputStreamReader;
import java.io.ObjectOutputStream;
import java.nio.charset.Charset;
import java.text.SimpleDateFormat;
import java.util.Date;

public class Trainer {

	public double LRATE_BIAS_INITIAL = 0.007;
	public double LRATE_FEATURES_INITIAL = 0.007;
	public double LRATE_MW_INITIAL = 0.001;
	public double K_BIAS_INITIAL = 0.001;
	public double K_FEATURES_INITIAL = 0.01;
	public double K_MW_INITIAL = 0.02;

	public static final int NUM_EPOCHS_SPAN = 30;
	public static final double MIN_ERROR_DIFF = 0.00001;

	public static final double GLOBAL_MEAN = 3.6033;
	public static final double CINEMATCH_BASELINE = 0.9514;

	public static final int NUM_USERS = 458293;
	public static final int NUM_MOVIES = 17770;
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

	public int NUM_FEATURES;

	// Data that is stored in memory
	public int[] users = new int[NUM_TRAINING_PROBE_POINTS];
	public short[] movies = new short[NUM_TRAINING_PROBE_POINTS];
	public short[] dates = new short[NUM_TRAINING_PROBE_POINTS];
	public byte[] ratings = new byte[NUM_TRAINING_PROBE_POINTS];
	public int[][] probeData = new int[NUM_4_POINTS][4];
	public int[][] qualData = new int[NUM_5_POINTS][3];

	// Current weights
	public double[][] userFeatures;
	public double[][] movieFeatures;
	public double[] userBias;
	public double[] movieBias;
	public double[][] mw;
	public double[][] sum_mw;
	public double[] norm;

	// Current lrates
	public double LRATE_BIAS;
	public double LRATE_FEATURES;
	public double LRATE_MW;
	public double K_BIAS;
	public double K_FEATURES;
	public double K_MW;

	public Trainer(int numFeatures) {
		this.NUM_FEATURES = numFeatures;

		// Initialize things that are specific to the training session.
		initializeVars();
	}

	public Trainer(int numFeatures, double LRATE_BIAS, double LRATE_FEATURES,
			double LRATE_MW, double K_BIAS, double K_FEATURES, double K_MW) {
		// Set the constants to the specified values.
		this.NUM_FEATURES = numFeatures;
		this.LRATE_BIAS_INITIAL = LRATE_BIAS;
		this.LRATE_FEATURES_INITIAL = LRATE_FEATURES;
		this.LRATE_MW_INITIAL = LRATE_MW;
		this.K_BIAS_INITIAL = K_BIAS;
		this.K_FEATURES_INITIAL = K_FEATURES;
		this.K_MW_INITIAL = K_MW;

		// Initialize things that are specific to the training session.
		initializeVars();
	}

	private void initializeVars() {
		userFeatures = new double[NUM_USERS][NUM_FEATURES];
		movieFeatures = new double[NUM_MOVIES][NUM_FEATURES];
		userBias = new double[NUM_USERS];
		movieBias = new double[NUM_MOVIES];
		mw = new double[NUM_USERS][NUM_FEATURES];
		sum_mw = new double[NUM_USERS][NUM_FEATURES];
		norm = new double[NUM_USERS];

		LRATE_BIAS = LRATE_BIAS_INITIAL;
		LRATE_FEATURES = LRATE_FEATURES_INITIAL;
		LRATE_MW = LRATE_MW_INITIAL;
		K_BIAS = K_BIAS_INITIAL;
		K_FEATURES = K_FEATURES_INITIAL;
		K_MW = K_MW_INITIAL;

		// Initialize weights.
		for (int i = 0; i < userFeatures.length; i++) {
			for (int j = 0; j < userFeatures[i].length; j++) {
				userFeatures[i][j] = (Math.random() - 0.5) / 5000;
			}
		}
		for (int i = 0; i < movieFeatures.length; i++) {
			for (int j = 0; j < movieFeatures[i].length; j++) {
				movieFeatures[i][j] = (Math.random() - 0.5) / 5000;
			}
		}
	}

	public void train() throws NumberFormatException, IOException {
		System.out.println(timestampLine(String.format("Training %d features.",
				NUM_FEATURES)));
		System.out.println(String.format("%.3f %.3f %.3f %.3f %.3f %.3f",
				LRATE_BIAS, LRATE_FEATURES, LRATE_MW, K_BIAS, K_FEATURES, K_MW));

		// Read in input
		readInput();

		// Set up logfile.
		BufferedWriter logWriter = new BufferedWriter(new FileWriter(LOGFILE, true));
		logWriter.write("\n");

		// Start training
		precompute(NUM_TRAINING_POINTS);
		double previousRmse = calcProbeRmse();
		logRmse(logWriter, previousRmse, 0);
		int numEpochsToTrain = 0;
		for (int i = 1; true; i++) {
			// TRAIN WITH TRAINING SET ONLY (no probe).
			double rmse = trainWithNumPoints(NUM_TRAINING_POINTS);
			logRmse(logWriter, rmse, i);

			// If probe error has been going up, we should stop.
			double rmseDiff = previousRmse - rmse;
			if (rmseDiff < MIN_ERROR_DIFF) {
				System.out
						.println(timestampLine("Probe error has started"
								+ " to go up significantly; memorizing number of epochs to train."));
				numEpochsToTrain = i;
				break;
			}

			// Slow down learning rate as we're getting close to the answer.
			// if (rmseDiff < 0.0005) {
			// LRATE_BIAS = Math.min(LRATE_BIAS, 0.0005);
			// LRATE_FEATURES = Math.min(LRATE_FEATURES, 0.0005);
			// } else if (rmseDiff < 0.001) {
			// LRATE_BIAS = Math.min(LRATE_BIAS, 0.001);
			// LRATE_FEATURES = Math.min(LRATE_FEATURES, 0.001);
			// }
			LRATE_FEATURES *= .9;
			LRATE_BIAS *= .9;
			LRATE_MW *= .9;

			previousRmse = rmse;
		}
		generateOutput(String.format("%5.0f_no_probe", previousRmse * 100000));

		// TRAIN WITH PROBE.
		System.out.println("Re-training for " + numEpochsToTrain
				+ " epochs, but now with probe included.\n");
		initializeVars();
		precompute(NUM_TRAINING_PROBE_POINTS);
		for (int i = 1; i <= numEpochsToTrain + NUM_EPOCHS_SPAN; i++) {
			// Train with training set AND probe.
			double rmse = trainWithNumPoints(NUM_TRAINING_PROBE_POINTS);
			logRmse(logWriter, rmse, i);

			// Update LRATE's
			LRATE_FEATURES *= .9;
			LRATE_BIAS *= .9;
			LRATE_MW *= .9;

			if (i >= numEpochsToTrain) {
				generateOutput(String.format("%5.0f_%d", previousRmse * 100000, i));
			}
		}
		saveBestParams();

		System.out.println("Done!");
	}

	public double trainWithNumPoints(int numPoints) throws IOException {
		int user;
		short movie, date;
		byte rating;
		int prevUser = -1;
		double err, ub, mb, uf, mf, tmp_mw;
		short m;
		double[] tmp_sum = new double[NUM_FEATURES];

		for (int j = 0; j < numPoints; j++) {
			user = users[j];
			movie = movies[j];
			date = dates[j];
			rating = ratings[j];

			// Determine if current user has changed.
			if (user != prevUser) {
				// Reset sum_mw and calculate sums
				for (int k = 0; k < NUM_FEATURES; k++) {
					sum_mw[user][k] = 0;
				}
				for (int l = j; l < numPoints && users[l] == user; l++) {
					m = movies[l];
					for (int k = 0; k < NUM_FEATURES; k++) {
						sum_mw[user][k] += mw[m][k];
					}
				}
				// Reset and calculate tmp_sum
				for (int k = 0; k < NUM_FEATURES; k++) {
					tmp_sum[k] = 0;
				}
			}
			prevUser = user;

			// Calculate the error.
			err = rating - predictRating(movie, user);

			// Train biases.
			ub = userBias[user];
			mb = movieBias[movie];
			userBias[user] += LRATE_BIAS * (err - K_BIAS * ub);
			movieBias[movie] += LRATE_BIAS * (err - K_BIAS * mb);

			// Train all features for each point that we encounter.
			for (int k = 0; k < NUM_FEATURES; k++) {
				uf = userFeatures[user][k];
				mf = movieFeatures[movie][k];

				userFeatures[user][k] += LRATE_FEATURES * (err * mf - K_FEATURES * uf);
				movieFeatures[movie][k] += LRATE_FEATURES
						* (err * (uf + norm[user] * sum_mw[user][k]) - K_FEATURES * mf);

				// Update tmp_sum, which sums the gradients for mw
				tmp_sum[k] += err * norm[user] * mf;
			}

			// Check if this is the last movie for that user. If so,
			// train movie weights.
			if (j + 1 == numPoints || users[j + 1] != user) {
				for (int l = j; l >= 0 && users[l] == user; l--) {
					m = movies[l];
					for (int k = 0; k < NUM_FEATURES; k++) {
						tmp_mw = mw[m][k];
						mw[m][k] += LRATE_MW * (tmp_sum[k] - K_MW * tmp_mw);
						sum_mw[user][k] += mw[m][k] - tmp_mw;
					}
				}
			}
		}
		// Recalculate sum_mw
		for (int j = 0; j < NUM_USERS; j++) {
			for (int k = 0; k < NUM_FEATURES; k++) {
				sum_mw[j][k] = 0;
			}
		}
		for (int j = 0; j < numPoints; j++) {
			user = users[j];
			movie = movies[j];
			for (int k = 0; k < NUM_FEATURES; k++) {
				sum_mw[user][k] += mw[movie][k];
			}
		}

		// Test the model in probe set.
		return calcProbeRmse();
	}

	public void precompute(int numPoints) {
		int user;
		for (int j = 0; j < numPoints; j++) {
			user = users[j];
			norm[user]++;
		}
		for (int j = 0; j < norm.length; j++) {
			if (norm[j] == 0) {
				norm[j] = 1;
			} else {
				norm[j] = 1 / Math.sqrt(norm[j]);
			}
		}
		System.out.println(timestampLine("Finished precomputation.\n"));
	}

	public double predictRating(int movie, int user) {
		double ratingSum = 0;
		ratingSum += GLOBAL_MEAN;
		// Add in movie and user biases.
		ratingSum += movieBias[movie];
		ratingSum += userBias[user];
		// Take dot product of feature vectors.
		for (int i = 0; i < NUM_FEATURES; i++) {
			ratingSum += (userFeatures[user][i] + sum_mw[user][i] * norm[user])
					* movieFeatures[movie][i];
		}
		return ratingSum;
	}

	public double outputRating(int movie, int user) {
		double rating = predictRating(movie, user);
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

			rmse += Math.pow(rating - predictRating(movie, user), 2);
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
			date = (short) (Short.parseShort(parts[2]));
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

	private void saveBestParams() throws IOException {
		// Save params
		// Save bestUserFeatures
		FileOutputStream fileOut = new FileOutputStream("userFeatures");
		ObjectOutputStream objOut = new ObjectOutputStream(fileOut);
		objOut.writeObject(userFeatures);
		objOut.close();
		fileOut.close();
		// Save bestMovieFeatures
		fileOut = new FileOutputStream("movieFeatures");
		objOut = new ObjectOutputStream(fileOut);
		objOut.writeObject(movieFeatures);
		objOut.close();
		fileOut.close();
		// Save bestUserBias
		fileOut = new FileOutputStream("userBias");
		objOut = new ObjectOutputStream(fileOut);
		objOut.writeObject(userBias);
		objOut.close();
		fileOut.close();
		// Save bestMovieBias
		fileOut = new FileOutputStream("movieBias");
		objOut = new ObjectOutputStream(fileOut);
		objOut.writeObject(movieBias);
		objOut.close();
		fileOut.close();
		// Save best_mw
		fileOut = new FileOutputStream("mw");
		objOut = new ObjectOutputStream(fileOut);
		objOut.writeObject(mw);
		objOut.close();
		fileOut.close();
		// Save best_sum_mw
		fileOut = new FileOutputStream("sum_mw");
		objOut = new ObjectOutputStream(fileOut);
		objOut.writeObject(sum_mw);
		objOut.close();
		fileOut.close();
	}

	private void generateOutput(String fileName) throws IOException {
		FileWriter fstream = new FileWriter(fileName);
		BufferedWriter out = new BufferedWriter(fstream);
		int movie, user, date;
		double predictedRating;

		for (int i = 0; i < qualData.length; i++) {
			user = qualData[i][0];
			movie = qualData[i][1];
			date = qualData[i][2];

			predictedRating = outputRating(movie, user);
			out.write(String.format("%.4f\n", predictedRating));
		}
		out.close();
	}

	public static void main(String[] args) throws NumberFormatException,
			IOException {

		Trainer trainer;
		if (args.length == 1) {
			trainer = new Trainer(Integer.parseInt(args[0]));
		} else if (args.length == 7) {
			trainer = new Trainer(Integer.parseInt(args[0]),
					Double.parseDouble(args[1]), Double.parseDouble(args[2]),
					Double.parseDouble(args[3]), Double.parseDouble(args[4]),
					Double.parseDouble(args[5]), Double.parseDouble(args[6]));
		} else {
			System.exit(1);
			return;
		}
		trainer.train();
	}
}
