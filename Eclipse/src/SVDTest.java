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
import java.util.Random;

public class SVDTest {

	public double LRATE_FEATURES_INITIAL = 0.007;
	public double LRATE_USER_BIAS_INITIAL = 3e-3;
	public double LRATE_MOVIE_BIAS_INITIAL = 2e-3;
	public double K_FEATURES_INITIAL = 0.01;
	public double K_USER_BIAS_INITIAL = 0.001;
	public double K_MOVIE_BIAS_INITIAL = 0.001;
//	public double K_FEATURES_INITIAL = 0.008577;
//	public double K_USER_BIAS_INITIAL = 0.094474;
//	public double K_MOVIE_BIAS_INITIAL = 0.005;

	public Random rand;
	
	public static final double MIN_ERROR_DIFF = 0.00001;

	public static final double GLOBAL_MEAN = 3.6033;
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

	public int NUM_FEATURES;
	public int TEST_PARAM;

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

	// Current lrates
	public double LRATE_FEATURES = LRATE_FEATURES_INITIAL;
	public double LRATE_USER_BIAS = LRATE_USER_BIAS_INITIAL;
	public double LRATE_MOVIE_BIAS = LRATE_MOVIE_BIAS_INITIAL;
	public double K_FEATURES = K_FEATURES_INITIAL;
	public double K_USER_BIAS = K_USER_BIAS_INITIAL;
	public double K_MOVIE_BIAS = K_MOVIE_BIAS_INITIAL;

	public double LRATE_USER_BIAS_BEST;
	public double LRATE_MOVIE_BIAS_BEST;
	public double K_FEATURES_BEST;
	public double K_USER_BIAS_BEST;
	public double K_MOVIE_BIAS_BEST;

	public SVDTest(int numFeatures) {
		this.NUM_FEATURES = numFeatures;

		// Initialize things that are specific to the training session.
		initializeVars();
	}

	public SVDTest(int numFeatures, int testParam) {
		this.NUM_FEATURES = numFeatures;
		this.K_FEATURES_INITIAL = testParam;

		// Initialize things that are specific to the training session.
		initializeVars();
	}

	public SVDTest(int numFeatures, double LRATE_BIAS, double LRATE_FEATURES,
			double LRATE_MW, double K_BIAS, double K_FEATURES, double K_MW) {
		// Set the constants to the specified values.
		this.NUM_FEATURES = numFeatures;
		this.LRATE_FEATURES_INITIAL = LRATE_FEATURES;
		this.K_FEATURES_INITIAL = K_FEATURES;

		// Initialize things that are specific to the training session.
		initializeVars();
	}

	private void initializeVars() {
		userFeatures = new double[NUM_USERS][NUM_FEATURES];
		movieFeatures = new double[NUM_MOVIES][NUM_FEATURES];
		userBias = new double[NUM_USERS];
		movieBias = new double[NUM_MOVIES];

		LRATE_FEATURES = LRATE_FEATURES_INITIAL;
		LRATE_USER_BIAS = LRATE_USER_BIAS_INITIAL;
		LRATE_MOVIE_BIAS = LRATE_MOVIE_BIAS_INITIAL;

		rand = new Random(0);
		
		// Initialize weights.
		for (int i = 0; i < userFeatures.length; i++) {
			for (int j = 0; j < userFeatures[i].length; j++) {
				userFeatures[i][j] = (rand.nextFloat() - 0.5) / 50;
			}
		}
		for (int i = 0; i < movieFeatures.length; i++) {
			for (int j = 0; j < movieFeatures[i].length; j++) {
				movieFeatures[i][j] = (rand.nextFloat() - 0.5) / 50;
			}
		}
	}

	private double genRandomParam(double param) {
		// [.1, 10]
		double significand = rand.nextFloat() * 9 + 1;
		int exponent = (int) (rand.nextFloat() * 2 - 2);
		double factor = significand * Math.pow(10, exponent);
		
		return factor * param;
	}

	private void tweakOneParam() {
		int randChoice = (int) (rand.nextFloat() * 5);
		if (randChoice == 0)
			LRATE_USER_BIAS_INITIAL = genRandomParam(LRATE_USER_BIAS_INITIAL);
		else if (randChoice == 1)
			LRATE_MOVIE_BIAS_INITIAL = genRandomParam(LRATE_MOVIE_BIAS_INITIAL);
		else if (randChoice == 2)
			K_FEATURES = genRandomParam(K_FEATURES);
		else if (randChoice == 3)
			K_USER_BIAS = genRandomParam(K_USER_BIAS);
		else if (randChoice == 4)
			K_MOVIE_BIAS = genRandomParam(K_MOVIE_BIAS);
	}

	public void train() throws NumberFormatException, IOException {
		System.out.println(timestampLine(String.format("Training %d features.",
				NUM_FEATURES)));

		// Read in input
		readInput();

		// Set up logfile.
		BufferedWriter logWriter = new BufferedWriter(new FileWriter(LOGFILE, true));
		logWriter.write("\n");

		// Start training
		double bestRmse = Double.MAX_VALUE;
		double rmse;
		// Store current params as the old ones before we tweak a new one.
		LRATE_USER_BIAS_BEST = LRATE_USER_BIAS_INITIAL;
		LRATE_MOVIE_BIAS_BEST = LRATE_MOVIE_BIAS_INITIAL;
		K_FEATURES_BEST = K_FEATURES;
		K_USER_BIAS_BEST = K_USER_BIAS;
		K_MOVIE_BIAS_BEST = K_MOVIE_BIAS;
		while (true) {
			precompute(NUM_TRAINING_POINTS);
			double previousRmse = calcProbeRmse();
			logRmse(logWriter, previousRmse, 0);
			for (int i = 1; true; i++) {
				// TRAIN WITH TRAINING SET ONLY (no probe).
				rmse = trainWithNumPoints(NUM_TRAINING_POINTS);
				logRmse(logWriter, rmse, i);

				// If probe error has been going up, we should stop.
				double rmseDiff = previousRmse - rmse;
				if (rmseDiff < MIN_ERROR_DIFF) {
					System.out.println(String.format(
							"Tried: %.6f %.6f %.6f %.6f %.6f RMSE %.5f (%.2f%%)", 
							LRATE_USER_BIAS_INITIAL, LRATE_MOVIE_BIAS_INITIAL,
							K_FEATURES, K_USER_BIAS, K_MOVIE_BIAS, rmse, 
							(1 - rmse / CINEMATCH_BASELINE) * 100));
					System.out.println(String.format(
							"Best: %.6f %.6f %.6f %.6f %.6f RMSE %.5f (%.2f%%)", 
							LRATE_USER_BIAS_BEST, LRATE_MOVIE_BIAS_BEST,
							K_FEATURES_BEST, K_USER_BIAS_BEST, K_MOVIE_BIAS_BEST, bestRmse,
							(1 - bestRmse / CINEMATCH_BASELINE) * 100));
					break;
				}
				

				// Slow down learning rate as we're getting close to the answer.
				LRATE_FEATURES *= .9;

				previousRmse = rmse;
			}

			// If our rmse is a new low, then save the bestRmse and params. 
			// Otherwise, restore the best params.
			if (rmse < bestRmse) {
				bestRmse = rmse;
				
				LRATE_USER_BIAS_BEST = LRATE_USER_BIAS_INITIAL;
				LRATE_MOVIE_BIAS_BEST = LRATE_MOVIE_BIAS_INITIAL;
				K_FEATURES_BEST = K_FEATURES;
				K_USER_BIAS_BEST = K_USER_BIAS;
				K_MOVIE_BIAS_BEST = K_MOVIE_BIAS;
			} else {
				// Restore old params.
				LRATE_USER_BIAS_INITIAL = LRATE_USER_BIAS_BEST;
				LRATE_MOVIE_BIAS_INITIAL = LRATE_MOVIE_BIAS_BEST;
				K_FEATURES = K_FEATURES_BEST;
				K_USER_BIAS = K_USER_BIAS_BEST;
				K_MOVIE_BIAS = K_MOVIE_BIAS_BEST;
			}

			tweakOneParam();
			initializeVars();
		}
	}

	public double trainWithNumPoints(int numPoints) throws IOException {
		int user;
		short movie, date;
		byte rating;
		double err, uf, mf;

		for (int j = 0; j < numPoints; j++) {
			user = users[j];
			movie = movies[j];
			date = dates[j];
			rating = ratings[j];

			// Calculate the error.
			err = rating - predictRating(movie, user, date);

			// Train biases.
			userBias[user] += LRATE_USER_BIAS * (err - K_USER_BIAS * userBias[user]);
			movieBias[movie] += LRATE_MOVIE_BIAS * (err - K_MOVIE_BIAS * movieBias[movie]);

			// Train all features.
			for (int k = 0; k < NUM_FEATURES; k++) {
				uf = userFeatures[user][k];
				mf = movieFeatures[movie][k];

				userFeatures[user][k] += LRATE_FEATURES * (err * mf - K_FEATURES * uf);
				movieFeatures[movie][k] += LRATE_FEATURES * (err * uf - K_FEATURES * mf);
			}
		}

		// Test the model in probe set.
		return calcProbeRmse();
	}

	public void precompute(int numPoints) throws IOException {
		long ratingSum = 0;
		for (int i = 0; i < probeData.length; i++) {
			ratingSum += probeData[i][3];
		}
		System.out.println(((double) ratingSum) / probeData.length);
	}

	public double predictRating(int movie, int user, int date) {
		// Compute ratings
		double ratingSum = GLOBAL_MEAN;
		// Add in biases.
		ratingSum += userBias[user];
		ratingSum += movieBias[movie];
		// Take dot product of feature vectors.
		for (int i = 0; i < NUM_FEATURES; i++) {
			ratingSum += userFeatures[user][i] * movieFeatures[movie][i];
		}
		return ratingSum;
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

			predictedRating = outputRating(movie, user, date);
			out.write(String.format("%.4f\n", predictedRating));
		}
		out.close();
	}

	public static void main(String[] args) throws NumberFormatException,
			IOException {

		SVDTest trainer;
		if (args.length == 1) {
			trainer = new SVDTest(Integer.parseInt(args[0]));
		} else if (args.length == 2) {
			trainer = new SVDTest(Integer.parseInt(args[0]),
					Integer.parseInt(args[1]));
		} else if (args.length == 7) {
			trainer = new SVDTest(Integer.parseInt(args[0]),
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
