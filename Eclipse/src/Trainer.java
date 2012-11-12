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
	
	public static double LRATE_BIAS = 0.01;
	public static double LRATE_FEATURES = 0.01;
	public static final double K_BIAS = 0.05;
	public static final double K_FEATURES = 0.02;
	public static final double MAX_PROBE_ERROR_DIFF = 0.001;
	
	public static final double GLOBAL_MEAN = 3.6033;
	public static final double CINEMATCH_BASELINE = 0.9514;
	
	public static final int NUM_USERS = 458293;
	public static final int NUM_MOVIES = 17770;
	public static final int NUM_POINTS = 102416306;
	public static final int NUM_1_POINTS = 94362233;
	public static final int NUM_2_POINTS = 1965045;
	public static final int NUM_3_POINTS = 1964391;
	public static final int NUM_4_POINTS = 1374739;
	public static final int NUM_TRAINING_POINTS = 
			NUM_1_POINTS + NUM_2_POINTS + NUM_3_POINTS;
	
	public static final String INPUT_DATA = "all.dta";
	public static final String INPUT_INDEX = "all.idx";
	public static final String LOGFILE = "log.txt";
	
	public OutputGenerator outputGen;
	public int NUM_FEATURES;
	
	// Current weights
	public double[][] userFeatures;
	public double[][] movieFeatures;
	public double[] userBias = new double[NUM_USERS];
	public double[] movieBias = new double[NUM_MOVIES];
	
	// Best (so far) weights + probe RMSE measure
	public double[][] bestUserFeatures;
	public double[][] bestMovieFeatures;
	public double[] bestUserBias = new double[NUM_USERS];
	public double[] bestMovieBias = new double[NUM_MOVIES];
	public double bestRmse = Double.MAX_VALUE;
	
	public Trainer(OutputGenerator outputGen, int numFeatures) {
		// Set the constants to the specified values.
		this.outputGen = outputGen;
		this.NUM_FEATURES = numFeatures;
		userFeatures = new double[NUM_USERS][NUM_FEATURES];
		movieFeatures = new double[NUM_MOVIES][NUM_FEATURES];
		bestUserFeatures = new double[NUM_USERS][NUM_FEATURES];
		bestMovieFeatures = new double[NUM_MOVIES][NUM_FEATURES];
		
		// Initialize weights.
		for (int i = 0; i < userFeatures.length; i++) {
			for (int j = 0; j < userFeatures[i].length; j++) {
				userFeatures[i][j] = Math.sqrt(GLOBAL_MEAN / NUM_FEATURES);
			}
		}
		for (int i = 0; i < movieFeatures.length; i++) {
			for (int j = 0; j < movieFeatures[i].length; j++) {
				movieFeatures[i][j] = Math.sqrt(GLOBAL_MEAN / NUM_FEATURES);
			}
		}
		for (int i = 0; i < userBias.length; i++) {
			userBias[i] = -.1;
		}
		for (int i = 0; i < movieBias.length; i++) {
			movieBias[i] = .1;
		}
	}
	
	public void train() throws NumberFormatException, IOException {
		System.out.println(timestampLine(String.format("Training %d features.", NUM_FEATURES)));
		System.out.println(timestampLine("Instantiating objects..."));
		
		// Read input into memory
		int[] users = new int[NUM_TRAINING_POINTS];
		short[] movies = new short[NUM_TRAINING_POINTS];
		short[] dates = new short[NUM_TRAINING_POINTS];
		byte[] ratings = new byte[NUM_TRAINING_POINTS];
		int[][] probeData = new int[NUM_4_POINTS][4];
		InputStream fis = new FileInputStream(INPUT_DATA);
		BufferedReader br = new BufferedReader(new InputStreamReader(fis, Charset.forName("UTF-8")));
		InputStream fisIdx = new FileInputStream(INPUT_INDEX);
		BufferedReader brIdx = new BufferedReader(new InputStreamReader(fisIdx, Charset.forName("UTF-8")));
		
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
		int trainingDataIndex = 0, probeDataIndex = 0;
		while ((line = br.readLine()) != null) {
			if (dataIndices[lineNum] == 1 || dataIndices[lineNum] == 2 ||
					dataIndices[lineNum] == 3) {
				
				// Parse each line.
		    parts = line.split(" ");
		    user = Integer.parseInt(parts[0]) - 1;
		    movie = (short) (Short.parseShort(parts[1]) - 1);
		    date = (short) (Short.parseShort(parts[2]) - 1);
		    rating = (byte) (Byte.parseByte(parts[3]) - 1);
		    
		    users[trainingDataIndex] = user;
		    movies[trainingDataIndex] = movie;
		    dates[trainingDataIndex] = date;
		    ratings[trainingDataIndex] = rating;
		    
		    trainingDataIndex++;
			} else if (dataIndices[lineNum] == 4) {
				parts = line.split(" ");
		    user = Integer.parseInt(parts[0]) - 1;
		    movie = (short) (Short.parseShort(parts[1]) - 1);
		    date = (short) (Short.parseShort(parts[2]) - 1);
		    rating = (byte) (Byte.parseByte(parts[3]) - 1);
		    
		    probeData[probeDataIndex][0] = user;
		    probeData[probeDataIndex][1] = movie;
		    probeData[probeDataIndex][2] = date;
		    probeData[probeDataIndex][3] = rating;
		    
		    probeDataIndex++;
			}
			lineNum++;
			if (lineNum % 10000000 == 0) {
				System.out.println(timestampLine(lineNum + " / " + NUM_POINTS));
			}
		}
		System.out.println(timestampLine("Done loading data.\n"));
		
		// Set up logfile.
		BufferedWriter logWriter = new BufferedWriter(new FileWriter(LOGFILE, true));
		logWriter.write("\n");
		
		// Start training
		double previousRmse = Double.MAX_VALUE;
		double err, ub, mb, uf, mf;
		for (int i = 1; true; i++) {
			for (int j = 0; j < NUM_TRAINING_POINTS; j++) {
		    user = users[j];
		    movie = movies[j];
		    date = dates[j];
		    rating = ratings[j];
		    
		    // Train biases.
		    err = rating - predictRating(movie, user);
		    ub = userBias[user];
	    	mb = movieBias[movie];
	    	userBias[user] += LRATE_BIAS * (err - K_BIAS * ub);
	    	movieBias[movie] += LRATE_BIAS * (err - K_BIAS * mb);
		    
				// Train all features for each point that we encounter.
		    for (int k = 0; k < NUM_FEATURES; k++) {
		    	err = rating - predictRating(movie, user);
		    	uf = userFeatures[user][k];
		    	mf = movieFeatures[movie][k];
		    	userFeatures[user][k] += LRATE_FEATURES * (err * mf - K_FEATURES * uf);
		    	movieFeatures[movie][k] += LRATE_FEATURES * (err * uf - K_FEATURES * mf);
		    }
			}
			
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
			
			// Print + log some stats.
			double predictedPercent = (1 - rmse / CINEMATCH_BASELINE) * 100;
			String currentDate = new SimpleDateFormat("h:mm:ss a").format(new Date());
			String logline = currentDate +
					String.format(": epoch %d probe RMSE %.5f (%.2f%%) ",
							i, rmse, predictedPercent);
			System.out.println(logline);
			logWriter.write(logline + "\n");

			// If probe error has been going up, we should stop.
			double bestRmseDiff = rmse - bestRmse;
			if (bestRmseDiff > MAX_PROBE_ERROR_DIFF) {
				System.out.println(timestampLine("Probe error has started to go up significantly; stopping computation."));
				
				// Save params
				FileOutputStream fileOut = new FileOutputStream("userFeatures");
		    ObjectOutputStream objOut = new ObjectOutputStream(fileOut);
		    objOut.writeObject(bestUserFeatures);
		    objOut.close();
		    fileOut.close();
		    fileOut = new FileOutputStream("movieFeatures");
		    objOut = new ObjectOutputStream(fileOut);
		    objOut.writeObject(bestMovieFeatures);
		    objOut.close();
		    fileOut.close();
			
				// Generate solution and save it.
				outputGen.generateOutput(this, String.format("%d_%5.0f", i, rmse * 100000));
				
				break;
			}
			
			// Update the best RMSE and save weights.
		  if (rmse < bestRmse) {
				bestRmse = rmse;
				bestUserFeatures = copy2dArray(userFeatures);
				bestMovieFeatures = copy2dArray(movieFeatures);
				bestUserBias = userBias.clone();
				bestMovieBias = movieBias.clone();
		  }
		  
		  // Slow down learning rate if we're getting close to the answer.
		  double rmseDiff = previousRmse - rmse;
		  if (rmseDiff < 0.0001) {
				LRATE_BIAS = Math.min(LRATE_BIAS, 0.0001);
				LRATE_FEATURES = Math.min(LRATE_FEATURES, 0.0001);
			} else if (rmseDiff < 0.001) {
				LRATE_BIAS = Math.min(LRATE_BIAS, 0.001);
				LRATE_FEATURES = Math.min(LRATE_FEATURES, 0.001);
		  }
			previousRmse = rmse;
		}
		System.out.println("Done!");
	}
	
	public double predictRating(int movie, int user) {
		double ratingSum = 0;
//		ratingSum += GLOBAL_MEAN;
		// Add in movie and user biases.
//		ratingSum += movieBias[movie];
//		ratingSum += userBias[user];
		// Take dot product of feature vectors.
		for (int i = 0; i < NUM_FEATURES; i++) {
			ratingSum += userFeatures[user][i] * movieFeatures[movie][i];
		}
		ratingSum = addAndClip(ratingSum, 0);
    return ratingSum;
	}
	
	public double outputRating(int movie, int user) {
		double ratingSum = 0;
		for (int i = 0; i < NUM_FEATURES; i++) {
			ratingSum += bestUserFeatures[user][i] * bestMovieFeatures[movie][i];
		}
		ratingSum = addAndClip(ratingSum, 0);
	  return ratingSum;
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
	
	private double[][] copy2dArray(double[][] arrayToCopy) {
		double[][] arrayCopy = new double[arrayToCopy.length][];
		for (int i = 0; i < arrayToCopy.length; i++) {
			arrayCopy[i] = arrayToCopy[i].clone();
		}
		return arrayCopy;
	}
}
