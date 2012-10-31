import java.io.BufferedReader;
import java.io.FileInputStream;
import java.io.IOException;
import java.io.InputStream;
import java.io.InputStreamReader;
import java.nio.charset.Charset;

public class Trainer {
	
	public static final int NUM_FEATURES = 5;
	public static final int NUM_EPOCHS = 50;
	public static final double LEARNING_RATE = 0.01;
	public static final double K = 0.015;
	
	public static final int NUM_USERS = 458293;
	public static final int NUM_MOVIES = 17770;
	public static final int NUM_POINTS = 102416306;
	public static final int NUM_1_POINTS = 94362233;
	
	public static final String inputData = "all.dta";
	public static final String inputIndex = "all.idx";
	
	public double[][] userFeatures = new double[NUM_FEATURES][NUM_USERS];
	public double[][] movieFeatures = new double[NUM_FEATURES][NUM_MOVIES];
	
	public Trainer() {
		// Initialize weights.
		for (int i = 0; i < userFeatures.length; i++) {
			for (int j = 0; j < userFeatures[i].length; j++) {
				userFeatures[i][j] = 0.1;
			}
		}
		for (int i = 0; i < movieFeatures.length; i++) {
			for (int j = 0; j < movieFeatures[i].length; j++) {
				movieFeatures[i][j] = 0.1;
			}
		}
	}
	
	public void train() throws NumberFormatException, IOException {
		System.out.println(String.format("Training %d features with %d epochs.", NUM_FEATURES, NUM_EPOCHS));
		
		// Read input into memory
		int[][] trainingData = new int[NUM_1_POINTS][4];
		InputStream fis = new FileInputStream(inputData);
		BufferedReader br = new BufferedReader(new InputStreamReader(fis, Charset.forName("UTF-8")));
		InputStream fisIdx = new FileInputStream(inputIndex);
		BufferedReader brIdx = new BufferedReader(new InputStreamReader(fisIdx, Charset.forName("UTF-8")));
		
		// Read inputIndex
		byte[] dataIndices = new byte[NUM_POINTS];
		String line;
		byte index;
		int lineNum = 0;
		while ((line = brIdx.readLine()) != null) {
			index = Byte.parseByte(line);
			dataIndices[lineNum] = index;
			lineNum++;
		}
		
		// Read inputData
		System.out.println("Loading data...");
		String[] parts;
		int user, movie, date, rating;
		lineNum = 0;
		int trainingDataIndex = 0;
		while ((line = br.readLine()) != null) {
			if (dataIndices[lineNum] == 1) {
				// Parse each line.
		    parts = line.split(" ");
		    user = Integer.parseInt(parts[0]) - 1;
		    movie = Integer.parseInt(parts[1]) - 1;
		    date = Integer.parseInt(parts[2]);
		    rating = Integer.parseInt(parts[3]);
		    
		    trainingData[trainingDataIndex][0] = user;
		    trainingData[trainingDataIndex][1] = movie;
		    trainingData[trainingDataIndex][2] = date;
		    trainingData[trainingDataIndex][3] = rating;
		    
		    trainingDataIndex++;
			}
			lineNum++;
			if (lineNum % 1000000 == 0) {
				System.out.println(lineNum + " / " + NUM_POINTS);
			}
		}
		System.out.println("Done loading data.\n");
		
		// Start training
		for (int i = 0; i < NUM_EPOCHS; i++) {
			for (int j = 0; j < trainingData.length; j++) {
				if (j % 10000000 == 0) {
					System.out.println(String.format("epoch %d point %d", i, j));
				}
				
		    user = trainingData[j][0];
		    movie = trainingData[j][1];
		    date = trainingData[j][2];
		    rating = trainingData[j][3];
				
				// Train all features for each point that we encounter.
		    double err, uv;
		    for (int k = 0; k < NUM_FEATURES; k++) {
		    	err = rating - predictRating(movie, user);
		    	uv = userFeatures[k][user];
		    	userFeatures[k][user] += LEARNING_RATE * (err * movieFeatures[k][movie] - K * uv);
		    	movieFeatures[k][movie] += LEARNING_RATE * (err * uv - K * movieFeatures[k][movie]);
		    }
			}
		}
	}
	
	public double predictRating(int movie, int user) {
		double ratingSum = 0;
		for (int i = 0; i < NUM_FEATURES; i++) {
			ratingSum += userFeatures[i][user] * movieFeatures[i][movie];
			if (ratingSum > 5) {
				ratingSum = 5;
			} else if (ratingSum < 1) {
				ratingSum = 1;
			}
		}
    return ratingSum;
	}
}
