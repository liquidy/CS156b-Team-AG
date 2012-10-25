import java.io.BufferedReader;
import java.io.FileInputStream;
import java.io.IOException;
import java.io.InputStream;
import java.io.InputStreamReader;
import java.nio.charset.Charset;

public class Trainer {
	
	public static final int NUM_FEATURES = 1;
	public static final int NUM_EPOCHS = 125;
	public static final double LEARNING_RATE = 0.01;
	public static final double K = 0.015;
	
	public static final int NUM_USERS = 458293;
	public static final int NUM_MOVIES = 17770;
	public static final int NUM_POINTS = 102416306;
	
	public static final String inputFile = "all.dta";
	
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
		int[][] trainingData = new int[NUM_POINTS][4];
		InputStream fis = new FileInputStream(inputFile);
		BufferedReader br = new BufferedReader(new InputStreamReader(fis, Charset.forName("UTF-8")));
		String line;
		
		String[] parts;
		int user, movie, date, rating;
		int lineNum = 0;
		
		System.out.println("Loading data...");
		while ((line = br.readLine()) != null) {
			// Parse each line.
	    parts = line.split(" ");
	    user = Integer.parseInt(parts[0]) - 1;
	    movie = Integer.parseInt(parts[1]) - 1;
	    date = Integer.parseInt(parts[2]);
	    rating = Integer.parseInt(parts[3]);
	    
	    trainingData[lineNum][0] = user;
	    trainingData[lineNum][1] = movie;
	    trainingData[lineNum][2] = date;
	    trainingData[lineNum][3] = rating;
	    
	    lineNum++;
	    if (lineNum % 1000000 == 0)
	    	System.out.println(lineNum + " / " + NUM_POINTS);
		}
		System.out.println("Done loading data.\n");
		
		
//		// Start training
//		for (int i = 0; i < NUM_EPOCHS; i++) {
//		  System.out.println(i + " / " + NUM_EPOCHS);
//			for (int j = 0; j < trainingData.length; j++) {
//		    user = trainingData[j][0];
//		    movie = trainingData[j][1];
//		    date = trainingData[j][2];
//		    rating = trainingData[j][3];
//				
//				// Train all features for each point that we encounter.
//		    double err, uv;
//		    for (int k = 0; k < NUM_FEATURES; k++) {
//		    	err = rating - predictRating(movie, user);
//		    	uv = userFeatures[k][user];
//		    	userFeatures[k][user] += LEARNING_RATE *
//		    			(err * movieFeatures[k][movie] - K * userFeatures[k][user]);
//		    	movieFeatures[k][movie] += LEARNING_RATE *
//		    			(err * uv - K * movieFeatures[k][movie]);
//		    }
//			}
//		}
		// Start training
		for (int i = 0; i < NUM_FEATURES; i++) {
			for (int j = 0; j < NUM_EPOCHS; j++) {
				// Train all features for each point that we encounter.
		    double err, uv;
		    for (int k = 0; k < trainingData.length; k++) {
		    	user = trainingData[k][0];
			    movie = trainingData[k][1];
			    date = trainingData[k][2];
			    rating = trainingData[k][3];
			    
			    err = rating - predictRating(movie, user);
		    	uv = userFeatures[i][user];
		    	userFeatures[i][user] += LEARNING_RATE *
		    			(err * movieFeatures[i][movie] - K * userFeatures[i][user]);
		    	movieFeatures[i][movie] += LEARNING_RATE *
		    			(err * uv - K * movieFeatures[i][movie]);
		    }
		    System.out.println(String.format("feature %d epoch %d / %d", i, j, NUM_EPOCHS));
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
