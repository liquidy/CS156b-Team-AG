import java.io.BufferedReader;
import java.io.FileInputStream;
import java.io.IOException;
import java.io.InputStream;
import java.io.InputStreamReader;
import java.nio.charset.Charset;

public class Trainer {
	
	public static final int NUM_FEATURES = 10;
	public static final int NUM_EPOCHS = 10;
	public static final double LEARNING_RATE = 1e-3;
	
	public static final int NUM_USERS = 458293;
	public static final int NUM_MOVIES = 17770;
	
	public static final String inputFile = "all.dta";
	
	public double[][] userFeatures = new double[NUM_FEATURES][NUM_USERS];
	public double[][] movieFeatures = new double[NUM_FEATURES][NUM_MOVIES];
	
	public Trainer() {
		// Initialize weights.
		for (int i = 0; i < userFeatures.length; i++) {
			for (int j = 0; j < userFeatures[i].length; j++) {
				userFeatures[i][j] = 2.0 / Math.sqrt(NUM_FEATURES);
			}
		}
	}
	
	public void train() throws NumberFormatException, IOException {
		// Read input into memory
		int[][] trainingData = new int[102416306][4];
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
	    	System.out.println(lineNum);
		}
		System.out.println("Done loading data.\n");
		
		
		// Start training
		for (int i = 0; i < NUM_EPOCHS; i++) {
			for (int j = 0; j < trainingData.length; j++) {
		    user = trainingData[j][0];
		    movie = trainingData[j][1];
		    date = trainingData[j][2];
		    rating = trainingData[j][3];
				
				// Train all features for each point that we encounter.
		    double err, uv;
		    for (int k = 0; k < NUM_FEATURES; k++) {
		    	err = LEARNING_RATE * (rating - predictRating(movie, user));
		    	uv = userFeatures[k][user];
		    	userFeatures[k][user] += err * movieFeatures[k][movie];
		    	movieFeatures[k][movie] += err * uv;
		    }
			}
			System.out.println(i + " / " + NUM_EPOCHS);
		}
	}
	
	public double predictRating(int movie, int user) {
		double ratingSum = 0;
		for (int i = 0; i < NUM_FEATURES; i++) {
			ratingSum += userFeatures[i][user] * movieFeatures[i][movie];
		}
    return ratingSum;
	}
}
