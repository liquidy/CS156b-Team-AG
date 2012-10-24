import java.io.BufferedReader;
import java.io.BufferedWriter;
import java.io.FileInputStream;
import java.io.FileWriter;
import java.io.IOException;
import java.io.InputStream;
import java.io.InputStreamReader;
import java.nio.charset.Charset;

public class OutputGenerator {

	public static final String qualInput = "qual.dta";
	public static final String solutionOutput = "solution.dta";
	
	public void generateOutput(Trainer trainer) throws IOException {
		double[][] userFeatures = trainer.userFeatures;
		double[][] movieFeatures = trainer.movieFeatures;
		
		InputStream qualInputStream = new FileInputStream(qualInput);
		BufferedReader br = new BufferedReader(new InputStreamReader(qualInputStream, Charset.forName("UTF-8")));
		String line;
		
		FileWriter fstream = new FileWriter(solutionOutput);
	  BufferedWriter out = new BufferedWriter(fstream);
	  String[] parts;
	  int movie, user, date;
	  double predictedRating;
	  
		while ((line = br.readLine()) != null) {
			parts = line.split(" ");
	    user = Integer.parseInt(parts[0]) - 1;
	    movie = Integer.parseInt(parts[1]) - 1;
	    date = Integer.parseInt(parts[2]);
	    
	    predictedRating = predictRating(movie, user, userFeatures, movieFeatures);
			out.write(String.format("%.3f\n", predictedRating));
		}
		out.close();
	}
	
	public double predictRating(int movie, int user, double[][] userFeatures, double[][] movieFeatures) {
		double ratingSum = 0;
		for (int i = 0; i < Trainer.NUM_FEATURES; i++) {
			ratingSum += userFeatures[i][user] * movieFeatures[i][movie];
		}
		
		if (ratingSum > 5) {
			ratingSum = 5;
		} else if (ratingSum < 1) {
			ratingSum = 1;
		}
		
    return ratingSum;
	}
}
