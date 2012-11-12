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
	
	public void generateOutput(Trainer trainer, String fileName) throws IOException {
		InputStream qualInputStream = new FileInputStream(qualInput);
		BufferedReader br = new BufferedReader(
				new InputStreamReader(qualInputStream, Charset.forName("UTF-8")));
		String line;
		
		FileWriter fstream = new FileWriter(fileName);
	  BufferedWriter out = new BufferedWriter(fstream);
	  String[] parts;
	  int movie, user, date;
	  double predictedRating;
	  
		while ((line = br.readLine()) != null) {
			parts = line.split(" ");
	    user = Integer.parseInt(parts[0]) - 1;
	    movie = Integer.parseInt(parts[1]) - 1;
	    date = Integer.parseInt(parts[2]);
	    
	    predictedRating = trainer.outputRating(movie, user);
			out.write(String.format("%.3f\n", predictedRating));
		}
		out.close();
	}
}
