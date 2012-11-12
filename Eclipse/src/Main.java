import java.io.IOException;

public class Main {
	
	public static void main(String[] args) throws NumberFormatException, IOException {
		OutputGenerator outputGen = new OutputGenerator();
		Trainer trainer = new Trainer(outputGen, Integer.parseInt(args[0]));
		
		trainer.train();
	}
}
