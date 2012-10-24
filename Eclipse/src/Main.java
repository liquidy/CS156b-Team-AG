import java.io.IOException;

public class Main {
	
	public static void main(String[] args) throws NumberFormatException, IOException {
		Trainer trainer = new Trainer();
		trainer.train();
		
		OutputGenerator outputGen = new OutputGenerator();
		outputGen.generateOutput(trainer);
	}
}
