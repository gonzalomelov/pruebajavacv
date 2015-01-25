package src.main.code;


import java.nio.IntBuffer;


import org.bytedeco.javacpp.opencv_contrib.FaceRecognizer;
import org.bytedeco.javacpp.opencv_core.IplImage;
import org.bytedeco.javacpp.opencv_core.Mat;
import org.bytedeco.javacpp.opencv_core.MatVector;
import org.bytedeco.javacpp.opencv_objdetect.CvHaarClassifierCascade;





import static org.bytedeco.javacpp.opencv_contrib.*;
import static org.bytedeco.javacpp.opencv_core.*;
import static org.bytedeco.javacpp.opencv_highgui.*;
import static org.bytedeco.javacpp.opencv_imgproc.CV_BGR2GRAY;
import static org.bytedeco.javacpp.opencv_imgproc.cvCvtColor;





public class FaceRecognizerPrueba {
	
	private static final String CASCADE_FILE = "resources/haarcascade_frontalface_alt.xml";
	final static CvHaarClassifierCascade cascade = new CvHaarClassifierCascade(cvLoad(CASCADE_FILE));
	private static int TRAIN_IMG_NUMBER = 9;
	private static int NUMBER_OF_PEOPLE = 41;
	private static int NUMBER_OF_FACES = NUMBER_OF_PEOPLE*TRAIN_IMG_NUMBER;
	private static MatVector vec = new MatVector(NUMBER_OF_FACES);

	
	 public static void main(String[] args) {
		 	 
		 	//String dbPath = "C:/Users/gonzalo1/Desktop/imgdb/";
		 	String dbPath = "resources/facesdb/";
		 	String trainingDir = dbPath;

	        Mat labels = new Mat(NUMBER_OF_FACES, 1, CV_32SC1);
	        IntBuffer labelsBuf = labels.getIntBuffer();

	        Mat mt;
	        IplImage trainImage;
	        int index;
	        //para cada persona
	        for (int j = 1; j <= NUMBER_OF_PEOPLE; j++){
	        	//para cada imagen que tiene esa persona
	        	for(int i = 1; i <= TRAIN_IMG_NUMBER; i++){
	        		index = i - 1 + TRAIN_IMG_NUMBER*(j-1);
	        		
	        		trainImage = cvLoadImage(trainingDir + "s" + j + "/" + i +".pgm" );
	        		labelsBuf.put(index, j);
	        		mt = new Mat(preprocessImage(trainImage));
	        		vec.put(index , mt);
	        	}
	        }
	        
	         //FaceRecognizer faceRecognizer = createFisherFaceRecognizer();
	         //FaceRecognizer faceRecognizer = createEigenFaceRecognizer();
	        FaceRecognizer faceRecognizer = createLBPHFaceRecognizer();

	        

	        faceRecognizer.train(vec, labels);

	        int[] plabel = new int[1];
	        double[] pconfidence = new double[1];
	        
	        IplImage tgt;
	        for(int j = 1; j <= NUMBER_OF_PEOPLE; j++){
	        	tgt = cvLoadImage(trainingDir + "s" + j + "/10.pgm" );
	        	
	        	faceRecognizer.predict(new Mat(preprocessImage(tgt)), plabel, pconfidence);
	        	System.out.println("PERSONA: " + j);
		        System.out.println("Predicted label: " + plabel[0]);
		        System.out.println("Predicted confidence: " + pconfidence[0]);
	        	
	        }
	        
	    }
	 
	 
	 public static IplImage preprocessImage(IplImage image){
         IplImage gray = cvCreateImage(cvGetSize(image), IPL_DEPTH_8U, 1);

       
         //CvRect r1 = new CvRect(r.x()-10, r.y()-10, r.width()+10, r.height()+10);
         cvCvtColor(image, gray, CV_BGR2GRAY);


         return gray;
 }
	 

}
