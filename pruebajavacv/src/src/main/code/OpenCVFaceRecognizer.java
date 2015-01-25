package src.main.code;

import java.io.File;
import java.io.FilenameFilter;
import java.nio.IntBuffer;

import org.bytedeco.javacpp.Loader;
import org.bytedeco.javacpp.opencv_contrib.FaceRecognizer;
import org.bytedeco.javacpp.opencv_core.Mat;
import org.bytedeco.javacpp.opencv_core.MatVector;
import org.bytedeco.javacpp.opencv_objdetect;
import org.bytedeco.javacpp.opencv_objdetect.CvHaarClassifierCascade;
import org.bytedeco.javacpp.opencv_core.CvRect;



import static org.bytedeco.javacpp.opencv_contrib.*;
import static org.bytedeco.javacpp.opencv_core.*;
import static org.bytedeco.javacpp.opencv_highgui.*;
import static org.bytedeco.javacpp.opencv_objdetect.cvHaarDetectObjects;
import static org.bytedeco.javacpp.opencv_imgproc.CV_BGR2GRAY;
import static org.bytedeco.javacpp.opencv_imgproc.cvCvtColor;
import static org.bytedeco.javacpp.opencv_imgproc.cvEqualizeHist;
import static org.bytedeco.javacpp.opencv_imgproc.CV_INTER_LINEAR;
import static org.bytedeco.javacpp.opencv_imgproc.INTER_NEAREST;
import static org.bytedeco.javacpp.opencv_imgproc.cvResize;




public class OpenCVFaceRecognizer {
	
	private static final String CASCADE_FILE = "resources/haarcascade_frontalface_alt.xml";
	final static CvHaarClassifierCascade cascade = new CvHaarClassifierCascade(cvLoad(CASCADE_FILE));
	private static int TRAIN_IMG_NUMBER = 10;
	private static MatVector vec = new MatVector(2*TRAIN_IMG_NUMBER);
	
	 public static void main(String[] args) {
		 
		 	String terry = "terry_target.jpg";
		 	String aniston = "aniston.png";
		 	String lawrence = "lawrence.png";
		 	String terry2 = "terry2.jpg";
		 	String terry3 = "otraterry.jpg";
		 	String ronaldo = "cr7_target.jpg";
		 	String ronaldo2 = "cr74.jpg";
		 

		 	String path = "resources";
		 	String pathImg2 = "resources/targetimg/";
		 	String resultPath = path + "/output/";
	        String trainingDir = path;

	        
	        IplImage target2 = new IplImage();
            target2 = cvLoadImage(pathImg2 + terry3);
            CvSeq faces2 = detectFace(target2);
            CvRect r2 = new CvRect(cvGetSeqElem(faces2,0));
            target2=preprocessImage(target2, r2);
            IplImage target = IplImage.create(92,112, IPL_DEPTH_8U, 1);
            cvResize(target2 , target);
            cvSaveImage(resultPath+"foto.jpg", target);
            Mat targetImage = new Mat(target);

	        File root = new File(trainingDir);

	        FilenameFilter imgFilter = new FilenameFilter() {
	            public boolean accept(File dir, String name) {
	                name = name.toLowerCase();
	                return name.endsWith(".jpg") || name.endsWith(".pgm") || name.endsWith(".png");
	            }
	        };

	        File[] imageFiles = root.listFiles(imgFilter);

	        MatVector images = new MatVector(imageFiles.length);

	        Mat labels = new Mat(imageFiles.length, 1, CV_32SC1);
	        IntBuffer labelsBuf = labels.getIntBuffer();

	        int counter = 0;

	        IplImage[] trainImages = new IplImage[TRAIN_IMG_NUMBER];
	        
	        //MatVector vec = new MatVector();
	        IplImage imgs, destination;
	        Mat mt;
	        for(int i=1; i <= TRAIN_IMG_NUMBER; i++){
                trainImages[i-1]=cvLoadImage("resources/terry"+i+".jpg");
                CvSeq faces = detectFace(trainImages[i-1]);
                CvRect r = new CvRect(cvGetSeqElem(faces,0));
                //trainImages[i-1]=preprocessImage(trainImages[i-1], r);
                trainImages[i-1] = preprocessImage(trainImages[i-1], r);
               
                labelsBuf.put(i-1, 1);
                	destination = IplImage.create(target.width(),target.height(), IPL_DEPTH_8U, 1);
                	cvResize(trainImages[i-1] , destination);

                
                	cvSaveImage(resultPath+i+".jpg", destination);
                mt = new Mat(destination);
                vec.put(i-1, mt);
            }
	        
	        trainImages = new IplImage[TRAIN_IMG_NUMBER];
	        int val = 10;
	        for(int i=1; i <= TRAIN_IMG_NUMBER; i++){
                trainImages[i-1]=cvLoadImage("resources/cr7"+i+".jpg");
                CvSeq faces = detectFace(trainImages[i-1]);
                CvRect r = new CvRect(cvGetSeqElem(faces,0));
                //trainImages[i-1]=preprocessImage(trainImages[i-1], r);
                trainImages[i-1] = preprocessImage(trainImages[i-1], r);
                labelsBuf.put(i-1+TRAIN_IMG_NUMBER, 2);
                	destination = IplImage.create(target.width(),target.height(), IPL_DEPTH_8U, 1);
                	cvResize(trainImages[i-1] , destination);
                	val = 10 +i;
                	cvSaveImage(resultPath+val+".jpg", destination);
                
                mt = new Mat(destination);
                vec.put(i-1 +TRAIN_IMG_NUMBER, mt);
            }
	        
	       
	        


	        //FaceRecognizer faceRecognizer = createFisherFaceRecognizer();
	         //FaceRecognizer faceRecognizer = createEigenFaceRecognizer();
	         FaceRecognizer faceRecognizer = createLBPHFaceRecognizer(2,16,16,16,500.0);

	        

	        faceRecognizer.train(vec, labels);

	        int[] plabel = new int[1];
	        double[] pconfidence = new double[1];
	        
	        faceRecognizer.predict(targetImage, plabel, pconfidence);

	        System.out.println("Predicted label: " + plabel[0]);
	        System.out.println("Predicted confidence: " + pconfidence[0]);
	    }
	 
	 protected static CvSeq detectFace(IplImage originalImage) {
         CvSeq faces = null;
         Loader.load(opencv_objdetect.class);
         try {
                 IplImage grayImage = IplImage.create(originalImage.width(), originalImage.height(), IPL_DEPTH_8U, 1);
                 cvCvtColor(originalImage, grayImage, CV_BGR2GRAY);
                 CvMemStorage storage = CvMemStorage.create();
                 faces = cvHaarDetectObjects(grayImage, cascade, storage, 1.1, 1, 0);

         } catch (Exception e) {
                 e.printStackTrace();
         }
         return faces;
 }
	 
	 public static IplImage preprocessImage(IplImage image, CvRect r){
         IplImage gray = cvCreateImage(cvGetSize(image), IPL_DEPTH_8U, 1);
         IplImage roi = cvCreateImage(cvGetSize(image), IPL_DEPTH_8U, 1);
         CvRect r1 = new CvRect();
         r1.x(r.x()-10);
         r1.y(r.y()-10);
         r1.width(r.width()+10);
         r1.height(r.height()+10);
         //CvRect r1 = new CvRect(r.x()-10, r.y()-10, r.width()+10, r.height()+10);
         cvCvtColor(image, gray, CV_BGR2GRAY);
         cvSetImageROI(gray, r1);
         cvResize(gray, roi, CV_INTER_LINEAR);
         cvEqualizeHist(roi, roi);
         return roi;
 }
}
