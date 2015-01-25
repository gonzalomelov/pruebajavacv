package src.main.code;

import static org.bytedeco.javacpp.helper.opencv_objdetect.cvHaarDetectObjects;

import static org.bytedeco.javacpp.opencv_objdetect.CV_HAAR_DO_CANNY_PRUNING;
import static org.bytedeco.javacpp.opencv_core.IPL_DEPTH_8U;
import static org.bytedeco.javacpp.opencv_core.cvCreateImage;
import static org.bytedeco.javacpp.opencv_core.cvGetSeqElem;
import static org.bytedeco.javacpp.opencv_core.cvGetSize;
import static org.bytedeco.javacpp.opencv_core.cvLoad;
import static org.bytedeco.javacpp.opencv_core.cvSetImageROI;
import static org.bytedeco.javacpp.opencv_highgui.cvSaveImage;
import static org.bytedeco.javacpp.opencv_imgproc.CV_BGR2GRAY;
import static org.bytedeco.javacpp.opencv_imgproc.CV_INTER_LINEAR;
import static org.bytedeco.javacpp.opencv_imgproc.cvCvtColor;
import static org.bytedeco.javacpp.opencv_imgproc.cvEqualizeHist;
import static org.bytedeco.javacpp.opencv_imgproc.cvResize;

import org.bytedeco.javacv.FrameGrabber;
import org.bytedeco.javacpp.Loader;
import org.bytedeco.javacpp.opencv_objdetect;
import org.bytedeco.javacpp.opencv_core.*;
import org.bytedeco.javacpp.opencv_objdetect.CvHaarClassifierCascade;
import org.bytedeco.javacv.OpenCVFrameGrabber;

public class VideoProcessing {

	private static final String PATH = "resources/";
	private static final String CASCADE_FILE = PATH + "haarcascade_frontalface_alt.xml";
	final static CvHaarClassifierCascade cascade = new CvHaarClassifierCascade(cvLoad(CASCADE_FILE));
	
	public static void main(String[] args) {
		
		String resultPath = PATH+"output/";
		
		//Declare FrameGrabber to import video from "video.mp4"
	     FrameGrabber grabber = new OpenCVFrameGrabber(PATH + "/videos/VID_20150125_162330.mp4");  

	     
	     try {      
	         
	         //Start grabber to capture video
	         grabber.start();      
	          
	         //Declare img as IplImage
	         IplImage img;
	         
	         int i = 0;
	         
	         while (i < 10000) {
	           
	          //inser grabed video fram to IplImage img
	          img = grabber.grab();
	          System.out.println(i);
	          if((i % 20) == 0){
	        	  CvSeq faces = detectFace(img);
	        	  
	        	  if (!faces.isNull()){
	        		  System.out.println("NO ES NULL");
	        		  if(faces.total() > 0){
	        			  CvRect r = new CvRect(cvGetSeqElem(faces,0));
	        			  
				          img=preprocessImage(img, r);
				          
				          IplImage newFace = IplImage.create(92,112, IPL_DEPTH_8U, 1);
				          cvResize(img , newFace);
				          cvSaveImage(resultPath+"foto"+i+".jpg", newFace);
	        		  }else{
	        			  System.out.println("no hay cara");
	        		  }
	        		  
	        	  }else{
	        		  System.out.println("NULL");
	        	  }
	          }
	          
	          i++;
	          
	          }
	         System.out.println("fin");
	         }
	        catch (Exception e) {      
	        }
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
	
	
	 protected static CvSeq detectFace(IplImage originalImage) {
         CvSeq faces = null;
         Loader.load(opencv_objdetect.class);
         try {
//                 IplImage grayImage = IplImage.create(originalImage.width(), originalImage.height(), IPL_DEPTH_8U, 1);
//                 cvCvtColor(originalImage, grayImage, CV_BGR2GRAY);
                 CvMemStorage storage = CvMemStorage.create();
                 faces = cvHaarDetectObjects(
                		 originalImage,
        				cascade,
        				storage,
        				1.5,
        				3,
        				CV_HAAR_DO_CANNY_PRUNING);//cvHaarDetectObjects(grayImage, cascade, storage, 1.1, 1, 0);

         } catch (Exception e) {
                 e.printStackTrace();
         }
         return faces;
 }

}
