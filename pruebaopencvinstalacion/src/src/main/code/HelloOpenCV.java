package src.main.code;

import org.opencv.core.Core;

public class HelloOpenCV {

	public static void main(String[] args) {
	    System.out.println("Hello, OpenCV");

	    // Load the native library.
	    System.loadLibrary(Core.NATIVE_LIBRARY_NAME);
	    new Hello().run();
	  }
	
}
