//
//  ViewController.m
//  FFT_Test
//
//  Created by Simon Lucey on 9/3/16.
//  Copyright © 2016 CMU_16623. All rights reserved.
//

#import "ViewController.h"
#import "Accelerate/Accelerate.h"

#ifdef __cplusplus
#include <opencv2/opencv.hpp> // Includes the opencv library
#include <stdlib.h> // Include the standard library
#include "armadillo" // Includes the armadillo library
#endif

@interface ViewController ()

@end

@implementation ViewController

- (void)viewDidLoad {
    [super viewDidLoad];
    // Do any additional setup after loading the view, typically from a nib.
    
    // Simple comparison between Armadillo and OpenCV
    using namespace std;
    
    int trials = 100; // Number of iterations
    int D = 128; // Dimensionality of the square 2D matrix
                 // remember to keep powers of 2 since we are
                 // using the radix 2 FFT.
    
    // Initialize X
    arma::fmat X; X.randn(D,D); // Initialize random x matrix
    arma::cx_fmat Xf; Xf.zeros(D,D); // Allocate space for FFT

    cv::Mat X_cv = Arma2Cv(X); // Generate the OpenCV version
    cv::Mat Xf_cv; // Allocate space
    
    // Intialize the clock
    arma::wall_clock timer;

    // Apply FFT using OpenCV
    timer.tic();
    for(int i=0; i<trials; i++) {
        cv::dft(X_cv, Xf_cv);
    }
    double cv_n_secs = timer.toc();
    cout << "OpenCV took " << cv_n_secs << " seconds." << endl;

    // Apply the fft using armadillo
    timer.tic();
    for(int i=0; i<trials; i++) {
        Xf = fft2(X);
    }
    double arma_n_secs = timer.toc();
    cout << "Armadillo took " << arma_n_secs << " seconds." << endl;
    cout << "Armadillo is " << cv_n_secs/arma_n_secs << " times faster than OpenCV!!!" << endl;
    
    // Apply the fft using vDSP
    timer.tic();
    vDSP_FFT2 vfft(X); // Initialize for vDSP_FFT
    for(int i=0; i<trials; i++) {
        vfft.apply(X);
    }
    double vdsp_n_secs = timer.toc();
    cout << "vDSP took " << vdsp_n_secs << " seconds." << endl;
    cout << "vDSP is " << cv_n_secs/vdsp_n_secs << " times faster than OpenCV!!!" << endl;
    
    // Final bit of example code for writing a file out
    // in assignment 1 we want you to change this code snippet
    // so you can write data out from your App.
    //
    //
    NSArray *paths = NSSearchPathForDirectoriesInDomains
    (NSDocumentDirectory, NSUserDomainMask, YES);
    NSString *documentsDirectory = [paths objectAtIndex:0];
    
    //make a file name to write the data to using the documents directory:
    NSString *fileName = [NSString stringWithFormat:@"%@/example.arma",
                          documentsDirectory];
    
    // Get the full path and name of the file
    const char *fname = [fileName UTF8String];
    
    // Randomly initialize a variable and then save it
    arma::fmat tmp(3,3); tmp.randn();
    tmp.save(fname);
    cout << "writing out example.arma to the Documents directory!!!" << endl;
        
}

- (void)didReceiveMemoryWarning {
    [super didReceiveMemoryWarning];
    // Dispose of any resources that can be recreated.
}

//==============================================================================
// Quick function to convert to Armadillo matrix header
arma::fmat Cv2Arma(cv::Mat &cvX)
{
    arma::fmat X(cvX.ptr<float>(0), cvX.cols, cvX.rows, false); // This is the transpose of the OpenCV X_
    return X; // Return the new matrix (no new memory allocated)
}
//==============================================================================
// Quick function to convert to OpenCV (floating point) matrix header
cv::Mat Arma2Cv(arma::fmat &X)
{
    cv::Mat cvX = cv::Mat(X.n_cols, X.n_rows,CV_32F, X.memptr()).clone();
    return cvX; // Return the new matrix (new memory allocated)
}

//-------------------------------------------------------------------
// Simple class used to employ vDSP's efficient FFTs within Armadillo
//
// Written by Simon Lucey 2016
//-------------------------------------------------------------------
class vDSP_FFT2{
    public:
    int wlog2_; // Log 2 width of matrix (rounded to nearest integer)
    int hlog2_; // Log 2 height of matrix (rounded to nearest integer)
    int total_size_; // Total size of the signal
    FFTSetup setup_; // Setup stuff for fft (part of vDSP)
    float *ptr_xf_; // pointer to output of FFT on x
    DSPSplitComplex xf_; // Special vDSP struct for complex arrays
    
    // Class functions
    
    // Constructor based on size of x
    vDSP_FFT2(arma::fmat &x){
        
        // Get the width and height in power of 2
        wlog2_ = ceil(log2(x.n_cols));
        hlog2_ = ceil(log2(x.n_rows));
        
        // Setup FFT for Radix2 FFT
        int nlog2 = std::max(wlog2_, hlog2_); // Get the max value
        setup_ = vDSP_create_fftsetup(nlog2, FFT_RADIX2);
        
        // Get the total size
        total_size_ = pow(2, wlog2_)*pow(2,hlog2_);
 
        // Allocate space for result of the FFT
        ptr_xf_ = (float *) malloc(total_size_*sizeof(float));
        
        // Special struct that vDSP uses for FFTs
        xf_ = DSPSplitComplex{ptr_xf_, ptr_xf_ + total_size_/2};
    };
    
    // Destructor
    ~vDSP_FFT2(){
        // Destroy everything setup for FFT
        vDSP_destroy_fftsetup(setup_);
        
        // Free up the memory
        free(ptr_xf_);
    };
    
    // Member function to apply the 2D FFT
    DSPSplitComplex *apply(arma::fmat &x) {
        
        // Split the signal into real and imaginary components
        vDSP_ctoz((DSPComplex *) x.memptr(), 2, &xf_, 1, total_size_/2);
        
        // Apply the FFT to the signal
        vDSP_fft2d_zrip(setup_, &xf_, 1, 0, wlog2_, hlog2_, FFT_FORWARD);
        
        // Return the pointer
        return(&xf_);
    };
    
    // Display the contents of the fft
    void display() {
        int w = pow(2,wlog2_); int h = pow(2,hlog2_);
        for(int i=0; i<(w*h)/2; i++) {
            std::cout << xf_.realp[i]/2 << " j*" << xf_.imagp[i]/2 << std::endl;
        }
    }
};
@end
