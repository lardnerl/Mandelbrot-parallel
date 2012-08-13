/*
 * CS3014 Mandelbrot Project
 * 
 * Using techniques we've covered in class, accelerate the rendering of
 * the M set.
 * 
 * Hints
 * 
 * 1) Vectorize
 * 2) Use threads
 * 3) Load Balance
 * 4) Profile and Optimise
 * 
 * Potential FAQ.
 * 
 * Q1) Why when I zoom in far while palying with the code, why does the image begin to render all blocky?
 * A1) In order to render at increasing depths we must use increasingly higher precision floats
 * 	   We quickly run out of precision with 32 bits floats. Change all floats to doubles if you want
 * 	   dive deeper. Eventually you will however run out of precision again and need to integrate an
 * 	   infinite precision math library or use other techniques.
 * 
 * Q2) Why do some frames render much faster than others?
 * A2) Frames with a lot of black, i.e, frames showing a lot of set M, show pixels that ran until the 
 *     maximum number of iterations was reached before bailout. This means more CPU time was consumed
 */



#include <iostream>
#include <cmath>
#include <xmmintrin.h>

#define TIMING
#ifdef TIMING
#include <sys/time.h>
#endif



#include "Screen.h"


/*
 * You can't change these values to accelerate the rendering.
 * Feel free to play with them to render different images though.
 */
const int 	MAX_ITS = 1000;			//Max Iterations before we assume the point will not escape
const int 	HXRES = 700; 			// horizontal resolution	
const int 	HYRES = 700;			// vertical resolution
const int 	MAX_DEPTH = 40;		// max depth of zoom
const float ZOOM_FACTOR = 1.02;		// zoom between each frame

/* Change these to zoom into different parts of the image */
const float PX = -0.702295281061;	// Centre point we'll zoom on - Real component
const float PY = +0.350220783400;	// Imaginary component


/*
 * The palette. Modifying this can produce some really interesting renders.
 * The colours are arranged R1,G1,B1, R2, G2, B2, R3.... etc.
 * RGB values are 0 to 255 with 0 being darkest and 255 brightest
 * 0,0,0 is black
 * 255,255,255 is white
 * 255,0,0 is bright red
 */
unsigned char pal[]={
	255,180,4,
	240,156,4,
	220,124,4,
	156,71,4,
	72,20,4,
	251,180,4,
	180,74,4,
	180,70,4,
	164,91,4,
	100,28,4,
	191,82,4,
	47,5,4,
	138,39,4,
	81,27,4,
	192,89,4,
	61,27,4,
	216,148,4,
	71,14,4,
	142,48,4,
	196,102,4,
	58,9,4,
	132,45,4,
	95,15,4,
	92,21,4,
	166,59,4,
	244,178,4,
	194,121,4,
	120,41,4,
	53,14,4,
	80,15,4,
	23,3,4,
	249,204,4,
	97,25,4,
	124,30,4,
	151,57,4,
	104,36,4,
	239,171,4,
	131,57,4,
	111,23,4,
	4,2,4};
const int PAL_SIZE = 40;  //Number of entries in the palette 

float getMax(__m128 r){
	float * temp = (float *)malloc(sizeof(float) * 4);
	_mm_store_ps(temp,r);

	float result = temp[0];
	int i;
	for(i=1; i<4; i++) if(result < temp[i]) result = temp[i];
	return result;
}

bool allZeros(__m128 r){
	float * temp = (float *)malloc(sizeof(float) * 4);
	_mm_store_ps(temp,r);
	int i;
	for(i=0; i<4; i++) if(temp[i] != 0) return false;
	return true;
}

// Not passing by reference so am free to alter???
// Update iterations here
void updateIterations(__m128 x, __m128 y, __m128& result){
	x = _mm_mul_ps(x,x);
	y = _mm_mul_ps(y,y);
	x = _mm_add_ps(x,y);
	__m128 four = _mm_set1_ps(4.0);
    float temp[] = {0,0,0,0};
	
	//update iterations based on the check	
	int check = _mm_movemask_ps(_mm_cmplt_ps(x, four));
	if(check&&0x1 == 0x1) temp[3] = 1;
	if(check&&0x2 == 0x2) temp[2] = 1;
	if(check&&0x4 == 0x4) temp[1] = 1;
	if(check&&0x8 == 0x8) temp[0] = 1;
	
	__m128 t = _mm_setr_ps(temp[0], temp[1], temp[2], temp[3]);
	
	result = _mm_add_ps(result, t);
}

/* 
 * Return a four bit integer of the points that escape
 * iterations is set to the number of iterations until escape.
 */
__m128 member_iterations(__m128 cx_s, __m128 cy_s){
	__m128 x = _mm_set1_ps(0.0);
	__m128 y = _mm_set1_ps(0.0);
	__m128 two = _mm_set1_ps(2.0);
	__m128 result = _mm_set1_ps(1.0);
	
	do{
		__m128 xtemp = _mm_add_ps(_mm_sub_ps(_mm_mul_ps(x,x),_mm_mul_ps(y,y)),cx_s); // x*x - y*y + cx
		y = _mm_add_ps(_mm_mul_ps(two,_mm_mul_ps(y,x)),cy_s); //2*x*y + cy
		x = xtemp;
		updateIterations(x, y, result);
	}while( getMax(result) < MAX_ITS && !allZeros(result));
	// Make sure i return the proper crap
	return result;
	
}

int main()
{	
	int hx, hy;

	float m=1.0; /* initial  magnification		*/

	/* Create a screen to render to */
	Screen *screen;
	screen = new Screen(HXRES, HYRES);

	int depth=0;

#ifdef TIMING
  struct timeval start_time;
  struct timeval stop_time;
  long long total_time = 0;
#endif

	/*
		Define the constant vectors
	*/
	
	// Multipling by 1/HXRES is quicker than dividing by HXRES
	__m128 RCP_HXRES_s = _mm_set1_ps(1.0 / HXRES);
	__m128 RCP_HYRES_s = _mm_set1_ps(1.0 / HYRES);
	
	// Adding minus 0.5 is quicker than subtracting a half
	__m128 minus_a_half = _mm_set1_ps(-0.5);
	
	__m128 PX_s = _mm_set1_ps(PX);
	__m128 PY_s = _mm_set1_ps(PY);
	while (depth < MAX_DEPTH) {
#ifdef TIMING
	        /* record starting time */
	        gettimeofday(&start_time, NULL);
#endif
		for (hy=0; hy<HYRES; hy++) {
		
			__m128 four_div_m = _mm_set1_ps(4.0 / m);
		
			float cy = (((float)hy/(float)HXRES) -0.5 + (PX/(4.0/m)))*(4.0f/m);
			__m128 cy_s = _mm_set1_ps(cy);
			
			for (hx=0; (hx+4)<HXRES; hx+=4) {
				
				__m128 hx_s = _mm_setr_ps((float)hx, (float)(hx+1), (float)(hx+2), (float)(hx+3));
				__m128 cx_s = _mm_mul_ps(hx_s,RCP_HXRES_s); // hx * 1/HXRES
				cx_s = _mm_add_ps(cx_s,minus_a_half); // (hx * 1/HXRES) + (-0.5)
				
				// Found multipling by the reciproal to be quicker here
				cx_s = _mm_mul_ps(_mm_add_ps(cx_s, _mm_mul_ps(PX_s, _mm_rcp_ps(four_div_m))),four_div_m); // (((hx * 1/HXRES) + (-0.5)) + (PX * (1 / (4 / m)))) * (4/m)
				
				// Store points into arrays.  Made sure the memory is aligned
				float * cx_arr = (float *)malloc(sizeof(float) * 4);
				_mm_store_ps(cx_arr,cx_s);
				
				float * cy_arr = (float *)malloc(sizeof(float) * 4);
				_mm_store_ps(cy_arr,cy_s);
				
				__m128 iterations = member_iterations(cx_s, cy_s);
				float * iterations_arr = (float *)malloc(sizeof(float) * 4);
				_mm_store_ps(iterations_arr,iterations);
				
				int j, k;
				// Check all possible combinations of (cx, cy) - 16	

				for(k=0; k<4; k++){
					if (iterations_arr[k] > 0) {
						/* Point is not a member, colour based on number of iterations before escape */
						int i=(((int)iterations_arr[k])%40) - 1;
						int b = i*3;
						screen->putpixel((hx+k), (hy), pal[b], pal[b+1], pal[b+2]);
					} else {
						/* Point is a member, colour it black */
						screen->putpixel((hx+k), (hy), 0, 0, 0);
					}
				}
				
			}
		}
#ifdef TIMING
		gettimeofday(&stop_time, NULL);
		total_time += (stop_time.tv_sec - start_time.tv_sec) * 1000000L + (stop_time.tv_usec - start_time.tv_usec);
#endif
		/* Show the rendered image on the screen */
		screen->flip();
		std::cout << "Render done " << depth++ << " " << m << std::endl;

		/* Zoom in */
		m *= ZOOM_FACTOR;
	}
	
	sleep(10);
#ifdef TIMING
	std::cout << "Total executing time " << total_time << " microseconds\n";
#endif
	std::cout << "Clean Exit"<< std::endl;

}
