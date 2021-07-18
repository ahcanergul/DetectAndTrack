#pragma once
//#define DEBUG1 // foreground histogram i�in debugging

#define SSTR( x ) (static_cast< std::ostringstream >(( std::ostringstream() << std::dec << x ) ).str()) // number to string / itoa()
#define Center( r ) (Point((r.x + r.width/2),(r.y + r.height/2))) // r rect merkezi

Rect Rescale(Rect2d bbox, Size oldSize, Size newSize)
{
	float w = newSize.width / oldSize.width; // change ratio in bbox size 
	float h = newSize.height / oldSize.height;
	Point center = Center(bbox);

	w = MAX(bbox.width * w, 3); // 0 boyuta inmemesi i�in en az 3 ile s�n�rlad�k
	h = MAX(bbox.height * h, 3);
	return Rect(center.x - w / 2, center.y - h / 2, w, h);
}

/*
--------------------- foregroundHistProb -------------------------
	arkaplandaki pixel de�erlerini boxun olas�l���ndan ��kar�yoruz 
	val: bir de�erin histogramdan ��kar�lmas� i�in gereken en az pixel say�s�  
*/
void foregroundHistProb(Mat in, Size distSize, Mat& hist, Mat& probHist, int val=4) 
{
	Mat mask = Mat(in.size(), CV_8U, Scalar(0)); // foreground mask
	mask(Rect(Point(distSize.width, distSize.height), Point(in.cols - distSize.width, in.rows - distSize.height))).setTo(Scalar::all(255)); // set all pixel in mask

	Mat back_hist, fore_hist;
	float range[] = { 0,255 }; // one channel gray image parameters
	const float* ranges[] = { range };
	int hist_size[] = { 256 };//bins = 256
	int channels[] = { 0 };

	calcHist(&in, 1, channels, 255 - mask, back_hist, 1, hist_size, ranges, true, false); //arkaplan histogram�
	calcHist(&in, 1, channels, mask, fore_hist, 1, hist_size, ranges, true, false); // �nplan histogram�


	normalize(back_hist, back_hist, 0, 255, NORM_MINMAX, -1, Mat()); //normalize histograms
	normalize(fore_hist, fore_hist, 0, 255, NORM_MINMAX, -1, Mat());
	threshold(back_hist, back_hist, val, 255, THRESH_BINARY); // val == arkaplandan bir de�erin al�nma s�n�r� default 4 - 4 pixel veya az varsa al�nabilir demek
	hist = back_hist;// | hist; //--> TEST old histogram� da hesaba katarak daha iyi sonuc elde edebilir miyiz ?
	fore_hist = fore_hist & (255 - hist); // XOR(or substract) foreground with background
	normalize(fore_hist, fore_hist, 0, 255, NORM_MINMAX, -1, Mat()); // de�erleri normalize ediyoruz --> hatal� ��kt� verebiliyor(teoride gerekli de�il)  

#ifdef DEBUG1 // histogram� g�rmek i�in �izim --> klasik opencv histogram ��kt�s�n�n ayn�s�
	Mat histImage = Mat::zeros(600, 600, CV_8UC3);
	int hist_w = 512, hist_h = 400;
	int bin_w = cvRound((double)hist_w / 256);
	for (int i = 1; i < 256; i++)
	{
		line(histImage, Point(bin_w * (i - 1), hist_h - cvRound(hist.at<float>(i - 1))),
			Point(bin_w * (i), hist_h - cvRound(hist.at<float>(i))),
			Scalar(255, 0, 0), 2, 8, 0);
		line(histImage, Point(bin_w * (i - 1), hist_h - cvRound(fore_hist.at<float>(i - 1))),
			Point(bin_w * (i), hist_h - cvRound(fore_hist.at<float>(i))),
			Scalar(0, 255, 0), 2, 8, 0);
	}
	imshow("calcHist Demo", histImage); // histogram to look up table
#endif

	Mat out;
	calcBackProject(&in, 1, channels, fore_hist, out, ranges); // sonu�taki olas�l�k haritam�z
	threshold(out, probHist, 1, 255, THRESH_BINARY); // ��k�� olas�l�klar�n� 0 ve 1 olarak almak i�in thresh --> normal denenebilir fakat bu daha etkili sonu� verdi
	imshow("prob Demo", probHist); // olas�k haritas� g�sterimi
}

Size momentSize(Mat probmap)
{
	Moments mu = moments(probmap);
	float X = mu.m10 / mu.m00;
	float Y = mu.m01 / mu.m00;
	float a = (mu.m20 / mu.m00) - X * X;
	float b = 2 * ((mu.m11 / mu.m00) - X * Y);
	float c = (mu.m02 / mu.m00) - Y * Y;
	float width = sqrt(((a + c) - sqrt(b * b + (a - c) * (a - c))) / 2);
	float height = sqrt(((a + c) + sqrt(b * b + (a - c) * (a - c))) / 2);

	return Size(width,height);
}