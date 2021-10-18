#pragma once
//#define DEBUG1 // foreground histogram için debugging

#if __cplusplus > 199711L
#define SSTR(X) (std::to_string(X))
#else
#define SSTR( x ) (static_cast< std::ostringstream &>(( std::ostringstream() << std::dec << x ) ).str()) // number to string / itoa()
#endif
#define Center( r ) (Point((r.x + r.width/2),(r.y + r.height/2))) // r rect merkezi

template <typename recT>
recT Rescale(recT bbox, Size oldSize, Size newSize, Size frame ,int frameRatio = frame_ratio)
{
	float w = newSize.width / oldSize.width; // change ratio in bbox size 
	float h = newSize.height / oldSize.height;
	Point center = Center(bbox);

	w = MIN(MIN(w, 2 * center.x - w / frameRatio), 2 * (frame.width - center.x - w / frameRatio)) +1; // 0 boyuta inmemesi için en az 3 ile sýnýrladýk
	h = MIN(MIN(h, 2 * center.y - h / frameRatio), 2 * (frame.height - center.y - h / frameRatio)) +1;
	return recT(center.x - w / 2, center.y - h / 2, w, h);
}


inline Size momentSize(Mat probmap)
{
	Moments mu = moments(probmap);
	float X = mu.m10 / mu.m00;
	float Y = mu.m01 / mu.m00;
	float a = (mu.m20 / mu.m00) - X * X;
	float b = 2 * ((mu.m11 / mu.m00) - X * Y);
	float c = (mu.m02 / mu.m00) - Y * Y;
	float width = sqrt(((a + c) - sqrt(b * b + (a - c) * (a - c))) / 2);
	float height = sqrt(((a + c) + sqrt(b * b + (a - c) * (a - c))) / 2);

	return Size(width, height);
}

template <typename recT>
class scaleBox
{
	public:
		Size distSize, baseSize; // sizes getting from moments (new size / base size)
		Mat back_hist_old = Mat(Size(1, 256), CV_32F, Scalar(0)); // TEST --> eski histogramý tutmak için
		recT bbox; // prev box

		void init(Mat grayFrame, recT bbox);
		recT updateSize(Mat grayFrame, recT bbox);

	protected:
		void foregroundHistProb(Mat in, Size distSize, Mat& hist, Mat& probHist, int value=val);
	private:
		Mat grayROI, probmap; // target pixels probabilities map
};

/*
--------------------- foregroundHistProb -------------------------
	arkaplandaki pixel deðerlerini boxun olasýlýðýndan çýkarýyoruz 
	val: bir deðerin histogramdan çýkarýlmasý için gereken en az pixel sayýsý  
*/
template <typename recT>
void scaleBox<recT>::foregroundHistProb(Mat in, Size distSize, Mat& hist, Mat& probHist, int value) 
{
	Mat mask = Mat(in.size(), CV_8U, Scalar(0)); // foreground mask
	mask(Rect(Point(distSize.width, distSize.height), Point(in.cols - distSize.width, in.rows - distSize.height))).setTo(Scalar::all(255)); // set all pixel in mask

	Mat back_hist, fore_hist;
	float range[] = { 0,255 }; // one channel gray image parameters
	const float* ranges[] = { range };
	int hist_size[] = { 256 };//bins = 256
	int channels[] = { 0 };

	calcHist(&in, 1, channels, 255 - mask, back_hist, 1, hist_size, ranges, true, false); //arkaplan histogramý
	calcHist(&in, 1, channels, mask, fore_hist, 1, hist_size, ranges, true, false); // önplan histogramý


	normalize(back_hist, back_hist, 0, 255, NORM_MINMAX, -1, Mat()); //normalize histograms
	normalize(fore_hist, fore_hist, 0, 255, NORM_MINMAX, -1, Mat());
	threshold(back_hist, back_hist, value, 255, THRESH_BINARY); // val == arkaplandan bir deðerin alýnma sýnýrý default 4 - 4 pixel veya az varsa alýnabilir demek
	hist = back_hist;// | hist; //--> TEST old histogramý da hesaba katarak daha iyi sonuc elde edebilir miyiz ?
	fore_hist = fore_hist & (255 - hist); // XOR(or substract) foreground with background
	normalize(fore_hist, fore_hist, 0, 255, NORM_MINMAX, -1, Mat()); // deðerleri normalize ediyoruz --> hatalý çýktý verebiliyor(teoride gerekli deðil)  

#ifdef DEBUG1 // histogramý görmek için çizim --> klasik opencv histogram çýktýsýnýn aynýsý
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
	calcBackProject(&in, 1, channels, fore_hist, out, ranges); // sonuçtaki olasýlýk haritamýz
	threshold(out, probHist, 1, 255, THRESH_BINARY); // çýkýþ olasýlýklarýný 0 ve 1 olarak almak için thresh --> normal denenebilir fakat bu daha etkili sonuç verdi
	imshow("prob Demo", probHist); // olasýk haritasý gösterimi
}


template <typename recT>
void scaleBox<recT>::init(Mat grayFrame, recT bbox)
{
	grayROI = grayFrame(bbox); // ROI the gray !!!
	this->distSize = Size(grayROI.cols / frame_ratio, grayROI.rows / frame_ratio); // outer box's extra size
	foregroundHistProb(grayROI, this->distSize, back_hist_old, probmap, val); // calc prob map that represents target shape

	this->baseSize = momentSize(probmap); // get size from moments with using prob's map
	this->bbox = bbox;
	//cout << "base values[width-height] =  " << baseSize << endl;
}


template <typename recT>
recT scaleBox<recT>::updateSize(Mat grayFrame, recT bbox)
{
	distSize = Size(this->bbox.width / frame_ratio, this->bbox.height / frame_ratio);
	this->bbox= Rect(Center(bbox) - Point((this->bbox.width + distSize.width) / 2, (this->bbox.height + distSize.height) / 2),
			 Center(bbox) + Point((this->bbox.width + distSize.width) / 2, (this->bbox.height + distSize.height) / 2)); // get new object square position to calc size
	grayROI = grayFrame(this->bbox); // ROI with new position but old size 

	foregroundHistProb(grayROI, distSize, back_hist_old, probmap);

	Size nSize = momentSize(probmap); 	// moment calc
	return Rescale<recT>(this->bbox, baseSize, nSize, grayFrame.size(), frame_ratio); // rescale with moment
}
