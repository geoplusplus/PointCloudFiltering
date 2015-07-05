#include "OpenniFilter.h"

OpenniFilter::OpenniFilter() : viewer("PCL OpenNI Viewer") 
{
    // initialize display
    namedWindow(windowTracker, CV_WINDOW_AUTOSIZE);
    namedWindow(windowFilter, CV_WINDOW_AUTOSIZE);
    setMouseCallback(windowTracker, mouseCallback, this);

    // initialize filter windows
    namedWindow(windowParam, CV_WINDOW_AUTOSIZE);
    // the hue should be in accordence with what user has chosen
    cvCreateTrackbar("Hue Deviation", windowParam.c_str(), &iHueDev, 255);
    cvCreateTrackbar("LowS",          windowParam.c_str(), &iLowS,  255); //Saturation (0 - 255)
    cvCreateTrackbar("HighS",         windowParam.c_str(), &iHighS, 255);
    cvCreateTrackbar("LowV",          windowParam.c_str(), &iLowV,  255); //Value (0 - 255)
    cvCreateTrackbar("HighV",         windowParam.c_str(), &iHighV, 255);

}

OpenniFilter::~OpenniFilter()
{
    destroyWindow(windowTracker);
    destroyWindow(windowFilter);
    destroyWindow(windowParam);
}


void OpenniFilter::cloud_cb_ (const pcl::PointCloud<pcl::PointXYZRGBA>::ConstPtr &cloud)
{
    if (!viewer.wasStopped())
    {
        if (cloud->isOrganized())
        {
            // initialize all the Mats to store intermediate steps
            int cloudHeight = cloud->height;
            int cloudWidth = cloud->width;
            rgbFrame = Mat(cloudHeight, cloudWidth, CV_8UC3);
            drawing = Mat(cloudHeight, cloudWidth, CV_8UC3, NULL);
            grayFrame = Mat(cloudHeight, cloudWidth, CV_8UC1, NULL);
            hsvFrame = Mat(cloudHeight, cloudWidth, CV_8UC3, NULL);
            contourMask = Mat(cloudHeight, cloudWidth, CV_8UC1, NULL);

            if (!cloud->empty())
            {
                for (int h = 0; h < rgbFrame.rows; h ++)
                {
                    for (int w = 0; w < rgbFrame.cols; w++)
                    {
                        pcl::PointXYZRGBA point = cloud->at(w, cloudHeight-h-1);
                        Eigen::Vector3i rgb = point.getRGBVector3i();
                        rgbFrame.at<Vec3b>(h,w)[0] = rgb[2];
                        rgbFrame.at<Vec3b>(h,w)[1] = rgb[1];
                        rgbFrame.at<Vec3b>(h,w)[2] = rgb[0];
                    }
                }

                // do the filtering 
                int xPos = 0;
                int yPos = 0;
                mtx.lock();
                xPos = mouse_x;
                yPos = mouse_y;
                mtx.unlock();

                // color filtering based on what is chosen by users
                cvtColor(rgbFrame, hsvFrame, CV_RGB2HSV);
                Vec3b pixel = hsvFrame.at<Vec3b>(xPos,yPos);

                int hueLow = pixel[0] < iHueDev ? pixel[0] : pixel[0] - iHueDev;
                int hueHigh = pixel[0] > 255 - iHueDev ? pixel[0] : pixel[0] + iHueDev;
                // inRange(hsvFrame, Scalar(hueLow, pixel[1]-20, pixel[2]-20), Scalar(hueHigh, pixel[1]+20, pixel[2]+20), grayFrame);
                inRange(hsvFrame, Scalar(hueLow, iLowS, iLowV), Scalar(hueHigh, iHighS, iHighV), grayFrame);

                // removes small objects from the foreground by morphological opening
                erode(grayFrame, grayFrame, getStructuringElement(MORPH_ELLIPSE, Size(5,5)));
                dilate(grayFrame, grayFrame, getStructuringElement(MORPH_ELLIPSE, Size(5,5)));

                // morphological closing (removes small holes from the foreground)
                dilate(grayFrame, grayFrame, getStructuringElement(MORPH_ELLIPSE, Size(5,5)));
                erode(grayFrame, grayFrame, getStructuringElement(MORPH_ELLIPSE, Size(5,5)));

                // gets contour from the grayFrame and keeps the largest contour
                Mat cannyOutput;
                vector<vector<Point> > contours;
                vector<Vec4i> hierarchy;
                int thresh = 100;
                Canny(grayFrame, cannyOutput, thresh, thresh * 2, 3);
                findContours(cannyOutput, contours, hierarchy, CV_RETR_TREE, CV_CHAIN_APPROX_SIMPLE, Point(0, 0));
                int largestContourArea, largestContourIndex = 0;
                int defaultContourArea = 1000; // 1000 seems to work find in most cases... cannot prove this
                vector<vector<Point> > newContours;
                for (int i = 0; i < contours.size(); i++)
                {
                    double area = contourArea(contours[i], false);
                    if (area > defaultContourArea)
                        newContours.push_back(contours[i]);
                }

                // draws the largest contour: 
                drawing = Mat::zeros(cannyOutput.size(), CV_8UC3);
                for (int i = 0; i < newContours.size(); i++)
                    drawContours(drawing, newContours, i, Scalar(255, 255, 255), CV_FILLED, 8, hierarchy, 0, Point());

                // gets the filter by setting everything within the contour to be 1. 
                inRange(drawing, Scalar(1, 1, 1), Scalar(255, 255, 255), contourMask);

                // filters the point cloud based on contourMask
                // again go through the point cloud and filter out unnecessary points
                pcl::PointCloud<pcl::PointXYZRGBA>::Ptr resultCloud (new pcl::PointCloud<pcl::PointXYZRGBA>);
                pcl::PointXYZRGBA newPoint;
                for (int h = 0; h < contourMask.rows; h ++)
                {
                    for (int w = 0; w < contourMask.cols; w++)
                    {
                        if (contourMask.at<uchar>(h,w) > 0)
                        {
                            newPoint = cloud->at(w,h);
                            resultCloud->push_back(newPoint);
                        }
                    }
                }

                if (xPos == 0 && yPos == 0)
                    viewer.showCloud(cloud);
                else
                    viewer.showCloud(resultCloud);
                
                imshow("tracker", rgbFrame);
                imshow("filtered result", contourMask);
                char key = waitKey(1);
                if (key == 27) 
                {
                    interface->stop();
                    return;
                }
            }
            else
                cout << "Warning: Point Cloud is empty" << endl;
        }
        else
            cout << "Warning: Point Cloud is not organized" << endl;
    }
}

void OpenniFilter::run ()
{
    interface = new pcl::OpenNIGrabber();
    boost::function<void (const pcl::PointCloud<pcl::PointXYZRGBA>::ConstPtr&)> f =
        boost::bind (&OpenniFilter::cloud_cb_, this, _1);

    interface->registerCallback(f);
    interface->start();
    while (!viewer.wasStopped())
    {
        boost::this_thread::sleep (boost::posix_time::seconds (1));
    }

    interface->stop();
}

void OpenniFilter::mouseCallback(int event, int x, int y, int flags, void *param)
{
    OpenniFilter *self = static_cast<OpenniFilter *>(param);
    self->doMouseCallback(event, x, y, flags, NULL);
}

void OpenniFilter::doMouseCallback(int event, int x, int y, int flags, void* userData)
{
    if (event == EVENT_RBUTTONDOWN)
    {
        mtx.lock();
        mouse_x = x;
        mouse_y = y;
        mtx.unlock();
        cout << "mouse_x" << x << "mouse_y" << y << endl; 
    }
}
