#include <pcl/io/openni_grabber.h>
#include <pcl/visualization/cloud_viewer.h>
#include <pcl/point_types.h>
#include <opencv2/core/core.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <mutex>

using namespace std;
using namespace cv;

class SimpleOpenNIViewer
{
public:
    SimpleOpenNIViewer () : viewer ("PCL OpenNI Viewer") 
    {
        cv::namedWindow("tracker", 1);
        cv::namedWindow("filtered result", 1);
        cv::setMouseCallback("tracker", mouseCallback, this);
    }

    void cloud_cb_ (const pcl::PointCloud<pcl::PointXYZRGBA>::ConstPtr &cloud)
    {
        if (!viewer.wasStopped())
        {
            if (cloud->isOrganized())
            {
                // initialize all the Mats to store intermediate steps
                rgbFrame = cv::Mat(cloud->height, cloud->width, CV_8UC3);
                drawing = cv::Mat(cloud->height, cloud->width, CV_8UC3, NULL);
                grayFrame = cv::Mat(cloud->height, cloud->width, CV_8UC1, NULL);
                hsvFrame = cv::Mat(cloud->height, cloud->width, CV_8UC3, NULL);
                contourMask = cv::Mat(cloud->height, cloud->width, CV_8UC1, NULL);

                if (!cloud->empty())
                {
                    for (int h = 0; h < rgbFrame.rows; h ++)
                    {
                        for (int w = 0; w < rgbFrame.cols; w++)
                        {
                            pcl::PointXYZRGBA point = cloud->at(w, h);
                            Eigen::Vector3i rgb = point.getRGBVector3i();
                            rgbFrame.at<cv::Vec3b>(h,w)[0] = rgb[2];
                            rgbFrame.at<cv::Vec3b>(h,w)[1] = rgb[1];
                            rgbFrame.at<cv::Vec3b>(h,w)[2] = rgb[0];
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
                    cv::Vec3b pixel = hsvFrame.at<cv::Vec3b>(xPos,yPos);
                    int hueLow = pixel[0] < 10 ? pixel[0] : pixel[0] - 10;
                    int hueHigh = pixel[0] > 245 ? pixel[0] : pixel[0] + 10;
                    inRange(hsvFrame, Scalar(hueLow, pixel[1]-20, pixel[2]-20), Scalar(hueHigh, pixel[1]+20, pixel[2]+20), grayFrame);

                    // removes small objects from the foreground by morphological opening
                    cv::erode(grayFrame, grayFrame, cv::getStructuringElement(cv::MORPH_ELLIPSE, cv::Size(5,5)));
                    cv::dilate(grayFrame, grayFrame, cv::getStructuringElement(cv::MORPH_ELLIPSE, cv::Size(5,5)));

                    // morphological closing (removes small holes from the foreground)
                    cv::dilate(grayFrame, grayFrame, cv::getStructuringElement(cv::MORPH_ELLIPSE, cv::Size(5,5)));
                    cv::erode(grayFrame, grayFrame, cv::getStructuringElement(cv::MORPH_ELLIPSE, cv::Size(5,5)));

                    // gets contour from the grayFrame and keeps the largest contour
                    Mat cannyOutput;
                    vector<vector<Point> > contours;
                    vector<Vec4i> hierarchy;
                    int thresh = 100;
                    Canny(grayFrame, cannyOutput, thresh, thresh*2, 3);
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

                    // contourMask.at<uchar>(1, 2);

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

                    viewer.showCloud (resultCloud);
                    imshow("tracker", rgbFrame);
                    imshow("filtered result", contourMask);
                    waitKey(1);
                }
                else
                    cout << "Warning: Point Cloud is empty!" << endl;
            }
            else
                cout << "Warning: Point Cloud is not organized!" << endl;
        }
    }

    void run ()
    {
        pcl::Grabber* interface = new pcl::OpenNIGrabber();
        boost::function<void (const pcl::PointCloud<pcl::PointXYZRGBA>::ConstPtr&)> f =
            boost::bind (&SimpleOpenNIViewer::cloud_cb_, this, _1);

        interface->registerCallback (f);

        interface->start ();

        while (!viewer.wasStopped())
        {
            boost::this_thread::sleep (boost::posix_time::seconds (1));
        }

        interface->stop ();
    }


private: 
    static void mouseCallback(int event, int x, int y, int flags, void *param)
    {
        SimpleOpenNIViewer *self = static_cast<SimpleOpenNIViewer *>(param);
        self->doMouseCallback(event, x, y, flags, NULL);
    }

    void doMouseCallback(int event, int x, int y, int flags, void* userData)
    {
        if (event == cv::EVENT_RBUTTONDOWN)
        {
            mtx.lock();
            mouse_x = x;
            mouse_y = y;
            mtx.unlock();
            cout << "mouse_x" << x << "mouse_y" << y << endl; 
        }
    }

    pcl::visualization::CloudViewer viewer;
    cv:: Mat rgbFrame, hsvFrame, grayFrame, drawing, contourMask;
    mutex mtx;
    int mouse_x = 0;
    int mouse_y = 0;
};

int main ()
{
    SimpleOpenNIViewer v;
    v.run ();
    while(true){};
    return 0;
}