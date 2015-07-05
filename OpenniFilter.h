#include <pcl/io/openni_grabber.h>
#include <pcl/visualization/cloud_viewer.h>
#include <pcl/point_types.h>
#include <opencv2/core/core.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <mutex>

using namespace std;
using namespace cv;

class OpenniFilter
{
public:
    OpenniFilter();
    virtual ~OpenniFilter();

    /** runs the cloud viewer */
    void run();

private: 
    /** processes cloud data */
    void cloud_cb_ (const pcl::PointCloud<pcl::PointXYZRGBA>::ConstPtr &cloud);

    /** processes mouse actions */
    void doMouseCallback(int event, int x, int y, int flags, void* userData);
    
    /** provides an interface for opencv to utilize the call back function */
    static void mouseCallback(int event, int x, int y, int flags, void *param); 

    /** a simple cloud viewer with limited functionality */   
    pcl::visualization::CloudViewer viewer;

    /** provides interface for the viewer to display cloud data */
    pcl::Grabber* interface;

    /** intermediate frames to store processed image data */
    Mat rgbFrame, hsvFrame, grayFrame, drawing, contourMask;

    /** synchronization data for the cloud thread to capture the mouse position */
    mutex mtx;
    int mouse_x = 0;
    int mouse_y = 0;

    /** data used for filtering */
    int iLowS = 30;          
    int iHighS = 246;
    int iLowV = 54;        
    int iHighV = 255;
    int iHueDev = 27;

    /** names for the windows to be displayed */
    string windowParam = "parameters";
    string windowTracker = "tracker";
    string windowFilter = "filtered frame";
};
