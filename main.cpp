/**
 * This program shows how to perform occlusion between
 * real and virtual object by using color data and
 * depth data.
 *
 * @version   $Id$ 1.0 main.cpp
 *
 * @author   Niyati Shah
 *
 * Cite: https://docs.opencv.org/master/d2/dbd/tutorial_distance_transform.html
 *
 *
 *
 *	$Log$
 */
#include <iostream>
#include <opencv2/opencv.hpp>


using namespace std;
using namespace cv;


#define COLORS_black  Scalar(0,0,0)
#define COLORS_white  Scalar(255,255,255)

Point prevPt(-1, -1);
int prevX = 0, prevY = 0;

Mat FinalMask,tempImg, markerMask, OrigImage;

int depthSegment[6];

// Functions defined.
Mat ThresholdDepth(int level, const Mat &clustered);
void CalculateHist(Mat dst);
Mat getCluster(Mat depthsrc);
static void onMouse(int event, int x, int y, int flags, void*);
Mat getWatershed(Mat ReadImage);
Mat createMask(Mat img);
void clipVirtualObject(int x, int y, Mat image);
void Draw(Mat image);

int main(int argc, char** argv)
{

    string img = "disp.png";
    string depth = "depth";
    int num = 1;

    // Read in the image
    Mat colorImage =imread(to_string(num) +img,1 ) ;// depth image
    Mat depthImage = imread(to_string(num)+depth +img, 1); // Color image

    // Check if color image found
    if(colorImage.empty()){
        cout<<"No image found";
        return -11;
    }
    // Check if depth image found
    if(depthImage.empty()){
        cout<<"No image found";
        return -11;
    }

    // Create a copy of color image
    colorImage.copyTo(tempImg);

    //Initialise previous points for virtual objects
    prevX = colorImage.cols / 2;
    prevY = colorImage.rows / 2;

    // Initialize a mask to store color segmentation.
    Mat colormask = Mat::zeros(colorImage.size(), colorImage.type());

    // call watershed on color image
    colormask   = getWatershed(colorImage);

    // convert the mask to color image (3 channel image)
    cvtColor(colormask, colormask, COLOR_GRAY2BGR);

    // Get the data clustered into 5 levels
    Mat clustered = getCluster(depthImage);

    // Initialize the depth mask by the size of colormask
    Mat depthMask = Mat::zeros(colormask.size(), colormask.type());

    while (1)
    {
        // Display final image for scene
        imshow("Image", colorImage);

        // Initialize a waitkey for user inputs
        int c = waitKey(0);
        if ((char)c == 27)
            break;

        // Select the depth levels to place the input
        // ie Z axis
        /*
         *  Level 1: Completely in front of the object
         *  Level 5: Completely behind all the objects
         *  Level 2-4 : In between the objects
         */
        if ((char)c == '1')
            depthMask = ThresholdDepth(1,clustered);
        if ((char)c == '2')
            depthMask = ThresholdDepth(2,clustered);
        if ((char)c == '3')
            depthMask = ThresholdDepth(3,clustered);
        if ((char)c == '4')
            depthMask = ThresholdDepth(4,clustered);
        if ((char)c == '5')
            depthMask = ThresholdDepth(5,clustered);

        // Create a final mask with color and depth mask
        bitwise_and(depthMask, colormask, FinalMask);
        depthMask.copyTo(FinalMask);
        imshow("mask",FinalMask);


        // Select the X and Y axis to move the virtual object
        /*
         *  p | P : Draw/initialize virtual
         *  w | W : Move object up
         *  s | S : Move object down
         *  a | A : Move object left
         *  d | D : Move object right
         */
        if ((char)c == 'p' || (char)c == 'P' ) {
            //case Draw
            Draw(colorImage);
        }
        if ((char)c == 'w'|| (char)c == 'W') {
            //case KEY_UP:
            clipVirtualObject(0, -2, colorImage);
        }
        if ((char)c == 's'|| (char)c == 'S') {
            //case KEY_DOWN:
            clipVirtualObject(0, 2, colorImage);
        }
        if ((char)c == 'a'|| (char)c == 'A') {
            //case KEY_LEFT:
            clipVirtualObject(-2, 0, colorImage);
        }
        if ((char)c == 'd'|| (char)c == 'D') {
            //case KEY_RIGHT:
            clipVirtualObject(2, 0, colorImage);
        }
    }
    waitKey(0);
}

/**
*
* This function creates the different clusters uisng k means
* algorithm of the different depths. where k is 6
*
* @param depthsrc : Depth image
* @return : clustered depth image with number of clusters as 5
*/

Mat getCluster(Mat depthsrc) {

    // Initialize a samples Mat object
    Mat samples(depthsrc.rows * depthsrc.cols, 3, CV_32F);

    for (int y = 0; y < depthsrc.rows; y++)
        for (int x = 0; x < depthsrc.cols; x++)
            for (int z = 0; z < 3; z++)
                samples.at<float>(y + x * depthsrc.rows, z) = depthsrc.at<Vec3b>(y, x)[z];


    // Initailize the different parameters for K means
    int clusterCount = 6;
    Mat labels;
    int attempts = 5;
    Mat centers;
    kmeans(samples, clusterCount, labels, TermCriteria(CV_TERMCRIT_ITER | CV_TERMCRIT_EPS, 10000, 0.0001), attempts, KMEANS_PP_CENTERS, centers);

    // Apply the kmeans to the given image
    Mat clusteredDepth(depthsrc.size(), depthsrc.type());
    for (int y = 0; y < depthsrc.rows; y++)
        for (int x = 0; x < depthsrc.cols; x++)
        {
            int cluster_idx = labels.at<int>(y + x * depthsrc.rows, 0);
            clusteredDepth.at<Vec3b>(y, x)[0] = centers.at<float>(cluster_idx, 0);
            clusteredDepth.at<Vec3b>(y, x)[1] = centers.at<float>(cluster_idx, 1);
            clusteredDepth.at<Vec3b>(y, x)[2] = centers.at<float>(cluster_idx, 2);

        }

    // Calculate histogram to get all different values at which the cluster is created.
    CalculateHist(clusteredDepth);
    imshow("clustered depth", clusteredDepth);
    return clusteredDepth;
}

/**
*
* This function uses thresholds the depth image to create a depth
* mask at the given depth level
*
* @param level : Depth level (Z axis)
* @return : Depth mask at given depth level
*/

Mat ThresholdDepth(int level, const Mat &clustered)
{
    //Initialize resultant depth mask mat object
    Mat depthMask;
    /*
     *  Level 1: Completely in front of the object
     *  Level 5: Completely behind all the objects
     *  Level 2-4 : In between the objects
     */
    int n =sizeof(depthSegment)/sizeof(depthSegment[0])-level;
    threshold(clustered, depthMask, depthSegment[n], 255, 3);

    return depthMask;
}

/**
*
* This function calculates teh histogram to
* find the depth level
*
* @param img : clustered image
* @return : Depth mask at given depth level
*/

void CalculateHist(Mat img) {
    Mat gray = img;
    cvtColor(gray, gray, CV_RGB2GRAY);
    // Initialize parameters
    int histSize = 256;    // bin size
    float range[] = { 0, 255 };
    const float *ranges[] = { range };

    // Calculate histogram
    MatND hist;
    calcHist(&gray, 1, 0, Mat(), hist, 1, &histSize, ranges, true, false);

    int idx = 0;
    for (int h = 0; h < histSize; h++) {
        float binVal = hist.at<float>(h);
        if (binVal>0) {
            depthSegment[idx] = h;
            idx++;
        }
    }
}

/**
*
* This function is for the on mouse events which is used to
* mark the different segments of the image for it to perform
* supervised learning.
*
* @param event : integer for the type of mouse event
* @param x : x coordinate of the mouse event
* @param y : y coordinate of the mouse event
* @param flags : condition whenever a mouse event occurs
*/
static void onMouse(int event, int x, int y, int flags, void*)
{
    // if mouse event is not within the borders of given image then no action taken
    if (x < 0 || x >= OrigImage.cols || y < 0 || y >= OrigImage.rows)
        return;

    // check if the left mouse button released and set presvious point to (-1,-1)
    if (event == EVENT_LBUTTONUP || !(flags & EVENT_FLAG_LBUTTON))
        prevPt = Point(-1, -1);
    else if (event == EVENT_LBUTTONDOWN) // if the left button is clicked (no dragging)
        prevPt = Point(x, y);
    else if (event == EVENT_MOUSEMOVE && (flags & EVENT_FLAG_LBUTTON)) // if there is a dragging
    {
        // create a point x,y
        Point pt(x, y);
        // check if the x co-ordinate is less than 0
        // then previous point becomes current point
        if (prevPt.x < 0)
            prevPt = pt;
        // on the mask image and given image , make a line
        // using the point co-ordinates
        // of the current point and previous point
        line(markerMask, prevPt, pt, Scalar::all(255), 5, 8, 0);
        line(OrigImage, prevPt, pt, Scalar::all(255), 5, 8, 0);
        // store the current point in the previous point
        prevPt = pt;
        // display the image with the changes
        imshow("image", OrigImage);
    }

}

/**
*
* This function is used to perform watershed transform
* on the image. Here the two parts will be to
* the selected object and the remaining background.
*
* @param ReadImage : image of scene
* @return F mask : segmented mask
*/
Mat getWatershed(Mat ReadImage) {
    Mat  imgGray, TempImg = ReadImage;
    Scalar clr[] = { COLORS_black,COLORS_white };

    int blue = -1, green = -1, red = -1;
    //create a window with a name to display the image in a fixed (no resize allowed) size format.
    namedWindow("image", WINDOW_AUTOSIZE);
    // Copies the temporary image to a original image
    TempImg.copyTo(OrigImage);

    // Convert the image to gray scale and store in makerMask mat object
    cvtColor(OrigImage, markerMask, COLOR_BGR2GRAY);

    // To see the overlap of the segments over the original image
    //  the below command is used.
    cvtColor(markerMask, imgGray, COLOR_GRAY2BGR);


    // fill zeroes in a markerMask Mat object containing image
    markerMask = Scalar::all(0);


    // Display the given image
    imshow("image", OrigImage);

    //  Check for mouse click on image
    setMouseCallback("image", onMouse, 0);

    Mat Fmask = Mat::zeros(ReadImage.size(), ReadImage.type()), submask = Mat::ones(ReadImage.size(), ReadImage.type()) * 255;
    //
    // infinite for loop to keep waiting for the user to
    // select the appropriate option as described in the
    // help function
    //
    while (true) {

        // Use a waitkey to keep displaying the image till the user interrupts it.
        int c = waitKey(0);
        if ((char)c == 27)
            break;

        // If a mistake is made while selecting segments
        // click 'r' to clear the image and restart
        // selection of segments.
        if ((char)c == 'r') {
            // fill zeroes in a markerMask Mat object
            markerMask = Scalar::all(0);
            // copy the image
            TempImg.copyTo(OrigImage);
            // display  the image
            imshow("image", OrigImage);
        }

        // If 'w' or space is selected
        // start the watershed
        if ((char)c == 'w' || (char)c == ' ') {

            // initialise a segment count to 0
            // create a vector of vector to store multiple contours
            // create a 4dimensional vector to store hierarchy of contours
            int compCount = 0;
            vector<vector<Point> > contours;
            vector<Vec4i> hierarchy;

            //  find contours in the image
            // use the RETR_CCOMP to get all contours and put them into a two-level hierarchy
            // use the CHAIN_APPROX_SIMPLE to compress horizontal, vertical, and diagonal segments
            // into only their endpoints
            findContours(markerMask, contours, hierarchy, CV_RETR_CCOMP, CHAIN_APPROX_SIMPLE);

            // check if atleast one segment is selected
            if (!contours.empty()) {

                // Create a mat object markers that is of size of the masker and type CV_32S
                // and fill the fill zeroes in
                Mat markers(markerMask.size(), CV_32S);
                markers = Scalar::all(0);


                // For each of the contour that is found earlier, draw the shape
                // and also count the number of contours made,
                for (int idx = 0; idx >= 0; idx = hierarchy[idx][0], compCount++) {
                    drawContours(markers, contours, idx, Scalar::all(compCount + 1), -1, 8, hierarchy, INT_MAX);

                }

                // Create a vector object to store the
                // different colors that are randomly generated
                vector<Vec3b> colorTab;

                for (int idx = 0; idx < compCount; idx++) {
                    blue = clr[idx][0];
                    green = clr[idx][1];
                    blue = clr[idx][2];

                    colorTab.push_back(Vec3b((uchar)blue, (uchar)green, (uchar)red));
                }

                // run the watershed function on the image and mask
                watershed(TempImg, markers);

                // create a mat object to store output image
                Mat outputImage(markers.size(), CV_8UC3);

                // paint through the contours of the watershed
                // image by going through each row and column
                for (int row = 0; row < markers.rows; row++) {
                    for (int col = 0; col < markers.cols; col++) {
                        int index = markers.at<int>(row, col);
                        if (index == -1)
                            outputImage.at<Vec3b>(row, col) = Vec3b(255, 255, 255);
                        else if (index <= 0 || index > compCount)
                            outputImage.at<Vec3b>(row, col) = Vec3b(0, 0, 0);
                        else
                            outputImage.at<Vec3b>(row, col) = colorTab[index - 1];

                    }
                }

                //Get the mask as we want
                cvtColor(outputImage, Fmask, CV_RGB2GRAY);
                Mat sub_mat = Mat::ones(Fmask.size(), Fmask.type()) * 255;
                subtract(sub_mat, Fmask, Fmask);

                return Fmask;

            }
            else {
                // Error message if no segments selected
                cout << "\n\nPlease select at least one segment to form contours\n"
                        " OR Look at help doc to see how to select segments ";
                return Fmask;
            }
        }
    }
}

/**
*
* This function to draw the virtual object in this case a cube.
*
* @param image : Draw the virtual object on the passed image
*
*/
void Draw(Mat image){
    int scale = 50;
    vector< Point> contour;
    contour.push_back(Point(prevX, prevY+scale));
    contour.push_back(Point(prevX, prevY+scale*3));
    contour.push_back(Point(prevX+2*scale, prevY+scale*3));
    contour.push_back(Point(prevX+2*scale, prevY+scale));

    const Point *pts = (const cv::Point*) Mat(contour).data;
    int npts = Mat(contour).rows;
    polylines(image, &pts, &npts, 1, true, Scalar(0, 255, 0));
    fillConvexPoly(image, pts, 4, cv::Scalar(0, 0, 200));

    vector< Point> contour1;
    contour1.push_back(Point(prevX+2*scale, prevY+scale));
    contour1.push_back(Point(prevX+2*scale, prevY+3*scale));
    contour1.push_back(Point(prevX+3*scale, prevY+2*scale));
    contour1.push_back(Point(prevX+3*scale, prevY));

    const Point *pts1 = (const cv::Point*) Mat(contour1).data;
    npts = Mat(contour1).rows;
    polylines(image, &pts1, &npts, 1, true, Scalar(0, 255, 0));
    fillConvexPoly(image, pts1, 4, cv::Scalar(0, 255, 0));

    vector< Point> contour2;
    contour2.push_back(Point(prevX+scale, prevY));
    contour2.push_back(Point(prevX, prevY+scale));
    contour2.push_back(Point(prevX+2*scale, prevY+scale));
    contour2.push_back(Point(prevX+3*scale, prevY));

    const Point *pts2 = (const cv::Point*) Mat(contour2).data;
    npts = Mat(contour2).rows;
    polylines(image, &pts2, &npts, 1, true, Scalar(0, 255, 0));
    fillConvexPoly(image, pts2, 4, cv::Scalar(255, 0, 0));

}

/**
*
* This function to clip  virtual object and move it
*
* @param x : movement in x direction
* @param y : movement in y direction
* @param background : background scene image
*
*/
void clipVirtualObject(int x, int y, Mat background) {
    // Initialize the foreground scene
    Mat foreground = Mat::zeros(background.size(), background.type());

    tempImg.copyTo(background);

    // change the position of virtual object according to new point
    prevX = prevX + x;
    prevY = prevY + y;

    // Call draw fucntion for virtual object at new position.
    Draw(foreground);

    // initialize mat objecs for use.
    Mat bitAnd,sub,complement, com2;

    // Subtract the foreground and mask
    subtract(foreground, FinalMask, sub);

    // Perform bitwise and and subtraction to clip the data
    bitwise_and(foreground, FinalMask, bitAnd);
    subtract(sub, bitAnd, complement);

    // Create an alpha mask to add the clipped virtual object to the scene
    Mat alpha;
    complement.copyTo(alpha);

    alpha.convertTo(alpha, background.type(), 1.0/255);
    Mat out = Mat::zeros(background.size(), background.type());;
    multiply(alpha, foreground, foreground);
    multiply(Scalar::all(1.0) - alpha, background, background);
    add(foreground, background, out);
    subtract(foreground, FinalMask, complement);
    imshow("clipping", complement);
    background =complement + background;

}


