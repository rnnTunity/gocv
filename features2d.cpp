#include "features2d.h"


void matches2points_nndr(const std::vector<cv::KeyPoint>& train,
                         const std::vector<cv::KeyPoint>& query,
                         const std::vector<std::vector<cv::DMatch> >& matches,
                         float nndr,
                         std::vector<cv::Point2f>& capture_points,
     					 std::vector<cv::Point2f>& stream_points) {

  float dist1 = 0.0, dist2 = 0.0;
  for (size_t i = 0; i < matches.size(); i++) {
    cv::DMatch dmatch = matches[i][0];
    dist1 = matches[i][0].distance;
    dist2 = matches[i][1].distance;

    if (dist1 < nndr*dist2) {
      capture_points.push_back(train[dmatch.queryIdx].pt);
      stream_points.push_back(query[dmatch.trainIdx].pt);
    }
  }
}


int match_features( cv::Mat const& f1, cv::Mat const& f2,
                    std::vector<cv::KeyPoint>& keypointsF1,
                    std::vector<cv::KeyPoint>& keypointsF2,
					std::vector<cv::Point2f>& capture_points,
					std::vector<cv::Point2f>& stream_points)
{
	if (f1.size().area() > 0 && f2.size().area() > 0)
	{
        std::vector<std::vector<cv::DMatch> > dmatches;
        cv::Ptr<cv::DescriptorMatcher> matcher_l1 = cv::DescriptorMatcher::create("BruteForce-Hamming");

        matcher_l1->knnMatch(f1, f2, dmatches, 2);
        matches2points_nndr(keypointsF1, keypointsF2, dmatches, 0.8f, capture_points, stream_points);
	} else {
		return -1;
	}
    if (capture_points.size() == 0 || (capture_points.size() != stream_points.size()) ) {
        return  -1;
    }
	return 0;
}

Mat find_homography_a( KeyPoints keypointsF1, Mat features_a, KeyPoints keypointsF2, Mat features_b ){

std::vector<cv::KeyPoint> kp1;
std::vector<cv::KeyPoint> kp2;

int i;
for (i=0; i<keypointsF1.length; i++) {
    KeyPoint kp =  keypointsF1.keypoints[i];
    cv::KeyPoint k = cv::KeyPoint(kp.x, kp.y, kp.size, kp.angle, kp.response, kp.octave, kp.classID);
    kp1.push_back(k);
}

for (i=0; i<keypointsF2.length; i++) {
    KeyPoint kp =  keypointsF2.keypoints[i];
    cv::KeyPoint k = cv::KeyPoint(kp.x, kp.y, kp.size, kp.angle, kp.response, kp.octave, kp.classID);
    kp2.push_back(k);
}

std::vector<cv::Point2f> capture_points, stream_points;
std::vector<std::vector<cv::DMatch> > dmatches;

int match_result = match_features(*features_a, *features_b, kp1, kp2,
        capture_points, stream_points);

if (match_result == 0)
{
    cv::Mat res = cv::estimateRigidTransform(capture_points, stream_points, true);

    if (res.empty())
     return NULL;

    cv::Mat* resPtr = new cv::Mat();
    res.copyTo(*resPtr);
    return resPtr;
}
else
    return NULL;//&cv::Mat();
}
/////////////////////
void equalizeHist(Mat src, Mat des) {
    cv::equalizeHist(*src, *des);
}
/////////////////////

struct Lines createAndDetectLineSegmentDetector(Mat src) {
  cv::Ptr<cv::LineSegmentDetector> ls = cv::createLineSegmentDetector();
  std::vector<cv::Vec4f> lines_std;
    // Detect the lines
    ls->detect(*src, lines_std);

    LineSegment* line = new LineSegment[lines_std.size()];

    for (size_t i = 0; i < lines_std.size(); ++i) {
        LineSegment k = {lines_std[i][0], lines_std[i][1], lines_std[i][2], lines_std[i][3]};
        line[i] = k;
    }

    Lines ret = {line, (int)lines_std.size()};
    return ret;
}

/////////////////////

AKAZE AKAZE_Create() {
    // TODO: params

    return new cv::Ptr<cv::AKAZE>(cv::AKAZE::create(5, 0, 3, 0.001, 1, 1, 1));
}

void AKAZE_Close(AKAZE a) {
    delete a;
}

struct KeyPoints AKAZE_Detect(AKAZE a, Mat src) {
    std::vector<cv::KeyPoint> detected;
    (*a)->detect(*src, detected);

    KeyPoint* kps = new KeyPoint[detected.size()];

    for (size_t i = 0; i < detected.size(); ++i) {
        KeyPoint k = {detected[i].pt.x, detected[i].pt.y, detected[i].size, detected[i].angle,
                      detected[i].response, detected[i].octave, detected[i].class_id
                     };
        kps[i] = k;
    }

    KeyPoints ret = {kps, (int)detected.size()};
    return ret;
}

struct KeyPoints AKAZE_DetectAndCompute(AKAZE a, Mat src, Mat mask, Mat desc) {
    std::vector<cv::KeyPoint> detected;
    (*a)->detectAndCompute(*src, *mask, detected, *desc);

    KeyPoint* kps = new KeyPoint[detected.size()];

    for (size_t i = 0; i < detected.size(); ++i) {
        KeyPoint k = {detected[i].pt.x, detected[i].pt.y, detected[i].size, detected[i].angle,
                      detected[i].response, detected[i].octave, detected[i].class_id
                     };
        kps[i] = k;
    }

    KeyPoints ret = {kps, (int)detected.size()};
    return ret;
}

AgastFeatureDetector AgastFeatureDetector_Create() {
    // TODO: params
    return new cv::Ptr<cv::AgastFeatureDetector>(cv::AgastFeatureDetector::create());
}

void AgastFeatureDetector_Close(AgastFeatureDetector a) {
    delete a;
}

struct KeyPoints AgastFeatureDetector_Detect(AgastFeatureDetector a, Mat src) {
    std::vector<cv::KeyPoint> detected;
    (*a)->detect(*src, detected);

    KeyPoint* kps = new KeyPoint[detected.size()];

    for (size_t i = 0; i < detected.size(); ++i) {
        KeyPoint k = {detected[i].pt.x, detected[i].pt.y, detected[i].size, detected[i].angle,
                      detected[i].response, detected[i].octave, detected[i].class_id
                     };
        kps[i] = k;
    }

    KeyPoints ret = {kps, (int)detected.size()};
    return ret;
}

BRISK BRISK_Create() {
    // TODO: params
    return new cv::Ptr<cv::BRISK>(cv::BRISK::create());
}

void BRISK_Close(BRISK b) {
    delete b;
}

struct KeyPoints BRISK_Detect(BRISK b, Mat src) {
    std::vector<cv::KeyPoint> detected;
    (*b)->detect(*src, detected);

    KeyPoint* kps = new KeyPoint[detected.size()];

    for (size_t i = 0; i < detected.size(); ++i) {
        KeyPoint k = {detected[i].pt.x, detected[i].pt.y, detected[i].size, detected[i].angle,
                      detected[i].response, detected[i].octave, detected[i].class_id
                     };
        kps[i] = k;
    }

    KeyPoints ret = {kps, (int)detected.size()};
    return ret;
}

struct KeyPoints BRISK_DetectAndCompute(BRISK b, Mat src, Mat mask, Mat desc) {
    std::vector<cv::KeyPoint> detected;
    (*b)->detectAndCompute(*src, *mask, detected, *desc);

    KeyPoint* kps = new KeyPoint[detected.size()];

    for (size_t i = 0; i < detected.size(); ++i) {
        KeyPoint k = {detected[i].pt.x, detected[i].pt.y, detected[i].size, detected[i].angle,
                      detected[i].response, detected[i].octave, detected[i].class_id
                     };
        kps[i] = k;
    }

    KeyPoints ret = {kps, (int)detected.size()};
    return ret;
}

GFTTDetector GFTTDetector_Create() {
    // TODO: params
    return new cv::Ptr<cv::GFTTDetector>(cv::GFTTDetector::create());
}

void GFTTDetector_Close(GFTTDetector a) {
    delete a;
}

struct KeyPoints GFTTDetector_Detect(GFTTDetector a, Mat src) {
    std::vector<cv::KeyPoint> detected;
    (*a)->detect(*src, detected);

    KeyPoint* kps = new KeyPoint[detected.size()];

    for (size_t i = 0; i < detected.size(); ++i) {
        KeyPoint k = {detected[i].pt.x, detected[i].pt.y, detected[i].size, detected[i].angle,
                      detected[i].response, detected[i].octave, detected[i].class_id
                     };
        kps[i] = k;
    }

    KeyPoints ret = {kps, (int)detected.size()};
    return ret;
}

KAZE KAZE_Create() {
    // TODO: params
    return new cv::Ptr<cv::KAZE>(cv::KAZE::create());
}

void KAZE_Close(KAZE a) {
    delete a;
}

struct KeyPoints KAZE_Detect(KAZE a, Mat src) {
    std::vector<cv::KeyPoint> detected;
    (*a)->detect(*src, detected);

    KeyPoint* kps = new KeyPoint[detected.size()];

    for (size_t i = 0; i < detected.size(); ++i) {
        KeyPoint k = {detected[i].pt.x, detected[i].pt.y, detected[i].size, detected[i].angle,
                      detected[i].response, detected[i].octave, detected[i].class_id
                     };
        kps[i] = k;
    }

    KeyPoints ret = {kps, (int)detected.size()};
    return ret;
}

struct KeyPoints KAZE_DetectAndCompute(KAZE a, Mat src, Mat mask, Mat desc) {
    std::vector<cv::KeyPoint> detected;
    (*a)->detectAndCompute(*src, *mask, detected, *desc);

    KeyPoint* kps = new KeyPoint[detected.size()];

    for (size_t i = 0; i < detected.size(); ++i) {
        KeyPoint k = {detected[i].pt.x, detected[i].pt.y, detected[i].size, detected[i].angle,
                      detected[i].response, detected[i].octave, detected[i].class_id
                     };
        kps[i] = k;
    }

    KeyPoints ret = {kps, (int)detected.size()};
    return ret;
}

MSER MSER_Create() {
    // TODO: params
    return new cv::Ptr<cv::MSER>(cv::MSER::create());
}

void MSER_Close(MSER a) {
    delete a;
}

struct KeyPoints MSER_Detect(MSER a, Mat src) {
    std::vector<cv::KeyPoint> detected;
    (*a)->detect(*src, detected);

    KeyPoint* kps = new KeyPoint[detected.size()];

    for (size_t i = 0; i < detected.size(); ++i) {
        KeyPoint k = {detected[i].pt.x, detected[i].pt.y, detected[i].size, detected[i].angle,
                      detected[i].response, detected[i].octave, detected[i].class_id
                     };
        kps[i] = k;
    }

    KeyPoints ret = {kps, (int)detected.size()};
    return ret;
}

FastFeatureDetector FastFeatureDetector_Create() {
    // TODO: params
    return new cv::Ptr<cv::FastFeatureDetector>(cv::FastFeatureDetector::create());
}

void FastFeatureDetector_Close(FastFeatureDetector f) {
    delete f;
}

struct KeyPoints FastFeatureDetector_Detect(FastFeatureDetector f, Mat src) {
    std::vector<cv::KeyPoint> detected;
    (*f)->detect(*src, detected);

    KeyPoint* kps = new KeyPoint[detected.size()];

    for (size_t i = 0; i < detected.size(); ++i) {
        KeyPoint k = {detected[i].pt.x, detected[i].pt.y, detected[i].size, detected[i].angle,
                      detected[i].response, detected[i].octave, detected[i].class_id
                     };
        kps[i] = k;
    }

    KeyPoints ret = {kps, (int)detected.size()};
    return ret;
}

ORB ORB_Create() {
    // TODO: params
    return new cv::Ptr<cv::ORB>(cv::ORB::create());
}

void ORB_Close(ORB o) {
    delete o;
}

struct KeyPoints ORB_Detect(ORB o, Mat src) {
    std::vector<cv::KeyPoint> detected;
    (*o)->detect(*src, detected);

    KeyPoint* kps = new KeyPoint[detected.size()];

    for (size_t i = 0; i < detected.size(); ++i) {
        KeyPoint k = {detected[i].pt.x, detected[i].pt.y, detected[i].size, detected[i].angle,
                      detected[i].response, detected[i].octave, detected[i].class_id
                     };
        kps[i] = k;
    }

    KeyPoints ret = {kps, (int)detected.size()};
    return ret;
}

struct KeyPoints ORB_DetectAndCompute(ORB o, Mat src, Mat mask, Mat desc) {
    std::vector<cv::KeyPoint> detected;
    (*o)->detectAndCompute(*src, *mask, detected, *desc);

    KeyPoint* kps = new KeyPoint[detected.size()];

    for (size_t i = 0; i < detected.size(); ++i) {
        KeyPoint k = {detected[i].pt.x, detected[i].pt.y, detected[i].size, detected[i].angle,
                      detected[i].response, detected[i].octave, detected[i].class_id
                     };
        kps[i] = k;
    }

    KeyPoints ret = {kps, (int)detected.size()};
    return ret;
}

SimpleBlobDetector SimpleBlobDetector_Create() {
    // TODO: params
    return new cv::Ptr<cv::SimpleBlobDetector>(cv::SimpleBlobDetector::create());
}

void SimpleBlobDetector_Close(SimpleBlobDetector b) {
    delete b;
}

struct KeyPoints SimpleBlobDetector_Detect(SimpleBlobDetector b, Mat src) {
    std::vector<cv::KeyPoint> detected;
    (*b)->detect(*src, detected);

    KeyPoint* kps = new KeyPoint[detected.size()];

    for (size_t i = 0; i < detected.size(); ++i) {
        KeyPoint k = {detected[i].pt.x, detected[i].pt.y, detected[i].size, detected[i].angle,
                      detected[i].response, detected[i].octave, detected[i].class_id
                     };
        kps[i] = k;
    }

    KeyPoints ret = {kps, (int)detected.size()};
    return ret;
}
