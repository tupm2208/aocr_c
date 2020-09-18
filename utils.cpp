#include "utils.h"

#include <math.h>
#include <fstream>
#include <utility>
#include <vector>
#include <iostream>
#include <regex>

#include "tensorflow/cc/ops/const_op.h"
#include "tensorflow/cc/ops/image_ops.h"
#include "tensorflow/cc/ops/standard_ops.h"
#include "tensorflow/core/framework/graph.pb.h"
#include "tensorflow/core/framework/tensor.h"
#include "tensorflow/core/graph/default_device.h"
#include "tensorflow/core/graph/graph_def_builder.h"
#include "tensorflow/core/lib/core/errors.h"
#include "tensorflow/core/lib/core/stringpiece.h"
#include "tensorflow/core/lib/core/threadpool.h"
#include "tensorflow/core/lib/io/path.h"
#include "tensorflow/core/lib/strings/stringprintf.h"
#include "tensorflow/core/platform/env.h"
#include "tensorflow/core/platform/init_main.h"
#include "tensorflow/core/platform/logging.h"
#include "tensorflow/core/platform/types.h"
#include "tensorflow/core/public/session.h"
#include "tensorflow/core/util/command_line_flags.h"
#include "tensorflow/cc/client/client_session.h"

// #include <cv.hpp>
// #include "opencv2/opencv.hpp"
#include <opencv2/core/mat.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/highgui/highgui.hpp>
#include "tensorflow/core/public/session.h"

using namespace std;
using namespace cv;

using tensorflow::Flag;
using tensorflow::Tensor;
using tensorflow::Status;
using tensorflow::string;
using tensorflow::int32;

/** Read a model graph definition (xxx.pb) from disk, and creates a session object you can use to run it.
 */
Status loadGraph(const string &graph_file_name,
                 shared_ptr<tensorflow::Session> *session) {
    tensorflow::GraphDef graph_def;
    Status load_graph_status =
            ReadBinaryProto(tensorflow::Env::Default(), graph_file_name, &graph_def);
    if (!load_graph_status.ok()) {
        return tensorflow::errors::NotFound("Failed to load compute graph at '",
                                            graph_file_name, "'");
    }
    session->reset(tensorflow::NewSession(tensorflow::SessionOptions()));
    Status session_create_status = (*session)->Create(graph_def);
    if (!session_create_status.ok()) {
        return session_create_status;
    }
    int node_count = graph_def.node_size();
    
    
    // ofstream myfile ("example.txt");
    // for (int i = 0; i < node_count; i++)
    // {
    //         auto n = graph_def.node(i);
    //         myfile << n.name() << endl;
    //         cout<<"Names : "<< n.name() << node_count <<endl;

    // }
    // myfile.close();
    return Status::OK();
}

tensorflow::Scope transposeScope = tensorflow::Scope::NewRootScope();
vector<Tensor> temOutputs;
tensorflow::ClientSession sess(transposeScope);

Tensor handleCNNOutput(vector<Tensor> outputTensors) {
    
    auto transOps = tensorflow::ops::ConjugateTranspose(transposeScope, outputTensors[0], tensorflow::ops::Const(transposeScope, {1, 0, 2}));
    TF_CHECK_OK(sess.Run({transOps}, &outputTensors));

    int width = outputTensors[0].shape().dim_size(0);
    auto padValue = tensorflow::ops::Const(transposeScope, {{0, 400-width}, {0, 0}, {0, 0}});
    auto padOps = tensorflow::ops::Pad(transposeScope, outputTensors[0], padValue);
    TF_CHECK_OK(sess.Run({padOps}, &outputTensors));

    return outputTensors[0];
}

void getTensorMask(vector<int> limit, vector<Tensor> &outputMask) {
    for (int i = 0; i < 400; i++) {
        tensorflow::TensorShape mks{int(limit.size()), 1};
        Tensor mask(tensorflow::DT_FLOAT, mks);
        auto input_tensor_mapped = mask.tensor<float, 2>();

        int c = 0;
        for(int j: limit) {
            
            input_tensor_mapped(c, 0) = i < j? 1.0f: 0.0f;
            c += 1;
        }
        outputMask.push_back(mask);
        // cout << mask.shape() << endl;
    }
}

vector<vector<int>> handleFinalOutput(std::vector<Tensor> &inputs) {

    int batch = inputs[0].shape().dim_size(0);
    vector<vector<int>> listOutput;
    for(int i = 0; i<batch; i++) {
        vector<int> element;
        listOutput.push_back(element);
    }
    vector<Tensor> outputs;
    for (Tensor outputTensor: inputs) {
        auto argOps = tensorflow::ops::ArgMax(transposeScope, outputTensor, tensorflow::ops::Const(transposeScope, 1));
        TF_CHECK_OK(sess.Run({argOps}, &temOutputs));

        auto castOps = tensorflow::ops::Cast(transposeScope, temOutputs[0], tensorflow::DataType::DT_INT32);
        TF_CHECK_OK(sess.Run({castOps}, &temOutputs));

        auto bz_map = temOutputs[0].tensor<int32, 1>();
        for(int i = 0; i<batch; i++) {
            listOutput.at(i).push_back(bz_map(i));
        }
    }

    cout << listOutput[0].size() << " " << listOutput[1].size() << endl;
    return listOutput;
}


Tensor cnnPart(vector<Mat> images, std::shared_ptr<tensorflow::Session> session, vector<Tensor> &outputMask) {

    string inputLayer = "img_data";
    vector<string> outputLayer = {"Squeeze"};
    vector<tensorflow::Output> OutputList;
    
    vector<Tensor> outputs;
    vector <int> maskLimit;
    Tensor currentTensor;
    Tensor nextTensor;
    
    for(int i = 0; i < images.size(); i++) {
        Mat image = images[i];
        Tensor imageTensor = convertMatToTensorYolo(image);
        Status runStatus = session->Run({{inputLayer, imageTensor}}, outputLayer, {}, &temOutputs);

        if (!runStatus.ok()){
            LOG(INFO) << "Running model CNN part failed: " << runStatus;
        }
        maskLimit.push_back(temOutputs[0].shape().dim_size(1));
        if (i == 0) {
            currentTensor = handleCNNOutput(temOutputs);
        } else {
            nextTensor = handleCNNOutput(temOutputs);
            auto t1 = tensorflow::ops::Const(transposeScope, currentTensor);
            auto t2 = tensorflow::ops::Const(transposeScope, nextTensor);
            auto concatOps = tensorflow::ops::Concat(transposeScope, {t1, t2}, 1);
            TF_CHECK_OK(sess.Run({concatOps}, &outputs));

            currentTensor = outputs[0];
        }
    }
    getTensorMask(maskLimit, outputMask);
    return currentTensor;
}

vector<string> readLabels() {
    string label_path = "/home/tupm/HDD/projects/tensorflow-object-detection-cpp/assets/text_models/labels.txt";
    vector<string> outputlabels;
    string tem("");
    char x;
    ifstream inFile;
    inFile.open(label_path);
    if (!inFile) {
        cout << "Unable to open file";
        exit(1); // terminate with error
    }
    while(getline(inFile, tem)){
        // cout << tem << "\n";
        outputlabels.push_back(tem);
    }
    // cout << "number of labels: " << outputlabels.size() << endl;
    return outputlabels;
}

/** Read a labels map file (xxx.pbtxt) from disk to translate class numbers into human-readable labels.
 */
Status readLabelsMapFile(const string &fileName, map<int, string> &labelsMap) {

    // Read file into a string
    ifstream t(fileName);
    if (t.bad())
        return tensorflow::errors::NotFound("Failed to load labels map at '", fileName, "'");
    stringstream buffer;
    buffer << t.rdbuf();
    string fileString = buffer.str();

    // Search entry patterns of type 'item { ... }' and parse each of them
    smatch matcherEntry;
    smatch matcherId;
    smatch matcherName;
    const regex reEntry("item \\{([\\S\\s]*?)\\}");
    const regex reId("[0-9]+");
    const regex reName("\'.+\'");
    string entry;

    auto stringBegin = sregex_iterator(fileString.begin(), fileString.end(), reEntry);
    auto stringEnd = sregex_iterator();

    int id;
    string name;
    for (sregex_iterator i = stringBegin; i != stringEnd; i++) {
        matcherEntry = *i;
        entry = matcherEntry.str();
        regex_search(entry, matcherId, reId);
        if (!matcherId.empty())
            id = stoi(matcherId[0].str());
        else
            continue;
        regex_search(entry, matcherName, reName);
        if (!matcherName.empty())
            name = matcherName[0].str().substr(1, matcherName[0].str().length() - 2);
        else
            continue;
        labelsMap.insert(pair<int, string>(id, name));
    }
    return Status::OK();
}

Tensor convertMatToTensor(Mat &input)
{
    int height = input.rows;
    int width = input.cols;
    int depth = input.channels();

    Tensor imgTensor(tensorflow::DT_UINT8, tensorflow::TensorShape({1, height, width, depth}));

    uint8_t *p = imgTensor.flat<uint8_t>().data();
    Mat fakeMat(input.rows, input.cols, CV_8UC3, p);
    input.convertTo(fakeMat, CV_8UC3);

    // float* p = imgTensor.flat<float>().data();
    // Mat outputImg(height, width, CV_32FC3, p);
    // input.convertTo(outputImg, CV_32FC3);

    return imgTensor;
}

Tensor convertMatToTensorYolo(Mat &input)
{   
    cv::cvtColor(input, input, cv::COLOR_BGR2GRAY);
    threshold(input,input, 125, 255, THRESH_BINARY);
    int height = input.rows;
    int width = input.cols;
    int depth = input.channels();

    width = 32 * width/height;
    height = 32;
    cv::resize(input, input, cv::Size(cv::Size2d(width, 32)));

    Tensor imgTensor(tensorflow::DT_FLOAT, tensorflow::TensorShape({1, 1, height, width}));

    float *p = imgTensor.flat<float>().data();
    Mat fakeMat(input.rows, input.cols, CV_32FC1, p);
    input.convertTo(fakeMat, CV_32FC1);

    return imgTensor;
}

/** Convert Mat image into tensor of shape (1, height, width, d) where last three dims are equal to the original dims.
 */
Status readTensorFromMat(const Mat &mat, Tensor &outTensor) {

    auto root = tensorflow::Scope::NewRootScope();
    using namespace ::tensorflow::ops;

    // Trick from https://github.com/tensorflow/tensorflow/issues/8033
    float *p = outTensor.flat<float>().data();
    Mat fakeMat(mat.rows, mat.cols, CV_32FC3, p);
    mat.convertTo(fakeMat, CV_32FC3);

    auto input_tensor = Placeholder(root.WithOpName("input"), tensorflow::DT_FLOAT);
    vector<pair<string, tensorflow::Tensor>> inputs = {{"input", outTensor}};
    auto uint8Caster = Cast(root.WithOpName("uint8_Cast"), outTensor, tensorflow::DT_UINT8);

    // This runs the GraphDef network definition that we've just constructed, and
    // returns the results in the output outTensor.
    tensorflow::GraphDef graph;
    TF_RETURN_IF_ERROR(root.ToGraphDef(&graph));

    vector<Tensor> outTensors;
    unique_ptr<tensorflow::Session> session(tensorflow::NewSession(tensorflow::SessionOptions()));

    TF_RETURN_IF_ERROR(session->Create(graph));
    TF_RETURN_IF_ERROR(session->Run({inputs}, {"uint8_Cast"}, {}, &outTensors));

    outTensor = outTensors.at(0);
    return Status::OK();
}

/** Draw bounding box and add caption to the image.
 *  Boolean flag _scaled_ shows if the passed coordinates are in relative units (true by default in tensorflow detection)
 */
void drawBoundingBoxOnImage(Mat &image, double yMin, double xMin, double yMax, double xMax, double score, string label, bool scaled=true) {
    cv::Point tl, br;
    if (scaled) {
        tl = cv::Point((int) (xMin * image.cols), (int) (yMin * image.rows));
        br = cv::Point((int) (xMax * image.cols), (int) (yMax * image.rows));
    } else {
        tl = cv::Point((int) xMin, (int) yMin);
        br = cv::Point((int) xMax, (int) yMax);
    }
    cv::rectangle(image, tl, br, cv::Scalar(0, 255, 255), 1);

    // Ceiling the score down to 3 decimals (weird!)
    float scoreRounded = floorf(score * 1000) / 1000;
    string scoreString = to_string(scoreRounded).substr(0, 5);
    string caption = label + " (" + scoreString + ")";

    // Adding caption of type "LABEL (X.XXX)" to the top-left corner of the bounding box
    int fontCoeff = 12;
    cv::Point brRect = cv::Point(tl.x + caption.length() * fontCoeff / 1.6, tl.y + fontCoeff);
    cv::rectangle(image, tl, brRect, cv::Scalar(0, 255, 255), -1);
    cv::Point textCorner = cv::Point(tl.x, tl.y + fontCoeff * 0.9);
    cv::putText(image, caption, textCorner, FONT_HERSHEY_SIMPLEX, 0.4, cv::Scalar(255, 0, 0));
}

/** Draw bounding boxes and add captions to the image.
 *  Box is drawn only if corresponding score is higher than the _threshold_.
 */
void drawBoundingBoxesOnImage(Mat &image,
                              tensorflow::TTypes<float>::Flat &scores,
                              tensorflow::TTypes<float>::Flat &classes,
                              tensorflow::TTypes<float,3>::Tensor &boxes,
                              map<int, string> &labelsMap,
                              vector<size_t> &idxs) {
    for (int j = 0; j < idxs.size(); j++)
        drawBoundingBoxOnImage(image,
                               boxes(0,idxs.at(j),0), boxes(0,idxs.at(j),1),
                               boxes(0,idxs.at(j),2), boxes(0,idxs.at(j),3),
                               scores(idxs.at(j)), labelsMap[classes(idxs.at(j))]);
}

/** Calculate intersection-over-union (IOU) for two given bbox Rects.
 */
double IOU(Rect2f box1, Rect2f box2) {

    float xA = max(box1.tl().x, box2.tl().x);
    float yA = max(box1.tl().y, box2.tl().y);
    float xB = min(box1.br().x, box2.br().x);
    float yB = min(box1.br().y, box2.br().y);

    float intersectArea = abs((xB - xA) * (yB - yA));
    float unionArea = abs(box1.area()) + abs(box2.area()) - intersectArea;

    return 1. * intersectArea / unionArea;
}

/** Return idxs of good boxes (ones with highest confidence score (>= thresholdScore)
 *  and IOU <= thresholdIOU with others).
 */
vector<size_t> filterBoxes(tensorflow::TTypes<float>::Flat &scores,
                           tensorflow::TTypes<float, 3>::Tensor &boxes,
                           double thresholdIOU, double thresholdScore) {

    vector<size_t> sortIdxs(scores.size());
    iota(sortIdxs.begin(), sortIdxs.end(), 0);

    // Create set of "bad" idxs
    set<size_t> badIdxs = set<size_t>();
    size_t i = 0;
    while (i < sortIdxs.size()) {
        if (scores(sortIdxs.at(i)) < thresholdScore)
            badIdxs.insert(sortIdxs[i]);
        if (badIdxs.find(sortIdxs.at(i)) != badIdxs.end()) {
            i++;
            continue;
        }

        Rect2f box1 = Rect2f(Point2f(boxes(0, sortIdxs.at(i), 1), boxes(0, sortIdxs.at(i), 0)),
                             Point2f(boxes(0, sortIdxs.at(i), 3), boxes(0, sortIdxs.at(i), 2)));
        for (size_t j = i + 1; j < sortIdxs.size(); j++) {
            if (scores(sortIdxs.at(j)) < thresholdScore) {
                badIdxs.insert(sortIdxs[j]);
                continue;
            }
            Rect2f box2 = Rect2f(Point2f(boxes(0, sortIdxs.at(j), 1), boxes(0, sortIdxs.at(j), 0)),
                                 Point2f(boxes(0, sortIdxs.at(j), 3), boxes(0, sortIdxs.at(j), 2)));
            if (IOU(box1, box2) > thresholdIOU)
                badIdxs.insert(sortIdxs[j]);
        }
        i++;
    }

    // Prepare "good" idxs for return
    vector<size_t> goodIdxs = vector<size_t>();
    for (auto it = sortIdxs.begin(); it != sortIdxs.end(); it++)
        if (badIdxs.find(sortIdxs.at(*it)) == badIdxs.end())
            goodIdxs.push_back(*it);

    return goodIdxs;
}

