#include <fstream>
#include <utility>
#include <vector>
#include <iostream>

#include "tensorflow/cc/ops/const_op.h"
#include "tensorflow/cc/ops/image_ops.h"
#include "tensorflow/cc/ops/standard_ops.h"
#include "tensorflow/core/framework/graph.pb.h"
#include "tensorflow/core/graph/default_device.h"
#include "tensorflow/core/graph/graph_def_builder.h"
#include "tensorflow/core/lib/core/threadpool.h"
#include "tensorflow/core/lib/io/path.h"
#include "tensorflow/core/lib/strings/stringprintf.h"
#include "tensorflow/core/platform/init_main.h"
#include "tensorflow/core/public/session.h"
#include "tensorflow/core/util/command_line_flags.h"
#include "tensorflow/cc/client/client_session.h"

#include <opencv2/core/mat.hpp>
#include <opencv2/videoio.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
// #include <cv.hpp>

#include <time.h>

#include "utils.h"

using tensorflow::Flag;
using tensorflow::Tensor;
using tensorflow::Status;
using tensorflow::string;
using tensorflow::int32;

using namespace std;
using namespace cv;

string ROOTDIR = "/home/tupm/HDD/projects/tensorflow-object-detection-cpp/assets/";
string GRAPH = "text_models/general_binary_model.pb";

std::shared_ptr<tensorflow::Session> session;
string graphPath = tensorflow::io::JoinPath(ROOTDIR, GRAPH);
Status loadingStatus = loadGraph(graphPath, &session);
vector<string> textLables = readLabels();

vector<string> orc(vector<Mat> images) {
    std::vector<Tensor> outputs;
    vector<string> inputLayers;
    vector<Tensor> inputValues;
    Status runStatus;

    Tensor embeddingTensor = cnnPart(images, session, inputValues);

    int noBatch = embeddingTensor.shape().dim_size(1);
    tensorflow::TensorShape bz{noBatch};
    Tensor batchsize(tensorflow::DT_INT32, bz);
    auto bz_map = batchsize.tensor<int32, 1>();
    bz_map(0) = noBatch;
    for (int i = 0; i < noBatch; i++) {
        bz_map(i) = 1;
    }

    // cout << inputValues.size() << " " << noBatch << endl;
    

    for (int i = 0; i < 400; i++) {
        string name = "encoder_mask" + std::to_string(i);
        inputLayers.push_back(name);
    }

    inputLayers.push_back("transpose_1");
    inputLayers.push_back("decoder0");
    inputValues.push_back(embeddingTensor);
    inputValues.push_back(batchsize);

    vector<string> outputLayers;
    for (int l = 0; l < 42; l++){
        string tem = "";
        if (l!=0) {
            tem = "_" + to_string(l);
        }
        string layer = "model_with_buckets/embedding_attention_decoder/attention_decoder/AttnOutputProjection"+tem+"/AttnOutputProjection/BiasAdd";
        outputLayers.push_back(layer);
        
    }
  
    LOG(INFO) << embeddingTensor.shape();

    std::vector<std::pair<std::string, tensorflow::Tensor>> feed_dict;

    for (int i = 0; i< 402; i++) {
        feed_dict.push_back(std::pair<std::string, tensorflow::Tensor>(inputLayers[i], inputValues[i]));
    }

    session->Run(feed_dict, outputLayers, {}, &outputs);

    vector<vector<int>> outputResult = handleFinalOutput(outputs);

    vector<string> finalString;
    for (vector<int> idxs: outputResult) {
        string tem = "";
        for (int i: idxs) {
            if (i<3) {
                break;
            }
            tem += textLables.at(i-3);
        }
        finalString.push_back(tem);

        // cout << "data: " << tem << endl;
    }
    return finalString;
}

int main(int argc, char* argv[]) {

    
    time_t start, end;
    Mat image = imread("/home/tupm/HDD/projects/koya_screenshot_ocr/logs/20200916224955/IMG_2932/crop_images/text_box_2.jpg");
    vector<Mat> images{image};
    
    for (int i=0; i< 14; i++) {
        
        time(&start);
        auto start_time = std::chrono::high_resolution_clock::now();
        orc(images);
        // cout << textLables.substr(6, 3)<< endl;
        auto end_time = std::chrono::high_resolution_clock::now();
        auto time = end_time - start_time;
        float end = time/std::chrono::milliseconds(1);
        cout << "fps: " << (float)end << endl;
    }

    return 0;
}