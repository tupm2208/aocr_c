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

int main(int argc, char* argv[]) {


    // Set dirs variables
    string ROOTDIR = "/home/tupm/HDD/projects/tensorflow-object-detection-cpp/assets/";
    string GRAPH = "text_models/general_binary_model.pb";

    // Set input & output nodes names
    string inputLayer = "img_data";
    vector<string> outputLayer = {"Squeeze"};

    std::unique_ptr<tensorflow::Session> session;
    string graphPath = tensorflow::io::JoinPath(ROOTDIR, GRAPH);
    LOG(INFO) << "graphPath:" << graphPath;
    Status loadGraphStatus = loadGraph(graphPath, &session);
    if (!loadGraphStatus.ok()) {
        LOG(ERROR) << "loadGraph(): ERROR" << loadGraphStatus;
        return -1;
    } else
        LOG(INFO) << "loadGraph(): frozen graph loaded" << endl;

    
    Mat frame;
    Tensor imageTensor, embeddingTensor;
    std::vector<Tensor> outputs;

    // FPS count
    int nFrames = 25;
    int iFrame = 0;
    double fps = 0.;
    time_t start, end;
    time(&start);

    Mat image = imread("/home/tupm/HDD/projects/koya_screenshot_ocr/logs/20200916224955/IMG_2932/crop_images/text_box_2.jpg");
    

    imageTensor = convertMatToTensorYolo(image);

    Status runStatus = session->Run({{inputLayer, imageTensor}}, outputLayer, {}, &outputs);

    if (!runStatus.ok()){
        LOG(INFO) << "Running model failed: " << runStatus;
        return -1;
    }

    // tensorflow::TTypes<float>::Flat embeddingData = outputs[0].flat<float>();
    float* p = outputs[0].flat<float>().data();

    tensorflow::Scope transposeScope = tensorflow::Scope::NewRootScope();

    tensorflow::TensorShape embeddingShape = outputs[0].shape();
    
    int eh = embeddingShape.dim_size(0);
    int ew = embeddingShape.dim_size(1);
    int channels = embeddingShape.dim_size(2);

    auto transOps = tensorflow::ops::ConjugateTranspose(transposeScope, outputs[0], tensorflow::ops::Const(transposeScope, {1, 0, 2}));

    tensorflow::ClientSession sess(transposeScope);
    std::vector<Tensor> outputs1;
    TF_CHECK_OK(sess.Run({transOps}, &outputs1));

    auto padValue = tensorflow::ops::Const(transposeScope, {{0, 400-ew}, {0, 0}, {0, 0}});
    auto padOps = tensorflow::ops::Pad(transposeScope, outputs1[0], padValue);

    TF_CHECK_OK(sess.Run({padOps}, &outputs));

    tensorflow::TensorShape bz{1};
    Tensor batchsize(tensorflow::DT_INT32, bz);
    auto bz_map = batchsize.tensor<int32, 1>();
    bz_map(0) = 1;
    vector<string> inputLayers = {"transpose_1", "decoder0"};
    vector<Tensor> inputValues = {outputs[0], batchsize};
    

    for (int i = 0; i < 400; i++) {
        tensorflow::TensorShape mks{1, 1};
        Tensor mask(tensorflow::DT_FLOAT, mks);
        auto input_tensor_mapped = mask.tensor<float, 2>();
        input_tensor_mapped(0, 0) = i < ew? 1.0f: 0.0f;
        inputValues.push_back(mask);

        string name = "encoder_mask" + std::to_string(i);
        inputLayers.push_back(name);
    }


    vector<string> outputLayers;


    for (int l = 0; l < 42; l++){
        string tem = "";
        if (l!=0) {
            tem = "_" + to_string(l);
        }
        string layer = "model_with_buckets/embedding_attention_decoder/attention_decoder/AttnOutputProjection"+tem+"/AttnOutputProjection/BiasAdd";
        outputLayers.push_back(layer);
        
    }

    for (int l = 0; l < 42; l++){
        
        string tem = "";
        if (l!=0) {
            tem = "_" + to_string(l);
        }
        string layer = "model_with_buckets/embedding_attention_decoder/attention_decoder/Attention_0"+tem+"/Softmax";
        outputLayers.push_back(layer);
    }
  
    LOG(INFO) << outputs[0].shape();

    std::vector<std::pair<std::string, tensorflow::Tensor>> feed_dict;

    for (int i = 0; i< 402; i++) {
        feed_dict.push_back(std::pair<std::string, tensorflow::Tensor>(inputLayers[i], inputValues[i]));
    }
    vector<Tensor> finalOutputs;
    // outputs.clear();
    runStatus = session->Run(feed_dict, outputLayers, {}, &finalOutputs);

    if (!runStatus.ok()){
        LOG(INFO) << "Running model failed: " << runStatus;
        return -1;
    }

    vector<Tensor> stepLogits;
    vector<Tensor> outputStepLogits;

    float_t *plant_pointer = finalOutputs[1].flat<float_t>().data();
    // int mx = 0;
    // float cr_max = -1;
    // for (int i = 0; i < 400; i++) {
    //     cout << *(plant_pointer + i) << " ";
    //     if (cr_max < *(plant_pointer + i)) {
    //         cr_max = *(plant_pointer + i);
    //         mx = i;
    //     }
    // }
    // cout << mx << " ";
    

    for (int i = 0; i< 42; i++){
        auto argOps = tensorflow::ops::ArgMax(transposeScope, finalOutputs[i], tensorflow::ops::Const(transposeScope, 1));
        TF_CHECK_OK(sess.Run({argOps}, &stepLogits));
        outputStepLogits.push_back(stepLogits[0]);
        cout << stepLogits[0].scalar<int>() << endl;
    }

    return 0;
}