#include "InstanceDetect.h"


using namespace std;
using namespace cv;
using namespace torch::indexing;
namespace ORB_SLAM2 {
vector<string> split2(const string& str, const string& delim) {
    vector<string> res;
    if("" == str) return res;
    char * strs = new char[str.length() + 1] ;
    strcpy(strs, str.c_str());

    char * d = new char[delim.length() + 1];
    strcpy(d, delim.c_str());

    char *p = strtok(strs, d);
    while(p) {
        string s = p;
        res.push_back(s);
        p = strtok(NULL, d);
    }
    return res;
}
InstanceDetect::InstanceDetect(Map *pMap) : mpMap(pMap), mbResetRequested(false), mbFinishRequested(false),
                                            mbFinished(true) {
    string modelpath = "YOLOv5_Torchscript/files/yolov5s.torchscript";
    long start = time_in_ms();
    model = torch::jit::load(modelpath);
    cout << "it took " << time_in_ms() - start << " ms to load the model" << endl;
    torch::jit::getProfilingMode() = false;
    torch::jit::getExecutorMode() = false;
    torch::jit::setGraphExecutorOptimize(false);
    torch::set_num_threads(4);

    ifstream f;
    labelpath = "labels.txt";
    f.open(labelpath);
    string str;
    while (std::getline(f, str)) {
        labels.push_back(str);
    }
    cout << "we get " << labels.size() << " labels" << endl;
    f.close();

    //read anchor_grid
    at::Tensor anchor_grid = torch::ones({3, 1, 3, 1, 1, 2});
    ifstream fa;
    string gridpath = "anchor_grid.txt";
    fa.open(gridpath);
    while (std::getline(fa,str)){
        vector<string> mp = split2(str,",");
        anchor_grid.index_put_({stoi(mp[0]),stoi(mp[1]),stoi(mp[2]),stoi(mp[3]),stoi(mp[4]),stoi(mp[5])}, torch::ones({1}) * stof(mp[6]) );
    }
    fa.close();
}

void InstanceDetect::Run() {
    mbFinished = false;
    cout << "InstanceDetect: detection thread started" << endl;
    while (1) {
        if (CheckNewDetectFrames()) {
            cout << "InstanceDetect: get in check new frames " << endl;
            auto im = midFrameQueue.front();
            detect(im);
            midFrameQueue.pop_front();
            im->release();
        }

        //cout << "before ResetIfRequested" << endl;
        ResetIfRequested();
        //cout << "after ResetIfRequested" << endl;

        if (CheckFinish()) {
            //cout << "into CheckFinish()" << endl;
            break;
        }
        //cout << "before usleep" << endl;
        usleep(5000);//usleep(5000);
        //cout << "after usleep" << endl;
    }
    SetFinish();

}

void InstanceDetect::InsertFrame(const cv::Mat &im) {
    cv::Mat *image= new cv::Mat(im);
    unique_lock<mutex> lock(mMutexInstanceDetectQueue);
    midFrameQueue.push_back(image);
}

bool InstanceDetect::CheckNewDetectFrames() {
    unique_lock<mutex> lock(mMutexInstanceDetectQueue);
    return (!midFrameQueue.empty());
}

void InstanceDetect::ResetIfRequested() {
    unique_lock<mutex> lock(mMutexReset);
    if (mbResetRequested) {
        midFrameQueue.clear();
        mbResetRequested = false;
    }
}


void InstanceDetect::RequestFinish()//要主动调用，来结束线程。
{
    unique_lock<mutex> lock(mMutexFinish);
    mbFinishRequested = true;
}

bool InstanceDetect::CheckFinish() {
    unique_lock<mutex> lock(mMutexFinish);
    return mbFinishRequested;
}

void InstanceDetect::SetFinish() {
    unique_lock<mutex> lock(mMutexFinish);
    mbFinished = true;
}



long InstanceDetect::time_in_ms(){
    struct timeval t;
    gettimeofday(&t, NULL);
    long time_ms = ((long)t.tv_sec)*1000+(long)t.tv_usec/1000;
    return time_ms;
}

at::Tensor box_area(at::Tensor box)
{
    return (box.index({2}) - box.index({0})) * (box.index({3}) - box.index({1}));
}

at::Tensor box_iou(at::Tensor box1, at::Tensor box2)
{
    at::Tensor area1 = box_area(box1.t());
    at::Tensor area2 = box_area(box2.t());

    at::Tensor inter = ( torch::min( box1.index({Slice(), {None}, Slice(2,None)}), box2.index({Slice(), Slice(2,None)}))
                         - torch::max( box1.index({Slice(), {None}, Slice(None,2)}), box2.index({Slice(), Slice(None,2)})) ).clamp(0).prod(2);
    return inter / (area1.index({Slice(), {None}}) + area2 - inter);

}

//Convert nx4 boxes from [x, y, w, h] to [x1, y1, x2, y2] where xy1=top-left, xy2=bottom-right
at::Tensor xywh2xyxy(at::Tensor x)
{
    at::Tensor y = torch::zeros_like(x);
    y.index({Slice(), 0}) = x.index({Slice(), 0}) - x.index({Slice(), 2}) / 2; // # top left x
    y.index({Slice(), 1}) = x.index({Slice(), 1}) - x.index({Slice(), 3}) / 2;
    y.index({Slice(), 2}) = x.index({Slice(), 0}) + x.index({Slice(), 2}) / 2;
    y.index({Slice(), 3}) = x.index({Slice(), 1}) + x.index({Slice(), 3}) / 2;

    return y;
}



at::Tensor non_max_suppression(at::Tensor pred, std::vector<string> labels, float conf_thres = 0.1, float iou_thres = 0.6,
                               bool merge=false,  bool agnostic=false)
{
    int nc = pred.sizes()[1] - 5;
    at::Tensor xc = pred.index({Slice(None), 4}) > conf_thres; // # candidates,

    int min_wh = 2;
    int max_wh = 4096;
    int max_det = 300;
    float time_limit = 10.0;
    bool redundant = true;  //require redundant detections
    bool multi_label = true;
    at::Tensor output;

    at::Tensor x;
    x = pred.index({xc});

    try{
        at::Tensor temp  = x.index({0});
    }catch(...){
        cout << "no objects, line 138 " << endl;
        at::Tensor temp;
        return temp;
    }

    x.index({Slice(),Slice(5,None)}) *= x.index({Slice(), Slice(4,5)});
    at::Tensor box = xywh2xyxy(x.index({Slice(), Slice(None,4)}));
    if(true){ //here mulit label in python code
        //i, j = (x[:, 5:] > conf_thres).nonzero().t()
        auto temp = (x.index({Slice(), Slice(5,None)}) > conf_thres).nonzero().t();
        at::Tensor i = temp[0];
        at::Tensor j = temp[1];
        x = torch::cat({  box.index({i}), x.index({i,j+5,{None}}), j.index({Slice(),{None}}).toType(torch::kFloat)  }, 1);
    }

    try{
        at::Tensor temp  = x.index({0});
    }catch(...){
        cout << "no objects, line 187 " << endl;
        at::Tensor temp;
        return temp;
    }

    int n = x.sizes()[0];
    at::Tensor c = x.index({Slice(), Slice(5,6)}) * max_wh;
    at::Tensor boxes = x.index({Slice(), Slice(None, 4)}) + c;
    at::Tensor scores = x.index({Slice(),4});

    at::Tensor i = nms(boxes, scores, iou_thres);

    if(i.sizes()[0] > max_det){
        i = i.index({Slice(0,max_det)});
    }

    if(merge){
        if( n > 1 && n < 3000){
            at::Tensor iou = box_iou(boxes.index({i}), boxes) > iou_thres;
            at::Tensor weights = iou * scores.index({ {None} });
            at::Tensor temp1 = torch::mm(weights, x.index({Slice(), Slice(None, 4)})).toType(torch::kFloat);
            at::Tensor temp2 = weights.sum(1, true);
            at::Tensor tempres = temp1 / temp2;
            x.index_put_({i, Slice(None, 4)}, tempres);
            if(redundant){
                i = i.index({iou.sum(1) > 1});
            }
        }
    }
    output = x.index({i});
    return output;
}


cv::Mat letterbox(Mat img, int new_height = 640, int new_width = 640, Scalar color = (114,114,114), bool autos = true, bool scaleFill=false, bool scaleup=true){
    int width = img.cols;
    int height = img.rows;
    float rh = float(new_height) / height;
    float rw = float(new_width) / width;
    float ratio;
    if(rw < rh){
        ratio = rw;}
    else{
        ratio = rh;}
    if (!scaleup){
        if(ratio >= 1.0){
            ratio = 1.0;
        }
    }
    int new_unpad_h = int(round(height * ratio));
    int new_unpad_w = int(round(width * ratio));
    int dh = new_height - new_unpad_h;
    int dw = new_width - new_unpad_w;

    if(autos){
        dw = dw % 64;
        dh = dh % 64;
    }

    dw /= 2;
    dh /= 2;//默认被二整除吧
    if( height != new_height or width != new_width){
        resize(img, img, Size(new_unpad_w, new_unpad_h), 0, 0, cv::INTER_LINEAR);
    }
    int top = int(round(dh - 0.1));
    int bottom = int(round(dh + 0.1));
    int left = int(round(dw - 0.1));
    int right = int(round(dw + 0.1));

    cv::copyMakeBorder(img, img, top, bottom, left, right, cv::BORDER_CONSTANT, Scalar(114,114,114));

    return img;
}


at::Tensor make_grid(int nx = 20, int ny = 20){
    std::vector<at::Tensor> temp = torch::meshgrid({torch::arange(ny), torch::arange(nx)});
    at::Tensor yv = temp[0];
    at::Tensor xv = temp[1];
    at::Tensor retu = torch::stack({xv, yv},2).view({1,1,ny,nx,2}).toType(torch::kFloat);
    return retu;
}

at::Tensor clip_coords(at::Tensor boxes, auto img_shape){
    boxes.index({Slice(), 0}).clamp_(0, img_shape[1]);
    boxes.index({Slice(), 1}).clamp_(0, img_shape[0]);
    boxes.index({Slice(), 2}).clamp_(0, img_shape[1]);
    boxes.index({Slice(), 3}).clamp_(0, img_shape[0]);
    return boxes;
}

float max_shape(int shape[], int len){
    float max = -10000;
    for(int i = 0; i < len; i++){
        if (shape[i] > max){
            max = shape[i];
        }
    }
    return max;
}


at::Tensor scale_coords(int img1_shape[], at::Tensor coords, int img0_shape[]){
    float gain = max_shape(img1_shape, 2) / max_shape(img0_shape, 3);
    float padw = (img1_shape[1] - img0_shape[1] * gain ) / 2.0;
    float padh = (img1_shape[0] - img0_shape[0] * gain ) / 2.0;

    coords.index_put_({Slice(), 0}, coords.index({Slice(), 0}) - padw);
    coords.index_put_({Slice(), 2}, coords.index({Slice(), 2}) - padw);
    coords.index_put_({Slice(), 1}, coords.index({Slice(), 1}) - padh);
    coords.index_put_({Slice(), 3}, coords.index({Slice(), 3}) - padh);
    coords.index_put_({Slice(), Slice(None, 4)}, coords.index({Slice(), Slice(None, 4)}) / gain);
    clip_coords(coords, img0_shape);
    return coords;
}

void InstanceDetect::detect(cv::Mat *image) {
//    string img_path = "object detection/build/test.png";
    //string img_path = "/home/zherlock/InstanceDetection/yolov5_old/test.png";
//    Mat img = imread(img_path);
    Mat img = Mat(*image);
    int im0_shape[3] = {img.rows, img.cols, img.channels()};
//    cout<<img.rows<<img.cols<<img.channels()<<endl;

//    img = letterbox(img);//zh,,resize
    cvtColor(img, img, CV_BGR2RGB);//***zh, bgr->rgb
    long start = time_in_ms();
    img.convertTo(img, CV_32FC3, 1.0f / 255.0f);//zh, 1/255
    auto tensor_img = torch::from_blob(img.data, {img.rows, img.cols, img.channels()});
    tensor_img = tensor_img.permute({2, 0, 1});
    tensor_img = tensor_img.unsqueeze(0);
    std::vector<torch::jit::IValue> inputs;
    inputs.push_back(tensor_img);
    cout << "it took " << time_in_ms() - start << " ms to preprocess image" << endl;
    start = time_in_ms();
    torch::jit::IValue output = model.forward(inputs);
    cout << "it took " << time_in_ms() - start << " ms to model forward" << endl;


//    //Detect layer
//    //set parameters
//    int ny, nx;
//    int na = 3;int no = 85;int bs = 1;
//    at::Tensor op;
//    at::Tensor y_0, y_1, y_2, y;
//    at::Tensor grid_1, grid_2, grid_3, grid;
//    float stride[3] = {8.0, 16.0, 32.0};
//
//
//    //run
//    for(int i = 0; i < 3; i++){
//        op = output.toList().get(i).toTensor().contiguous();
//        bs = op.sizes()[0]; ny = op.sizes()[2]; nx = op.sizes()[3];
//        op = op.view({bs, na, no, ny, nx}).permute({0, 1, 3, 4, 2}).contiguous();
//        grid = make_grid(nx, ny);
//        y = op.sigmoid();
//        at::Tensor test = y.index({Slice(),Slice(), Slice(),Slice(), Slice(0, 2)}) * 2.0 - 0.5;
//        test = y.index({Slice(),Slice(), Slice(),Slice(), Slice(0, 2)}) * 2.0 - 0.5 + grid;
//        at::Tensor temp = (y.index({Slice(),Slice(), Slice(),Slice(), Slice(0, 2)}) * 2.0 - 0.5 + grid) * stride[i];
//        y.index_put_({Slice(),Slice(), Slice(),Slice(), Slice(0, 2)}, temp);
//        temp = (2.0 * y.index({Slice(),Slice(), Slice(),Slice(), Slice(2, 4)})).pow(2) * anchor_grid.index({i});
//        y.index_put_({Slice(),Slice(), Slice(),Slice(), Slice(2, 4)}, temp);
//        y = y.view({bs, -1, no});
//        if(i == 0){
//            y_0 = y;
//        }
//        else if (i == 1){
//            y_1 = y;
//        }
//        else{
//            y_2 = y;
//        }
//    }
//
//    op = torch::cat({y_0, y_1,y_2}, 1);
//    op = op.view({-1, op.sizes()[2]});
//
//    start = time_in_ms();
//    float conf = 0.3;
//    op = non_max_suppression(op, labels, conf, 0.5, true,  false);
//    cout << "it took " << time_in_ms() - start << " ms to non_max_suppression" << endl;
//    try{
//        at::Tensor temp  = op.index({0});
//    }catch(...){
//        cout << "no objects, line 401 " << endl;
//        return;
//    }
//    int img_shape[2] = {tensor_img.sizes()[2], tensor_img.sizes()[3]};
//
//    at::Tensor temp;
//    temp = scale_coords(img_shape, op.index({Slice(), Slice(None, 4)}), im0_shape).round();
//    op.index_put_({Slice(), Slice(None, 4)}, temp);
//
//    for( int i = 0; i < op.sizes()[0]; i++ ){
//        cout << "start of object " << i+1 << endl;
//        at::Tensor opi = op.index({i});
//        int op_class = opi.index({5}).item().toInt();
//        cout << " op_class is " << op_class << ", and it's " << labels[op_class] << endl;
//
//        int x1 = opi.index({0}).item().toInt();
//        int y1 = opi.index({1}).item().toInt();
//
//        int x2 = opi.index({2}).item().toInt();
//        int y2 = opi.index({3}).item().toInt();
//
//        cout << " the bounding box is x1 = " << x1 << ", and y1 = " << y1 << ", and x2 = " << x2 << ", and y2 = " << y2 << endl;
//    }

}
}
