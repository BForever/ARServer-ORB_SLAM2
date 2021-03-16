#include "use.h"

int main(int argc,char * argv[]){

    //at::init_num_threads();//***zh,好像加上这句，速度有一点点提升

    string modelpath = "yolov5s.torchscript";
    long start = time_in_ms();
    torch::jit::script::Module model = torch::jit::load(modelpath);
    //long end = ;
    cout << "it took " << time_in_ms() - start << " ms to load the model" << endl;
    torch::jit::getProfilingMode() = false;
    torch::jit::getExecutorMode() = false;
    torch::jit::setGraphExecutorOptimize(false);


    string img_path = "test.png";
    Mat img = imread(img_path);
    Mat im0 = imread(img_path);
    img = letterbox(img);//zh,,resize

    cvtColor(img, img, CV_BGR2RGB);//***zh, bgr->rgb
    start = time_in_ms();
    img.convertTo(img, CV_32FC3, 1.0f / 255.0f);//zh, 1/255
    auto tensor_img = torch::from_blob(img.data, {img.rows, img.cols, img.channels()});
    tensor_img = tensor_img.permute({2, 0, 1});
    tensor_img = tensor_img.unsqueeze(0);
    std::vector<torch::jit::IValue> inputs;
    inputs.push_back(tensor_img);
    //end = ;
    cout << "it took " << time_in_ms() - start << " ms to preprocess image" << endl;
    start = time_in_ms();
    torch::jit::IValue output = model.forward(inputs);
    //end = time_in_ms();
    cout << "it took " << time_in_ms() - start << " ms to model forward" << endl;
    at::Tensor op = output.toTuple()->elements().at(0).toTensor();
    op = op.view({-1 ,op.sizes()[2] });

    ifstream f;
    std::vector<string> labels;
    string labelpath = "labels.txt";
    f.open(labelpath);
    string str;
    while (std::getline(f,str)){
        labels.push_back(str);
    }
    start = time_in_ms();
    op = non_max_suppression(op, labels, 0.4, 0.5, true,  false);
    cout << "it took " << time_in_ms() - start << " ms to non_max_suppression" << endl;
    try{
        at::Tensor temp  = op.index({0});
    }catch(...){
        cout << "no objects, line 401 " << endl;
        return -1;
    }


    for( int i = 0; i < op.sizes()[0]; i++ ){

        cout << "start of object " << i << endl;

        at::Tensor opi = op.index({i});

        int op_class = opi.index({5}).item().toInt();
        cout << " op_class is " << op_class << ", and it's " << labels[op_class] << endl;

        int x1 = opi.index({0}).item().toInt();
        int y1 = opi.index({1}).item().toInt();

        int x2 = opi.index({2}).item().toInt();
        int y2 = opi.index({3}).item().toInt();

        cout << " the bounding box is x1 = " << x1 << ", and y1 = " << y1 << ", and x2 = " << x2 << ", and y2 = " << y2 << endl;

        cv::rectangle(im0, Point(x1,y1), Point(x2,y2), Scalar(0,0,255), 2, 8, 0);
        cv::putText(im0,labels[op_class],Point(x1,y1),1,1.0,cv::Scalar(0,255,0),1);
    }


    imshow("test image", im0);
    waitKey();

    return 0;
}
