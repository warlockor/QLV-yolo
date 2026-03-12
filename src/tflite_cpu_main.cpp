#include <opencv2/opencv.hpp>

#include <tensorflow/lite/c/common.h>
#include <tensorflow/lite/interpreter.h>
#include <tensorflow/lite/interpreter_builder.h>
#include <tensorflow/lite/kernels/register.h>
#include <tensorflow/lite/model.h>

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <exception>
#include <iostream>
#include <limits>
#include <numeric>
#include <stdexcept>
#include <string>
#include <vector>

namespace
{
struct Detection
{
    cv::Rect box;
    int classId;
    float score;
    cv::Mat mask;
};

struct LetterboxResult
{
    cv::Mat image;
    float scale = 1.0f;
    int padW = 0;
    int padH = 0;
    int resizedW = 0;
    int resizedH = 0;
};

float Sigmoid(float x)
{
    return 1.0f / (1.0f + std::exp(-x));
}

LetterboxResult Letterbox(const cv::Mat& src, int dstW, int dstH)
{
    const float r = std::min(static_cast<float>(dstW) / static_cast<float>(src.cols),
                             static_cast<float>(dstH) / static_cast<float>(src.rows));
    const int newW = static_cast<int>(std::round(src.cols * r));
    const int newH = static_cast<int>(std::round(src.rows * r));
    cv::Mat resized;
    cv::resize(src, resized, cv::Size(newW, newH));

    const int padW = dstW - newW;
    const int padH = dstH - newH;
    const int left = padW / 2;
    const int right = padW - left;
    const int top = padH / 2;
    const int bottom = padH - top;

    cv::Mat output;
    cv::copyMakeBorder(resized, output, top, bottom, left, right, cv::BORDER_CONSTANT, cv::Scalar(114, 114, 114));
    return {output, r, left, top, newW, newH};
}

float IoU(const cv::Rect& a, const cv::Rect& b)
{
    const int x1 = std::max(a.x, b.x);
    const int y1 = std::max(a.y, b.y);
    const int x2 = std::min(a.x + a.width, b.x + b.width);
    const int y2 = std::min(a.y + a.height, b.y + b.height);
    const int w = std::max(0, x2 - x1);
    const int h = std::max(0, y2 - y1);
    const float inter = static_cast<float>(w * h);
    const float uni = static_cast<float>(a.area() + b.area()) - inter;
    return uni > 0.0f ? inter / uni : 0.0f;
}

std::vector<int> Nms(const std::vector<Detection>& dets, float iouThresh)
{
    std::vector<int> idx(dets.size());
    std::iota(idx.begin(), idx.end(), 0);
    std::sort(idx.begin(), idx.end(), [&](int i, int j) { return dets[i].score > dets[j].score; });

    std::vector<int> keep;
    std::vector<bool> removed(dets.size(), false);
    for (size_t m = 0; m < idx.size(); ++m)
    {
        const int i = idx[m];
        if (removed[i])
        {
            continue;
        }
        keep.push_back(i);
        for (size_t n = m + 1; n < idx.size(); ++n)
        {
            const int j = idx[n];
            if (removed[j] || dets[i].classId != dets[j].classId)
            {
                continue;
            }
            if (IoU(dets[i].box, dets[j].box) > iouThresh)
            {
                removed[j] = true;
            }
        }
    }
    return keep;
}

float Dequantize(int32_t q, float scale, int32_t zeroPoint)
{
    return (static_cast<float>(q) - static_cast<float>(zeroPoint)) * scale;
}

template <typename T>
T Quantize(float realValue, float scale, int32_t zeroPoint)
{
    const float q = std::round(realValue / scale) + static_cast<float>(zeroPoint);
    const float lo = static_cast<float>(std::numeric_limits<T>::min());
    const float hi = static_cast<float>(std::numeric_limits<T>::max());
    return static_cast<T>(std::max(lo, std::min(hi, q)));
}

cv::Mat SigmoidMat(const cv::Mat& src)
{
    cv::Mat expNeg;
    cv::exp(-src, expNeg);
    return 1.0f / (1.0f + expNeg);
}

void DrawDetections(cv::Mat& image, const std::vector<Detection>& detections)
{
    for (const auto& d : detections)
    {
        if (!d.mask.empty())
        {
            cv::Mat overlay = image.clone();
            overlay.setTo(cv::Scalar(0, 0, 255), d.mask);
            cv::addWeighted(overlay, 0.25, image, 0.75, 0.0, image);
        }
        cv::rectangle(image, d.box, cv::Scalar(0, 255, 0), 2);
        const std::string text =
            "cls=" + std::to_string(d.classId) + " score=" + cv::format("%.2f", static_cast<double>(d.score));
        int baseline = 0;
        cv::Size ts = cv::getTextSize(text, cv::FONT_HERSHEY_SIMPLEX, 0.5, 1, &baseline);
        const int y = std::max(0, d.box.y - ts.height - 4);
        cv::rectangle(image, cv::Rect(d.box.x, y, ts.width + 4, ts.height + 4), cv::Scalar(0, 255, 0), -1);
        cv::putText(image, text, cv::Point(d.box.x + 2, y + ts.height), cv::FONT_HERSHEY_SIMPLEX, 0.5,
                    cv::Scalar(0, 0, 0), 1);
    }
}

class YoloTfliteCpu
{
public:
    YoloTfliteCpu(const std::string& modelPath, int numClasses, float confThresh, float iouThresh)
        : m_numClasses(numClasses), m_confThresh(confThresh), m_iouThresh(iouThresh)
    {
        m_model = tflite::FlatBufferModel::BuildFromFile(modelPath.c_str());
        if (!m_model)
        {
            throw std::runtime_error("Failed to load .tflite model.");
        }

        tflite::ops::builtin::BuiltinOpResolver resolver;
        tflite::InterpreterBuilder builder(*m_model, resolver);
        builder(&m_interpreter);
        if (!m_interpreter)
        {
            throw std::runtime_error("Failed to create TFLite interpreter.");
        }

        m_interpreter->SetNumThreads(4);
        if (m_interpreter->AllocateTensors() != kTfLiteOk)
        {
            throw std::runtime_error("TFLite AllocateTensors failed.");
        }

        if (m_interpreter->inputs().empty())
        {
            throw std::runtime_error("No TFLite input tensor.");
        }

        m_inputIndex = m_interpreter->inputs()[0];
        const TfLiteTensor* inputTensor = m_interpreter->tensor(m_inputIndex);
        if (inputTensor->dims->size != 4)
        {
            throw std::runtime_error("Expected NHWC 4D input tensor.");
        }
        m_inputH = inputTensor->dims->data[1];
        m_inputW = inputTensor->dims->data[2];
        m_inputC = inputTensor->dims->data[3];
        if (m_inputC != 3)
        {
            throw std::runtime_error("Expected 3-channel input.");
        }
    }

    std::vector<Detection> Infer(const cv::Mat& bgr)
    {
        LetterboxResult lb = Letterbox(bgr, m_inputW, m_inputH);
        cv::Mat rgb;
        cv::cvtColor(lb.image, rgb, cv::COLOR_BGR2RGB);
        FillInput(rgb);

        if (m_interpreter->Invoke() != kTfLiteOk)
        {
            throw std::runtime_error("TFLite Invoke failed.");
        }

        return DecodeOutputs(bgr.size(), lb);
    }

private:
    void FillInput(const cv::Mat& rgb)
    {
        TfLiteTensor* input = m_interpreter->tensor(m_inputIndex);
        if (input->type == kTfLiteInt8)
        {
            int8_t* ptr = m_interpreter->typed_tensor<int8_t>(m_inputIndex);
            size_t k = 0;
            for (int y = 0; y < rgb.rows; ++y)
            {
                const cv::Vec3b* row = rgb.ptr<cv::Vec3b>(y);
                for (int x = 0; x < rgb.cols; ++x)
                {
                    for (int c = 0; c < 3; ++c)
                    {
                        const float v = static_cast<float>(row[x][c]) / 255.0f;
                        ptr[k++] = Quantize<int8_t>(v, input->params.scale, input->params.zero_point);
                    }
                }
            }
            return;
        }
        if (input->type == kTfLiteUInt8)
        {
            uint8_t* ptr = m_interpreter->typed_tensor<uint8_t>(m_inputIndex);
            size_t k = 0;
            for (int y = 0; y < rgb.rows; ++y)
            {
                const cv::Vec3b* row = rgb.ptr<cv::Vec3b>(y);
                for (int x = 0; x < rgb.cols; ++x)
                {
                    for (int c = 0; c < 3; ++c)
                    {
                        const float v = static_cast<float>(row[x][c]) / 255.0f;
                        ptr[k++] = Quantize<uint8_t>(v, input->params.scale, input->params.zero_point);
                    }
                }
            }
            return;
        }
        if (input->type == kTfLiteFloat32)
        {
            float* ptr = m_interpreter->typed_tensor<float>(m_inputIndex);
            size_t k = 0;
            for (int y = 0; y < rgb.rows; ++y)
            {
                const cv::Vec3b* row = rgb.ptr<cv::Vec3b>(y);
                for (int x = 0; x < rgb.cols; ++x)
                {
                    for (int c = 0; c < 3; ++c)
                    {
                        ptr[k++] = static_cast<float>(row[x][c]) / 255.0f;
                    }
                }
            }
            return;
        }
        throw std::runtime_error("Unsupported TFLite input tensor type.");
    }

    std::vector<float> OutputToFloat(const TfLiteTensor* t, int outputIndex) const
    {
        const size_t n = static_cast<size_t>(t->bytes) /
                         ((t->type == kTfLiteFloat32) ? sizeof(float) : sizeof(int8_t));
        std::vector<float> out;
        if (t->type == kTfLiteFloat32)
        {
            const float* p = m_interpreter->typed_output_tensor<float>(outputIndex);
            out.assign(p, p + n);
        }
        else if (t->type == kTfLiteInt8)
        {
            const int8_t* p = m_interpreter->typed_output_tensor<int8_t>(outputIndex);
            out.resize(n);
            for (size_t i = 0; i < n; ++i)
            {
                out[i] = Dequantize(static_cast<int32_t>(p[i]), t->params.scale, t->params.zero_point);
            }
        }
        else if (t->type == kTfLiteUInt8)
        {
            const uint8_t* p = m_interpreter->typed_output_tensor<uint8_t>(outputIndex);
            out.resize(n);
            for (size_t i = 0; i < n; ++i)
            {
                out[i] = Dequantize(static_cast<int32_t>(p[i]), t->params.scale, t->params.zero_point);
            }
        }
        else
        {
            throw std::runtime_error("Unsupported TFLite output tensor type.");
        }
        return out;
    }

    std::vector<Detection> DecodeOutputs(const cv::Size& origSize, const LetterboxResult& lb) const
    {
        struct TensorData
        {
            std::vector<int> shape;
            std::vector<float> data;
        };

        std::vector<TensorData> outs;
        for (int outTensorIndex : m_interpreter->outputs())
        {
            const TfLiteTensor* t = m_interpreter->tensor(outTensorIndex);
            TensorData td;
            for (int i = 0; i < t->dims->size; ++i)
            {
                td.shape.push_back(t->dims->data[i]);
            }
            td.data = OutputToFloat(t, outTensorIndex);
            outs.push_back(std::move(td));
        }
        if (outs.empty())
        {
            return {};
        }

        int detIdx = -1;
        int protoIdx = -1;
        for (size_t i = 0; i < outs.size(); ++i)
        {
            if (outs[i].shape.size() == 4)
            {
                protoIdx = static_cast<int>(i);
            }
            else if (outs[i].shape.size() >= 2)
            {
                detIdx = static_cast<int>(i);
            }
        }
        if (detIdx < 0)
        {
            return {};
        }

        const auto& det = outs[static_cast<size_t>(detIdx)];
        size_t rows = 0;
        size_t cols = 0;
        bool transposed = false;
        if (det.shape.size() == 3)
        {
            const size_t a = static_cast<size_t>(det.shape[1]);
            const size_t b = static_cast<size_t>(det.shape[2]);
            if (a >= b)
            {
                rows = a;
                cols = b;
            }
            else
            {
                rows = b;
                cols = a;
                transposed = true;
            }
        }
        else
        {
            cols = static_cast<size_t>(det.shape.back());
            rows = det.data.size() / std::max<size_t>(1, cols);
        }

        auto detAt = [&](size_t r, size_t c) -> float {
            if (!transposed)
            {
                return det.data[r * cols + c];
            }
            return det.data[c * rows + r];
        };

        int maskDim = static_cast<int>(cols) - (4 + m_numClasses);
        bool hasObj = false;
        if (maskDim <= 0 || maskDim > 256)
        {
            const int alt = static_cast<int>(cols) - (5 + m_numClasses);
            if (alt > 0 && alt <= 256)
            {
                maskDim = alt;
                hasObj = true;
            }
            else
            {
                maskDim = 0;
            }
        }
        const int clsOffset = hasObj ? 5 : 4;
        const int maskOffset = clsOffset + m_numClasses;

        std::vector<cv::Mat> protoChannels;
        int protoW = 0;
        int protoH = 0;
        if (protoIdx >= 0 && maskDim > 0)
        {
            const auto& proto = outs[static_cast<size_t>(protoIdx)];
            if (proto.shape.size() == 4)
            {
                const int d1 = proto.shape[1];
                const int d2 = proto.shape[2];
                const int d3 = proto.shape[3];
                bool nhwc = (d3 == maskDim);
                if (nhwc)
                {
                    protoH = d1;
                    protoW = d2;
                    protoChannels.assign(maskDim, cv::Mat(protoH, protoW, CV_32F));
                    for (int y = 0; y < protoH; ++y)
                    {
                        for (int x = 0; x < protoW; ++x)
                        {
                            for (int c = 0; c < maskDim; ++c)
                            {
                                const size_t idx = (static_cast<size_t>(y) * protoW + x) * maskDim + c;
                                protoChannels[c].at<float>(y, x) = proto.data[idx];
                            }
                        }
                    }
                }
                else if (d1 == maskDim)
                {
                    protoH = d2;
                    protoW = d3;
                    protoChannels.assign(maskDim, cv::Mat(protoH, protoW, CV_32F));
                    for (int c = 0; c < maskDim; ++c)
                    {
                        for (int y = 0; y < protoH; ++y)
                        {
                            for (int x = 0; x < protoW; ++x)
                            {
                                const size_t idx =
                                    (static_cast<size_t>(c) * protoH + static_cast<size_t>(y)) * protoW + x;
                                protoChannels[c].at<float>(y, x) = proto.data[idx];
                            }
                        }
                    }
                }
            }
        }

        std::vector<Detection> dets;
        std::vector<std::vector<float>> coeffs;
        for (size_t i = 0; i < rows; ++i)
        {
            const float cx = detAt(i, 0);
            const float cy = detAt(i, 1);
            const float w = detAt(i, 2);
            const float h = detAt(i, 3);
            const float obj = hasObj ? Sigmoid(detAt(i, 4)) : 1.0f;

            int bestCls = -1;
            float bestClsProb = 0.0f;
            for (int c = 0; c < m_numClasses && (clsOffset + c) < static_cast<int>(cols); ++c)
            {
                const float p = Sigmoid(detAt(i, static_cast<size_t>(clsOffset + c)));
                if (p > bestClsProb)
                {
                    bestClsProb = p;
                    bestCls = c;
                }
            }
            const float score = obj * bestClsProb;
            if (bestCls < 0 || score < m_confThresh)
            {
                continue;
            }

            float x1 = cx - w * 0.5f;
            float y1 = cy - h * 0.5f;
            float x2 = cx + w * 0.5f;
            float y2 = cy + h * 0.5f;
            if (x2 <= 2.0f && y2 <= 2.0f)
            {
                x1 *= static_cast<float>(m_inputW);
                x2 *= static_cast<float>(m_inputW);
                y1 *= static_cast<float>(m_inputH);
                y2 *= static_cast<float>(m_inputH);
            }
            x1 = (x1 - static_cast<float>(lb.padW)) / lb.scale;
            x2 = (x2 - static_cast<float>(lb.padW)) / lb.scale;
            y1 = (y1 - static_cast<float>(lb.padH)) / lb.scale;
            y2 = (y2 - static_cast<float>(lb.padH)) / lb.scale;

            x1 = std::max(0.0f, std::min(x1, static_cast<float>(origSize.width - 1)));
            y1 = std::max(0.0f, std::min(y1, static_cast<float>(origSize.height - 1)));
            x2 = std::max(0.0f, std::min(x2, static_cast<float>(origSize.width - 1)));
            y2 = std::max(0.0f, std::min(y2, static_cast<float>(origSize.height - 1)));

            const int bx = static_cast<int>(std::round(x1));
            const int by = static_cast<int>(std::round(y1));
            const int bw = static_cast<int>(std::round(std::max(0.0f, x2 - x1)));
            const int bh = static_cast<int>(std::round(std::max(0.0f, y2 - y1)));
            if (bw <= 1 || bh <= 1)
            {
                continue;
            }

            Detection d;
            d.box = cv::Rect(bx, by, bw, bh);
            d.classId = bestCls;
            d.score = score;
            dets.push_back(d);

            if (!protoChannels.empty() && maskDim > 0)
            {
                std::vector<float> cf(maskDim, 0.0f);
                for (int k = 0; k < maskDim && (maskOffset + k) < static_cast<int>(cols); ++k)
                {
                    cf[k] = detAt(i, static_cast<size_t>(maskOffset + k));
                }
                coeffs.push_back(std::move(cf));
            }
            else
            {
                coeffs.emplace_back();
            }
        }

        const std::vector<int> keep = Nms(dets, m_iouThresh);
        std::vector<Detection> filtered;
        filtered.reserve(keep.size());
        for (int i : keep)
        {
            Detection d = dets[i];
            if (!protoChannels.empty() && !coeffs[i].empty())
            {
                cv::Mat mask = cv::Mat::zeros(protoH, protoW, CV_32F);
                for (int k = 0; k < maskDim; ++k)
                {
                    mask += coeffs[i][k] * protoChannels[k];
                }
                mask = SigmoidMat(mask);

                cv::Mat maskInput;
                cv::resize(mask, maskInput, cv::Size(m_inputW, m_inputH), 0.0, 0.0, cv::INTER_LINEAR);
                const cv::Rect cropRect(lb.padW, lb.padH, std::max(1, lb.resizedW), std::max(1, lb.resizedH));
                cv::Mat maskCrop = maskInput(cropRect).clone();
                cv::Mat maskOrig;
                cv::resize(maskCrop, maskOrig, origSize, 0.0, 0.0, cv::INTER_LINEAR);
                cv::Mat maskBin;
                cv::threshold(maskOrig, maskBin, 0.5, 255.0, cv::THRESH_BINARY);
                maskBin.convertTo(d.mask, CV_8U);
            }
            filtered.push_back(std::move(d));
        }
        return filtered;
    }

private:
    int m_numClasses = 80;
    float m_confThresh = 0.25f;
    float m_iouThresh = 0.45f;
    int m_inputW = 640;
    int m_inputH = 640;
    int m_inputC = 3;

    std::unique_ptr<tflite::FlatBufferModel> m_model;
    std::unique_ptr<tflite::Interpreter> m_interpreter;
    int m_inputIndex = -1;
};

void PrintUsage(const char* argv0)
{
    std::cout << "Usage:\n"
              << argv0
              << " --model yolo26n_int8.tflite --image test.jpg --output result.jpg [--classes 80] [--conf 0.25] [--iou 0.45]\n";
}
}  // namespace

int main(int argc, char** argv)
{
    try
    {
        std::string modelPath;
        std::string imagePath;
        std::string outputPath = "result.jpg";
        int numClasses = 80;
        float confThresh = 0.25f;
        float iouThresh = 0.45f;

        for (int i = 1; i < argc; ++i)
        {
            const std::string arg = argv[i];
            auto next = [&](const std::string& name) -> std::string {
                if (i + 1 >= argc)
                {
                    throw std::runtime_error("Missing value for argument: " + name);
                }
                return argv[++i];
            };
            if (arg == "--model")
            {
                modelPath = next("--model");
            }
            else if (arg == "--image")
            {
                imagePath = next("--image");
            }
            else if (arg == "--output")
            {
                outputPath = next("--output");
            }
            else if (arg == "--classes")
            {
                numClasses = std::stoi(next("--classes"));
            }
            else if (arg == "--conf")
            {
                confThresh = std::stof(next("--conf"));
            }
            else if (arg == "--iou")
            {
                iouThresh = std::stof(next("--iou"));
            }
            else if (arg == "-h" || arg == "--help")
            {
                PrintUsage(argv[0]);
                return 0;
            }
            else
            {
                throw std::runtime_error("Unknown argument: " + arg);
            }
        }

        if (modelPath.empty() || imagePath.empty())
        {
            PrintUsage(argv[0]);
            return 1;
        }

        cv::Mat image = cv::imread(imagePath);
        if (image.empty())
        {
            throw std::runtime_error("Failed to read image: " + imagePath);
        }

        YoloTfliteCpu runner(modelPath, numClasses, confThresh, iouThresh);
        const std::vector<Detection> detections = runner.Infer(image);
        DrawDetections(image, detections);

        if (!cv::imwrite(outputPath, image))
        {
            throw std::runtime_error("Failed to save output image: " + outputPath);
        }

        std::cout << "TFLite CPU inference done. Detections: " << detections.size() << "\n";
        std::cout << "Saved output: " << outputPath << "\n";
        return 0;
    }
    catch (const std::exception& e)
    {
        std::cerr << "[ERROR] " << e.what() << "\n";
        return 1;
    }
}
