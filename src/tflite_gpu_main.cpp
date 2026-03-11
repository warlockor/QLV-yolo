#include <opencv2/opencv.hpp>

#include <tensorflow/lite/c/common.h>
#include <tensorflow/lite/interpreter.h>
#include <tensorflow/lite/interpreter_builder.h>
#include <tensorflow/lite/kernels/register.h>
#include <tensorflow/lite/model.h>

#if __has_include(<tensorflow/lite/delegates/gpu/delegate.h>)
#include <tensorflow/lite/delegates/gpu/delegate.h>
#define HAS_TFLITE_GPU_DELEGATE 1
#else
#define HAS_TFLITE_GPU_DELEGATE 0
#endif

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
};

struct LetterboxResult
{
    cv::Mat image;
    float scale = 1.0f;
    int padW = 0;
    int padH = 0;
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
    return {output, r, left, top};
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

void DrawDetections(cv::Mat& image, const std::vector<Detection>& detections)
{
    for (const auto& d : detections)
    {
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

class YoloTfliteGpu
{
public:
    YoloTfliteGpu(const std::string& modelPath, int numClasses, float confThresh, float iouThresh)
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

#if HAS_TFLITE_GPU_DELEGATE
        TfLiteGpuDelegateOptionsV2 options = TfLiteGpuDelegateOptionsV2Default();
        options.inference_preference = TFLITE_GPU_INFERENCE_PREFERENCE_SUSTAINED_SPEED;
        options.inference_priority1 = TFLITE_GPU_INFERENCE_PRIORITY_MIN_LATENCY;
        options.inference_priority2 = TFLITE_GPU_INFERENCE_PRIORITY_MAX_PRECISION;
        options.inference_priority3 = TFLITE_GPU_INFERENCE_PRIORITY_AUTO;
        m_gpuDelegate = TfLiteGpuDelegateV2Create(&options);
        if (!m_gpuDelegate)
        {
            throw std::runtime_error("Failed to create TFLite GPU delegate.");
        }
        if (m_interpreter->ModifyGraphWithDelegate(m_gpuDelegate) != kTfLiteOk)
        {
            throw std::runtime_error("ModifyGraphWithDelegate failed. GPU delegate may not be supported on this Linux ARM target.");
        }
#else
        throw std::runtime_error("TFLite GPU delegate header not found at build time.");
#endif

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
    }

    ~YoloTfliteGpu()
    {
#if HAS_TFLITE_GPU_DELEGATE
        if (m_gpuDelegate)
        {
            TfLiteGpuDelegateV2Delete(m_gpuDelegate);
            m_gpuDelegate = nullptr;
        }
#endif
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
        std::vector<Detection> dets;
        for (int outTensorIndex : m_interpreter->outputs())
        {
            const TfLiteTensor* t = m_interpreter->tensor(outTensorIndex);
            if (t->dims->size < 2)
            {
                continue;
            }
            const int lastDim = t->dims->data[t->dims->size - 1];
            if (lastDim < 6)
            {
                continue;
            }

            const std::vector<float> out = OutputToFloat(t, outTensorIndex);
            const size_t rows = out.size() / static_cast<size_t>(lastDim);
            for (size_t i = 0; i < rows; ++i)
            {
                const size_t base = i * static_cast<size_t>(lastDim);
                const float cx = out[base + 0];
                const float cy = out[base + 1];
                const float w = out[base + 2];
                const float h = out[base + 3];
                const float obj = Sigmoid(out[base + 4]);

                int bestCls = -1;
                float bestClsProb = 0.0f;
                const int clsCount = std::min(m_numClasses, lastDim - 5);
                for (int c = 0; c < clsCount; ++c)
                {
                    const float p = Sigmoid(out[base + 5 + c]);
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
                if (bw > 1 && bh > 1)
                {
                    dets.push_back({cv::Rect(bx, by, bw, bh), bestCls, score});
                }
            }
        }

        const std::vector<int> keep = Nms(dets, m_iouThresh);
        std::vector<Detection> filtered;
        filtered.reserve(keep.size());
        for (int i : keep)
        {
            filtered.push_back(dets[i]);
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
    TfLiteDelegate* m_gpuDelegate = nullptr;
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

        YoloTfliteGpu runner(modelPath, numClasses, confThresh, iouThresh);
        const std::vector<Detection> detections = runner.Infer(image);
        DrawDetections(image, detections);

        if (!cv::imwrite(outputPath, image))
        {
            throw std::runtime_error("Failed to save output image: " + outputPath);
        }

        std::cout << "TFLite GPU delegate inference done. Detections: " << detections.size() << "\n";
        std::cout << "Saved output: " << outputPath << "\n";
        return 0;
    }
    catch (const std::exception& e)
    {
        std::cerr << "[ERROR] " << e.what() << "\n";
        return 1;
    }
}
