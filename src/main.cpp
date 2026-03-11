#include <armnn/ArmNN.hpp>
#include <armnnTfLiteParser/ITfLiteParser.hpp>
#include <opencv2/opencv.hpp>

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <exception>
#include <fstream>
#include <iostream>
#include <limits>
#include <memory>
#include <numeric>
#include <stdexcept>
#include <string>
#include <utility>
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

float IoU(const cv::Rect& a, const cv::Rect& b)
{
    const int interX1 = std::max(a.x, b.x);
    const int interY1 = std::max(a.y, b.y);
    const int interX2 = std::min(a.x + a.width, b.x + b.width);
    const int interY2 = std::min(a.y + a.height, b.y + b.height);
    const int interW = std::max(0, interX2 - interX1);
    const int interH = std::max(0, interY2 - interY1);
    const float interArea = static_cast<float>(interW * interH);
    const float unionArea = static_cast<float>(a.area() + b.area()) - interArea;
    return unionArea > 0.0f ? interArea / unionArea : 0.0f;
}

std::vector<int> Nms(const std::vector<Detection>& detections, float iouThresh)
{
    std::vector<int> idx(detections.size());
    std::iota(idx.begin(), idx.end(), 0);
    std::sort(idx.begin(), idx.end(), [&](int i, int j) { return detections[i].score > detections[j].score; });

    std::vector<int> keep;
    std::vector<bool> removed(detections.size(), false);
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
            if (removed[j] || detections[i].classId != detections[j].classId)
            {
                continue;
            }
            if (IoU(detections[i].box, detections[j].box) > iouThresh)
            {
                removed[j] = true;
            }
        }
    }
    return keep;
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

template <typename T>
T QuantizeWithClamp(float realVal, float scale, int32_t offset)
{
    if (scale <= 0.0f)
    {
        throw std::runtime_error("Invalid quantization scale <= 0.");
    }
    const float qf = std::round(realVal / scale) + static_cast<float>(offset);
    const float lo = static_cast<float>(std::numeric_limits<T>::min());
    const float hi = static_cast<float>(std::numeric_limits<T>::max());
    return static_cast<T>(std::max(lo, std::min(hi, qf)));
}

float DequantizeToFloat(int32_t qVal, float scale, int32_t offset)
{
    return (static_cast<float>(qVal) - static_cast<float>(offset)) * scale;
}

class YoloArmnnInt8
{
public:
    YoloArmnnInt8(const std::string& modelPath, int numClasses, float confThresh, float iouThresh)
        : m_numClasses(numClasses), m_confThresh(confThresh), m_iouThresh(iouThresh)
    {
        m_parser = armnnTfLiteParser::ITfLiteParser::Create();
        if (!m_parser)
        {
            throw std::runtime_error("Failed to create TFLite parser.");
        }

        m_network = m_parser->CreateNetworkFromBinaryFile(modelPath.c_str());
        if (!m_network)
        {
            throw std::runtime_error("Failed to parse TFLite network.");
        }

        armnn::IRuntime::CreationOptions runtimeOptions;
        m_runtime = armnn::IRuntime::Create(runtimeOptions);
        if (!m_runtime)
        {
            throw std::runtime_error("Failed to create Arm NN runtime.");
        }

        m_inputNames = m_parser->GetSubgraphInputTensorNames(0);
        m_outputNames = m_parser->GetSubgraphOutputTensorNames(0);
        if (m_inputNames.empty() || m_outputNames.empty())
        {
            throw std::runtime_error("Model input/output tensor names are empty.");
        }

        m_inputBinding = m_parser->GetNetworkInputBindingInfo(0, m_inputNames[0]);
        for (const auto& name : m_outputNames)
        {
            m_outputBindings.push_back(m_parser->GetNetworkOutputBindingInfo(0, name));
        }

        std::vector<armnn::BackendId> preferredBackends = {
            armnn::Compute::GpuAcc,  // OpenCL backend (Mali)
            armnn::Compute::CpuAcc,
            armnn::Compute::CpuRef
        };
        armnn::OptimizerOptionsOpaque optimizerOptions;
        armnn::IOptimizedNetworkPtr optNetwork = armnn::Optimize(
            *m_network, preferredBackends, m_runtime->GetDeviceSpec(), optimizerOptions);
        if (!optNetwork)
        {
            throw std::runtime_error("Failed to optimize Arm NN network.");
        }

        armnn::Status status = m_runtime->LoadNetwork(m_networkId, std::move(optNetwork));
        if (status != armnn::Status::Success)
        {
            throw std::runtime_error("Failed to load network to Arm NN runtime.");
        }

        const auto inputShape = m_inputBinding.second.GetShape();
        if (inputShape.GetNumDimensions() != 4)
        {
            throw std::runtime_error("Expected NHWC 4D input tensor.");
        }
        m_inputH = static_cast<int>(inputShape[1]);
        m_inputW = static_cast<int>(inputShape[2]);
        m_inputC = static_cast<int>(inputShape[3]);
        if (m_inputC != 3)
        {
            throw std::runtime_error("Expected input channels = 3.");
        }
    }

    std::vector<Detection> Infer(const cv::Mat& bgr)
    {
        if (bgr.empty())
        {
            throw std::runtime_error("Input image is empty.");
        }

        LetterboxResult lb = Letterbox(bgr, m_inputW, m_inputH);
        cv::Mat rgb;
        cv::cvtColor(lb.image, rgb, cv::COLOR_BGR2RGB);

        armnn::InputTensors inputTensors;
        armnn::OutputTensors outputTensors;

        std::vector<uint8_t> inputU8;
        std::vector<int8_t> inputS8;
        PrepareInput(rgb, inputU8, inputS8, inputTensors);

        std::vector<std::vector<uint8_t>> outU8;
        std::vector<std::vector<int8_t>> outS8;
        std::vector<std::vector<float>> outF32;
        PrepareOutputStorage(outU8, outS8, outF32, outputTensors);

        armnn::Status runStatus = m_runtime->EnqueueWorkload(m_networkId, inputTensors, outputTensors);
        if (runStatus != armnn::Status::Success)
        {
            throw std::runtime_error("Arm NN inference failed.");
        }

        return DecodeDetections(outU8, outS8, outF32, bgr.size(), lb);
    }

private:
    void PrepareInput(const cv::Mat& rgb,
                      std::vector<uint8_t>& inputU8,
                      std::vector<int8_t>& inputS8,
                      armnn::InputTensors& inputTensors)
    {
        const armnn::TensorInfo& info = m_inputBinding.second;
        const size_t elemCount = info.GetNumElements();
        const float qScale = info.GetQuantizationScale();
        const int32_t qOffset = info.GetQuantizationOffset();

        if (info.GetDataType() == armnn::DataType::QAsymmU8)
        {
            inputU8.resize(elemCount);
            size_t k = 0;
            for (int y = 0; y < rgb.rows; ++y)
            {
                const cv::Vec3b* row = rgb.ptr<cv::Vec3b>(y);
                for (int x = 0; x < rgb.cols; ++x)
                {
                    for (int c = 0; c < 3; ++c)
                    {
                        const float v = static_cast<float>(row[x][c]) / 255.0f;
                        inputU8[k++] = QuantizeWithClamp<uint8_t>(v, qScale, qOffset);
                    }
                }
            }
            inputTensors.emplace_back(m_inputBinding.first, armnn::ConstTensor(info, inputU8.data()));
            return;
        }

        if (info.GetDataType() == armnn::DataType::QAsymmS8 || info.GetDataType() == armnn::DataType::QSymmS8)
        {
            inputS8.resize(elemCount);
            size_t k = 0;
            for (int y = 0; y < rgb.rows; ++y)
            {
                const cv::Vec3b* row = rgb.ptr<cv::Vec3b>(y);
                for (int x = 0; x < rgb.cols; ++x)
                {
                    for (int c = 0; c < 3; ++c)
                    {
                        const float v = static_cast<float>(row[x][c]) / 255.0f;
                        inputS8[k++] = QuantizeWithClamp<int8_t>(v, qScale, qOffset);
                    }
                }
            }
            inputTensors.emplace_back(m_inputBinding.first, armnn::ConstTensor(info, inputS8.data()));
            return;
        }

        throw std::runtime_error("Unsupported input data type. Expect INT8 or UINT8 quantized input.");
    }

    void PrepareOutputStorage(std::vector<std::vector<uint8_t>>& outU8,
                              std::vector<std::vector<int8_t>>& outS8,
                              std::vector<std::vector<float>>& outF32,
                              armnn::OutputTensors& outputTensors)
    {
        outU8.clear();
        outS8.clear();
        outF32.clear();
        outputTensors.clear();

        for (const auto& outputBinding : m_outputBindings)
        {
            const armnn::TensorInfo& info = outputBinding.second;
            const size_t elemCount = info.GetNumElements();
            if (info.GetDataType() == armnn::DataType::QAsymmU8)
            {
                outU8.emplace_back(elemCount);
                outputTensors.emplace_back(outputBinding.first, armnn::Tensor(info, outU8.back().data()));
            }
            else if (info.GetDataType() == armnn::DataType::QAsymmS8 || info.GetDataType() == armnn::DataType::QSymmS8)
            {
                outS8.emplace_back(elemCount);
                outputTensors.emplace_back(outputBinding.first, armnn::Tensor(info, outS8.back().data()));
            }
            else if (info.GetDataType() == armnn::DataType::Float32)
            {
                outF32.emplace_back(elemCount);
                outputTensors.emplace_back(outputBinding.first, armnn::Tensor(info, outF32.back().data()));
            }
            else
            {
                throw std::runtime_error("Unsupported output data type.");
            }
        }
    }

    std::vector<float> ToFloatVector(const armnn::TensorInfo& info,
                                     const uint8_t* dataU8,
                                     const int8_t* dataS8,
                                     const float* dataF32) const
    {
        const size_t n = info.GetNumElements();
        std::vector<float> out(n, 0.0f);
        const float scale = info.GetQuantizationScale();
        const int32_t offset = info.GetQuantizationOffset();

        if (info.GetDataType() == armnn::DataType::QAsymmU8)
        {
            for (size_t i = 0; i < n; ++i)
            {
                out[i] = DequantizeToFloat(static_cast<int32_t>(dataU8[i]), scale, offset);
            }
            return out;
        }
        if (info.GetDataType() == armnn::DataType::QAsymmS8 || info.GetDataType() == armnn::DataType::QSymmS8)
        {
            for (size_t i = 0; i < n; ++i)
            {
                out[i] = DequantizeToFloat(static_cast<int32_t>(dataS8[i]), scale, offset);
            }
            return out;
        }
        if (info.GetDataType() == armnn::DataType::Float32)
        {
            std::copy(dataF32, dataF32 + n, out.begin());
            return out;
        }
        throw std::runtime_error("Unsupported tensor type in dequantization.");
    }

    std::vector<Detection> DecodeDetections(const std::vector<std::vector<uint8_t>>& outU8,
                                            const std::vector<std::vector<int8_t>>& outS8,
                                            const std::vector<std::vector<float>>& outF32,
                                            const cv::Size& origSize,
                                            const LetterboxResult& lb)
    {
        // Typical YOLO exported head: [1, N, 4 + 1 + num_classes]
        // If your model differs, adapt this parser according to the actual output layout.
        std::vector<Detection> dets;

        size_t u8Idx = 0;
        size_t s8Idx = 0;
        size_t f32Idx = 0;
        for (const auto& outputBinding : m_outputBindings)
        {
            const armnn::TensorInfo& info = outputBinding.second;
            const auto shape = info.GetShape();
            if (shape.GetNumDimensions() < 2)
            {
                continue;
            }

            const uint8_t* ptrU8 = nullptr;
            const int8_t* ptrS8 = nullptr;
            const float* ptrF32 = nullptr;
            if (info.GetDataType() == armnn::DataType::QAsymmU8)
            {
                ptrU8 = outU8[u8Idx++].data();
            }
            else if (info.GetDataType() == armnn::DataType::QAsymmS8 || info.GetDataType() == armnn::DataType::QSymmS8)
            {
                ptrS8 = outS8[s8Idx++].data();
            }
            else
            {
                ptrF32 = outF32[f32Idx++].data();
            }

            const std::vector<float> out = ToFloatVector(info, ptrU8, ptrS8, ptrF32);

            const unsigned int lastDim = shape[shape.GetNumDimensions() - 1];
            if (lastDim < 6)
            {
                continue;
            }

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
                const int clsCount = std::min<int>(m_numClasses, static_cast<int>(lastDim - 5));
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

                // Handle normalized coordinates if values look like [0, 1].
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

                dets.push_back({cv::Rect(bx, by, bw, bh), bestCls, score});
            }
        }

        const std::vector<int> keep = Nms(dets, m_iouThresh);
        std::vector<Detection> filtered;
        filtered.reserve(keep.size());
        for (int idx : keep)
        {
            filtered.push_back(dets[idx]);
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

    armnnTfLiteParser::ITfLiteParserPtr m_parser;
    armnn::INetworkPtr m_network;
    armnn::IRuntimePtr m_runtime;
    armnn::NetworkId m_networkId = 0;
    armnn::BindingPointInfo m_inputBinding;
    std::vector<armnn::BindingPointInfo> m_outputBindings;
    std::vector<std::string> m_inputNames;
    std::vector<std::string> m_outputNames;
};

void DrawDetections(cv::Mat& image, const std::vector<Detection>& detections)
{
    for (const auto& d : detections)
    {
        cv::rectangle(image, d.box, cv::Scalar(0, 255, 0), 2);
        const std::string text =
            "cls=" + std::to_string(d.classId) + " score=" + cv::format("%.2f", static_cast<double>(d.score));
        int baseLine = 0;
        cv::Size textSize = cv::getTextSize(text, cv::FONT_HERSHEY_SIMPLEX, 0.5, 1, &baseLine);
        const int y = std::max(0, d.box.y - textSize.height - 4);
        cv::rectangle(image, cv::Rect(d.box.x, y, textSize.width + 4, textSize.height + 4), cv::Scalar(0, 255, 0), -1);
        cv::putText(image, text, cv::Point(d.box.x + 2, y + textSize.height), cv::FONT_HERSHEY_SIMPLEX, 0.5,
                    cv::Scalar(0, 0, 0), 1);
    }
}

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

        YoloArmnnInt8 detector(modelPath, numClasses, confThresh, iouThresh);
        const std::vector<Detection> detections = detector.Infer(image);
        DrawDetections(image, detections);

        if (!cv::imwrite(outputPath, image))
        {
            throw std::runtime_error("Failed to save output image: " + outputPath);
        }

        std::cout << "Inference done. Detections: " << detections.size() << "\n";
        std::cout << "Saved output: " << outputPath << "\n";
        return 0;
    }
    catch (const std::exception& e)
    {
        std::cerr << "[ERROR] " << e.what() << "\n";
        return 1;
    }
}
