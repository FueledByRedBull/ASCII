#include "core/frame_source.hpp"

#include <array>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <stdexcept>
#include <string>
#include <vector>

extern "C" {
#include <libavcodec/avcodec.h>
#include <libavformat/avformat.h>
#include <libavutil/channel_layout.h>
#include <libswscale/swscale.h>
}

#ifdef _WIN32
#include <fcntl.h>
#include <io.h>
#endif

using namespace ascii;

namespace {

void expect(bool condition, const char* message) {
    if (!condition) throw std::runtime_error(message);
}

struct MediaWriter {
    AVFormatContext* format = nullptr;
    AVCodecContext* codec = nullptr;
    AVFrame* frame = nullptr;
    AVPacket* packet = av_packet_alloc();
    ~MediaWriter() {
        av_packet_free(&packet);
        av_frame_free(&frame);
        avcodec_free_context(&codec);
        if (format) {
            if (format->pb) avio_closep(&format->pb);
            avformat_free_context(format);
        }
    }
};

void write_timestamped_video(const std::filesystem::path& path, const std::vector<int64_t>& pts,
                             bool longer_audio, Size size = {32, 24}, bool varying_alpha = false) {
    MediaWriter writer;
    expect(avformat_alloc_output_context2(&writer.format, nullptr, "matroska", path.string().c_str()) >= 0,
           "cannot allocate valid Matroska fixture");
    const AVCodec* codec = avcodec_find_encoder(AV_CODEC_ID_FFV1);
    expect(codec != nullptr, "FFV1 encoder required for timestamp fixtures");
    writer.codec = avcodec_alloc_context3(codec);
    expect(writer.codec != nullptr, "cannot allocate fixture codec");
    writer.codec->width = size.width;
    writer.codec->height = size.height;
    writer.codec->pix_fmt = AV_PIX_FMT_BGRA;
    writer.codec->time_base = {1, 1000};
    writer.codec->framerate = {25, 1};
    if (writer.format->oformat->flags & AVFMT_GLOBALHEADER) writer.codec->flags |= AV_CODEC_FLAG_GLOBAL_HEADER;
    expect(avcodec_open2(writer.codec, codec, nullptr) >= 0, "cannot open FFV1 encoder");
    AVStream* video = avformat_new_stream(writer.format, nullptr);
    expect(video && avcodec_parameters_from_context(video->codecpar, writer.codec) >= 0,
           "cannot set video fixture parameters");
    video->time_base = writer.codec->time_base;
    video->avg_frame_rate = writer.codec->framerate;
    AVStream* audio = nullptr;
    if (longer_audio) {
        audio = avformat_new_stream(writer.format, nullptr);
        expect(audio != nullptr, "cannot allocate audio fixture stream");
        audio->time_base = {1, 8000};
        audio->codecpar->codec_type = AVMEDIA_TYPE_AUDIO;
        audio->codecpar->codec_id = AV_CODEC_ID_PCM_S16LE;
        audio->codecpar->sample_rate = 8000;
        audio->codecpar->bits_per_coded_sample = 16;
        av_channel_layout_default(&audio->codecpar->ch_layout, 1);
    }
    expect(avio_open(&writer.format->pb, path.string().c_str(), AVIO_FLAG_WRITE) >= 0 &&
           avformat_write_header(writer.format, nullptr) >= 0, "cannot write fixture header");
    writer.frame = av_frame_alloc();
    expect(writer.frame && writer.packet, "cannot allocate fixture frame");
    writer.frame->format = writer.codec->pix_fmt;
    writer.frame->width = writer.codec->width;
    writer.frame->height = writer.codec->height;
    expect(av_frame_get_buffer(writer.frame, 32) >= 0, "cannot allocate fixture pixels");
    const auto drain = [&] {
        while (avcodec_receive_packet(writer.codec, writer.packet) == 0) {
            writer.packet->duration = 40;
            av_packet_rescale_ts(writer.packet, writer.codec->time_base, video->time_base);
            writer.packet->stream_index = video->index;
            expect(av_interleaved_write_frame(writer.format, writer.packet) >= 0, "cannot write fixture video");
            av_packet_unref(writer.packet);
        }
    };
    for (size_t i = 0; i < pts.size(); ++i) {
        expect(av_frame_make_writable(writer.frame) >= 0, "fixture frame must be writable");
        for (int y = 0; y < writer.frame->height; ++y) {
            auto* row = writer.frame->data[0] + y * writer.frame->linesize[0];
            for (int x = 0; x < writer.frame->width; ++x) {
                row[4 * x] = static_cast<uint8_t>(x * 7 + i * 13);
                row[4 * x + 1] = static_cast<uint8_t>(y * 9 + i * 17);
                row[4 * x + 2] = static_cast<uint8_t>(x * y + i * 31);
                row[4 * x + 3] = varying_alpha ? static_cast<uint8_t>(x * 11 + y * 13 + i * 17) : 255;
            }
        }
        writer.frame->pts = pts[i];
        writer.frame->duration = 40;
        expect(avcodec_send_frame(writer.codec, writer.frame) >= 0, "cannot encode fixture frame");
        drain();
    }
    expect(avcodec_send_frame(writer.codec, nullptr) >= 0, "cannot flush fixture video");
    drain();
    if (audio) {
        expect(av_new_packet(writer.packet, 32000) >= 0, "cannot allocate PCM fixture packet");
        std::fill_n(writer.packet->data, writer.packet->size, uint8_t{0});
        writer.packet->pts = writer.packet->dts = 0;
        writer.packet->duration = av_rescale_q(16000, AVRational{1, 8000}, audio->time_base);
        writer.packet->stream_index = audio->index;
        expect(av_interleaved_write_frame(writer.format, writer.packet) >= 0, "cannot write valid PCM audio");
    }
    expect(av_write_trailer(writer.format) >= 0, "cannot finish fixture video");
}

void write_pixel_format_video(const std::filesystem::path& path, const std::filesystem::path& reference,
                             bool change_shape = false, Size size = {32, 24}) {
    std::ofstream stream(path, std::ios::binary);
    for (AVPixelFormat format : {AV_PIX_FMT_YUVJ420P, AV_PIX_FMT_YUVJ444P}) {
        MediaWriter writer;
        const AVCodec* codec = avcodec_find_encoder(AV_CODEC_ID_MJPEG);
        expect(codec != nullptr, "MJPEG encoder required for pixel-format fixture");
        writer.codec = avcodec_alloc_context3(codec);
        const int width = change_shape && format == AV_PIX_FMT_YUVJ444P ? size.width * 2 : size.width;
        const int height = change_shape && format == AV_PIX_FMT_YUVJ444P ? size.height * 2 : size.height;
        writer.codec->width = width;
        writer.codec->height = height;
        writer.codec->pix_fmt = format;
        writer.codec->time_base = {1, 25};
        expect(avcodec_open2(writer.codec, codec, nullptr) >= 0, "cannot encode valid JPEG fixture");
        writer.frame = av_frame_alloc();
        writer.frame->width = width;
        writer.frame->height = height;
        writer.frame->format = format;
        expect(av_frame_get_buffer(writer.frame, 32) >= 0, "cannot allocate JPEG pixels");
        for (int plane = 0; plane < 3; ++plane) {
            const int divisor = plane > 0 && format == AV_PIX_FMT_YUVJ420P ? 2 : 1;
            for (int y = 0; y < (height + divisor - 1) / divisor; ++y) {
                auto* row = writer.frame->data[plane] + y * writer.frame->linesize[plane];
                for (int x = 0; x < (width + divisor - 1) / divisor; ++x) row[x] = static_cast<uint8_t>(70 + plane * 20 + x * 3);
            }
        }
        writer.frame->pts = 0;
        expect(avcodec_send_frame(writer.codec, writer.frame) >= 0 &&
               avcodec_receive_packet(writer.codec, writer.packet) == 0, "cannot get JPEG fixture packet");
        stream.write(reinterpret_cast<char*>(writer.packet->data), writer.packet->size);
        if (format == AV_PIX_FMT_YUVJ444P) {
            std::ofstream single(reference, std::ios::binary);
            single.write(reinterpret_cast<char*>(writer.packet->data), writer.packet->size);
            expect(single.good(), "cannot save reference JPEG");
        }
    }
    expect(stream.good(), "cannot save MJPEG stream");
}

void write_bmp(const std::filesystem::path& path, int width, int height) {
    const int stride = (width * 3 + 3) & ~3;
    std::vector<uint8_t> bytes(54 + stride * height, 0);
    const auto put32 = [&](int offset, int value) {
        for (int byte = 0; byte < 4; ++byte) bytes[offset + byte] = static_cast<uint8_t>(value >> (byte * 8));
    };
    bytes[0] = 'B'; bytes[1] = 'M';
    put32(2, static_cast<int>(bytes.size())); put32(10, 54); put32(14, 40);
    put32(18, width); put32(22, height); bytes[26] = 1; bytes[28] = 24;
    std::ofstream file(path, std::ios::binary);
    file.write(reinterpret_cast<char*>(bytes.data()), static_cast<std::streamsize>(bytes.size()));
    expect(file.good(), "cannot create valid BMP fixture");
}

int read_to_end(VideoFileSource& source, FrameReadStatus& final_status) {
    FrameBuffer frame;
    int count = 0;
    while ((final_status = source.read_next(frame)) == FrameReadStatus::Frame) {
        expect(++count < 1000, "unexpected fixture frame count");
    }
    return count;
}

void test_timestamps(const std::filesystem::path& directory, bool longer_audio, int64_t offset = 0) {
    const auto path = directory / (longer_audio ? "long-audio.mkv" : "vfr.mkv");
    std::vector<int64_t> pts = longer_audio ? std::vector<int64_t>{0, 40, 80, 120}
                                          : std::vector<int64_t>{0, 40, 80, 1000};
    for (auto& timestamp : pts) timestamp += offset;
    write_timestamped_video(path, pts, longer_audio);
    VideoFileSource source;
    expect(source.open(path.string()), "timestamp fixture must open");
    FrameReadStatus status;
    expect(read_to_end(source, status) == 4, "timestamp fixture must decode four frames");
    expect(status == FrameReadStatus::End, "valid video timestamps/audio duration must end cleanly");
    source.reset();
    expect(read_to_end(source, status) == 4 && status == FrameReadStatus::End,
           "reset must replay every timestamped frame and end cleanly");
}

void test_pixel_format(const std::filesystem::path& directory) {
    const auto path = directory / "pixel-format.mjpeg";
    const auto reference = directory / "reference.jpg";
    write_pixel_format_video(path, reference);
    VideoFileSource stream, single;
    expect(stream.open(path.string()) && single.open(reference.string()), "JPEG fixtures must open");
    FrameBuffer first, actual, expected;
    expect(stream.read(first) && stream.read(actual) && single.read(expected), "JPEG fixtures must decode");
    expect(actual.size() == expected.size(), "JPEG frame shape must match");
    int max_error = 0;
    for (size_t i = 0; i < actual.byte_size(); ++i) {
        max_error = std::max(max_error, std::abs(static_cast<int>(actual.data()[i]) - expected.data()[i]));
    }
    expect(max_error <= 2, "same-sized pixel-format change must use the matching conversion context");
}

FrameBuffer jpeg_rgb_reference(const std::filesystem::path& path) {
    MediaWriter decoder;
    const AVCodec* codec = avcodec_find_decoder(AV_CODEC_ID_MJPEG);
    expect(codec != nullptr, "JPEG decoder required for independent RGB reference");
    decoder.codec = avcodec_alloc_context3(codec);
    decoder.frame = av_frame_alloc();
    expect(decoder.codec && decoder.frame && decoder.packet &&
           avcodec_open2(decoder.codec, codec, nullptr) >= 0, "cannot allocate reference JPEG decoder");
    std::ifstream input(path, std::ios::binary | std::ios::ate);
    const auto length = input.tellg();
    expect(length > 0 && length < 10000000, "invalid generated JPEG length");
    expect(av_new_packet(decoder.packet, static_cast<int>(length)) >= 0, "cannot allocate reference JPEG packet");
    input.seekg(0);
    input.read(reinterpret_cast<char*>(decoder.packet->data), length);
    expect(input.good() && avcodec_send_packet(decoder.codec, decoder.packet) >= 0 &&
           avcodec_receive_frame(decoder.codec, decoder.frame) == 0, "cannot decode reference JPEG");
    MediaWriter rgb;
    rgb.frame = av_frame_alloc();
    expect(rgb.frame != nullptr, "cannot allocate reference RGB frame");
    rgb.frame->format = AV_PIX_FMT_RGB24;
    rgb.frame->width = decoder.frame->width;
    rgb.frame->height = decoder.frame->height;
    expect(av_frame_get_buffer(rgb.frame, 0) >= 0, "cannot allocate padded reference RGB storage");
    SwsContext* scale = sws_getContext(rgb.frame->width, rgb.frame->height,
        static_cast<AVPixelFormat>(decoder.frame->format), rgb.frame->width, rgb.frame->height,
        AV_PIX_FMT_RGB24, SWS_BILINEAR, nullptr, nullptr, nullptr);
    expect(scale != nullptr, "cannot allocate RGB reference converter");
    const int rows = sws_scale(scale, decoder.frame->data, decoder.frame->linesize, 0,
        rgb.frame->height, rgb.frame->data, rgb.frame->linesize);
    sws_freeContext(scale);
    expect(rows == rgb.frame->height, "reference conversion must produce every row");
    FrameBuffer result(rgb.frame->width, rgb.frame->height);
    for (int y = 0; y < result.height(); ++y) {
        const auto* row = rgb.frame->data[0] + y * rgb.frame->linesize[0];
        for (int x = 0; x < result.width(); ++x)
            result.set_pixel(x, y, Color(row[3 * x], row[3 * x + 1], row[3 * x + 2], 255));
    }
    return result;
}

void test_conversion_padding(const std::filesystem::path& directory) {
    // Includes narrow/odd rows and the width that exposed the old packed RGB overrun.
    for (Size size : {Size{1, 1}, Size{3, 5}, Size{7, 3}, Size{17, 9}, Size{700, 350}}) {
        const auto stem = "padded-" + std::to_string(size.width);
        const auto video_path = directory / (stem + ".mjpeg");
        const auto jpeg_path = directory / (stem + ".jpg");
        write_pixel_format_video(video_path, jpeg_path, false, size);
        const FrameBuffer expected = jpeg_rgb_reference(jpeg_path);
        for (const auto& path : {jpeg_path, video_path}) {
            VideoFileSource source;
            expect(source.open(path.string()), "narrow/odd JPEG source must open");
            FrameBuffer frame;
            expect(source.read_next(frame) == FrameReadStatus::Frame, "JPEG source must produce a frame");
            if (path == video_path)
                expect(source.read_next(frame) == FrameReadStatus::Frame, "MJPEG must preserve both pixel formats");
            expect(frame.size() == size && frame.byte_size() == static_cast<size_t>(size.width) * size.height * 4,
                   "padded conversion must preserve logical frame geometry");
            expect(std::equal(frame.data(), frame.data() + frame.byte_size(), expected.data()),
                   "RGBA conversion must match independent padded RGB conversion byte for byte");
            expect(source.read_next(frame) == FrameReadStatus::End && source.read_next(frame) == FrameReadStatus::End,
                   "JPEG conversion must end cleanly and stay at EOF");
        }
    }
    const auto path = directory / "alpha-odd.mkv";
    write_timestamped_video(path, {0, 40, 80}, false, {17, 9}, true);
    VideoFileSource source;
    expect(source.open(path.string()), "lossless alpha source must open");
    FrameBuffer frame;
    for (int i = 0; i < 3; ++i) {
        expect(source.read_next(frame) == FrameReadStatus::Frame, "lossless alpha frame must decode");
        for (int y = 0; y < frame.height(); ++y) {
            for (int x = 0; x < frame.width(); ++x) {
                const auto pixel = frame.get_pixel(x, y);
                expect(pixel.r == static_cast<uint8_t>(x * y + i * 31) &&
                       pixel.g == static_cast<uint8_t>(y * 9 + i * 17) &&
                       pixel.b == static_cast<uint8_t>(x * 7 + i * 13) && pixel.a == 255,
                       "decoded media must preserve RGB and remain opaque even with varying source alpha");
            }
        }
    }
    expect(source.read_next(frame) == FrameReadStatus::End, "alpha fixture must end cleanly");
}

void test_pipe_spec() {
    PipeSource pipe;
    for (const char* uri : {"pipe:1junkx1", "pipe:1x1junk", "pipe:1x1:rgb:30junk",
                            "pipe:1x1:rgb:30:extra", "pipe:1x1:rgb:", "pipe:1x1:",
                            "pipe:1x1:rgb:nan", "pipe:1x1:rgb:inf"}) {
        expect(!pipe.open(uri), "pipe fields must parse completely and reject extra/empty fields");
        expect(!pipe.is_open(), "invalid pipe must remain closed");
    }
    expect(pipe.open("pipe:2x3:rgba:29.97"), "valid raw-pipe description must open");
}

void test_exact_sequence(const std::filesystem::path& directory) {
    const auto images = directory / "exact-sequence";
    std::filesystem::create_directories(images);
    write_bmp(images / "a.bmp", 16, 16);
    write_bmp(images / "b.bmp", 32, 32);
    ImageSequenceSource sequence;
    expect(sequence.open((images / "b.bmp").string()), "exact sequence image must open");
    FrameBuffer frame;
    expect(sequence.read_next(frame) == FrameReadStatus::Frame && frame.width() == 32 && frame.height() == 32,
           "exact sequence path must select only the named image");
    expect(sequence.read_next(frame) == FrameReadStatus::End, "exact sequence must contain one image");
    expect(!sequence.open((images / "missing.bmp").string()), "missing exact image cannot select sibling files");
#ifdef _WIN32
    expect(sequence.open((images / "B.BMP").string()), "exact sequence path must honor Windows filename matching");
#endif
}

void test_pixel_limits(const std::filesystem::path& directory) {
    const auto path = directory / "limit-a.bmp";
    write_bmp(path, 16, 16);
    ImageSource image;
    image.set_pixel_limit(255);
    expect(!image.open(path.string()) && !image.is_open(), "image must reject dimensions over its pixel budget");
    image.set_pixel_limit(256);
    expect(image.open(path.string()), "image exactly at its pixel budget must open");
    ImageSequenceSource sequence;
    sequence.set_pixel_limit(255);
    expect(!sequence.open((directory / "limit-*.bmp").string()), "image sequence must propagate its pixel budget");
    write_bmp(directory / "limit-b.bmp", 32, 32);
    sequence.set_pixel_limit(256);
    expect(sequence.open((directory / "limit-*.bmp").string()), "sequence first image at its budget must open");
    FrameBuffer image_frame;
    expect(sequence.read_next(image_frame) == FrameReadStatus::Frame &&
           sequence.read_next(image_frame) == FrameReadStatus::Error,
           "later sequence images must respect the same pixel budget");
    PipeSource pipe;
    pipe.set_pixel_limit(255);
    expect(!pipe.open("pipe:16x16:rgb"), "pipe must reject dimensions over its pixel budget");
    const auto video = directory / "limit.mkv";
    write_timestamped_video(video, {0, 40}, false);
    VideoFileSource source;
    source.set_pixel_limit(767);
    expect(!source.open(video.string()) && !source.is_open(), "video metadata must respect the pixel budget");
    const auto changing = directory / "resolution-change.mjpeg";
    write_pixel_format_video(changing, directory / "resolution-reference.jpg", true);
    source.set_pixel_limit(768);
    if (source.open(changing.string())) {
        FrameBuffer frame;
        FrameReadStatus status;
        while ((status = source.read_next(frame)) == FrameReadStatus::Frame) {
            expect(frame.size().area() <= 768, "decoded frames cannot exceed the pixel budget");
        }
        expect(status == FrameReadStatus::Error, "oversized resolution change must fail decoding");
    }
}

void test_pipe_binary(const std::filesystem::path& directory) {
#ifdef _WIN32
    const auto path = directory / "raw.rgb";
    const std::array<uint8_t, 6> bytes{13, 10, 26, 85, 119, 153};
    {
        std::ofstream file(path, std::ios::binary);
        file.write(reinterpret_cast<const char*>(bytes.data()), bytes.size());
    }
    struct RestoreStdin {
        int descriptor = _dup(_fileno(stdin));
        int mode = _setmode(_fileno(stdin), _O_BINARY);
        std::ios::iostate state = std::cin.rdstate();
        ~RestoreStdin() {
            if (descriptor != -1) {
                _dup2(descriptor, _fileno(stdin));
                _close(descriptor);
            }
            if (mode != -1) _setmode(_fileno(stdin), mode);
            std::cin.clear(state);
        }
    } restore;
    expect(restore.descriptor != -1 && restore.mode != -1, "cannot preserve test stdin");
    const int input = _open(path.string().c_str(), _O_RDONLY | _O_BINARY);
    expect(input != -1, "cannot open raw fixture");
    const int redirect = _dup2(input, _fileno(stdin));
    _close(input);
    expect(redirect == 0 && _setmode(_fileno(stdin), _O_TEXT) != -1, "cannot redirect test stdin");
    std::cin.clear();
    {
        PipeSource pipe;
        expect(pipe.open("pipe:2x1:rgb"), "binary pipe fixture must open");
        FrameBuffer frame;
        expect(pipe.read_next(frame) == FrameReadStatus::Frame, "raw pipe must preserve CR/LF/Ctrl-Z bytes");
        const auto first = frame.get_pixel(0, 0);
        const auto second = frame.get_pixel(1, 0);
        expect(first.r == 13 && first.g == 10 && first.b == 26 &&
               second.r == 85 && second.g == 119 && second.b == 153, "binary pipe bytes must remain exact");
        expect(pipe.read_next(frame) == FrameReadStatus::End, "complete pipe must end cleanly");
    }
    const int restored_mode = _setmode(_fileno(stdin), _O_BINARY);
    expect(restored_mode == _O_TEXT, "pipe destruction must restore the original stdin mode");
    _setmode(_fileno(stdin), restored_mode);
#else
    (void)directory;
#endif
}

void test_truncation(const std::filesystem::path& directory) {
    const auto path = directory / "complete.mkv";
    write_timestamped_video(path, {0, 40, 80, 120, 160, 200, 240, 280}, false);
    std::ifstream input(path, std::ios::binary);
    std::vector<char> bytes((std::istreambuf_iterator<char>(input)), {});
    const auto truncated = directory / "truncated.mkv";
    std::ofstream output(truncated, std::ios::binary);
    output.write(bytes.data(), static_cast<std::streamsize>(bytes.size() * 3 / 4));
    output.close();
    VideoFileSource source;
    if (!source.open(truncated.string())) return;
    FrameReadStatus status;
    read_to_end(source, status);
    expect(status == FrameReadStatus::Error, "three-quarter truncated video must not end cleanly");
}

} // namespace

int main(int argc, char** argv) {
    try {
        const auto directory = std::filesystem::temp_directory_path() / "ascii-source-regressions";
        std::filesystem::create_directories(directory);
        const std::string selected = argc > 1 ? argv[1] : "all";
        expect(selected == "all" || selected == "pipe" || selected == "binary" || selected == "limits" ||
               selected == "vfr" || selected == "audio" || selected == "format" || selected == "truncated" ||
               selected == "sequence" || selected == "conversion",
               "unknown source regression group");
        if (selected == "all" || selected == "pipe") test_pipe_spec();
        if (selected == "all" || selected == "binary") test_pipe_binary(directory);
        if (selected == "all" || selected == "limits") test_pixel_limits(directory);
        if (selected == "all" || selected == "sequence") test_exact_sequence(directory);
        if (selected == "all" || selected == "vfr") {
            test_timestamps(directory, false);
            test_timestamps(directory, false, 2000);
        }
        if (selected == "all" || selected == "audio") test_timestamps(directory, true);
        if (selected == "all" || selected == "format") test_pixel_format(directory);
        if (selected == "all" || selected == "conversion") test_conversion_padding(directory);
        if (selected == "all" || selected == "truncated") test_truncation(directory);
        std::cout << "[OK] Source regression tests passed: " << selected << '\n';
        return 0;
    } catch (const std::exception& error) {
        std::cerr << "[FAIL] " << error.what() << '\n';
        return 1;
    }
}
