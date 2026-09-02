#include <rclcpp/rclcpp.hpp>
#include <sensor_msgs/msg/point_cloud2.hpp>
#include <pcl_conversions/pcl_conversions.h>
#include <pcl/point_cloud.h>
#include <pcl/point_types.h>

#include <algorithm>
#include <array>
#include <cctype>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <memory>
#include <mutex>
#include <queue>
#include <stdexcept>
#include <string>
#include <unordered_set>
#include <vector>

#define PT_INFO(...) \
    do { if (shouldFrameLog()) { RCLCPP_INFO(this->get_logger(), __VA_ARGS__); } } while (0)
#define PT_WARN(...) \
    do { if (shouldFrameLog()) { RCLCPP_WARN(this->get_logger(), __VA_ARGS__); } } while (0)

class PointCloudMemoryPool {
public:
    PointCloudMemoryPool(size_t initial_pool_size = 10, size_t reserve_size = 100000)
        : reserve_size_(reserve_size)
    {
        for (size_t i = 0; i < initial_pool_size; ++i) {
            auto cloud = std::make_shared<pcl::PointCloud<pcl::PointXYZ>>();
            cloud->points.reserve(reserve_size);
            in_pool_.insert(cloud.get());
            available_clouds_.push(cloud);
        }
    }

    pcl::PointCloud<pcl::PointXYZ>::Ptr acquire()
    {
        std::lock_guard<std::mutex> lock(mutex_);
        if (!available_clouds_.empty()) {
            auto cloud = available_clouds_.front();
            available_clouds_.pop();
            in_pool_.erase(cloud.get());
            cloud->clear();
            if (cloud->points.capacity() < reserve_size_) {
                cloud->points.reserve(reserve_size_);
            }
            return cloud;
        }
        auto cloud = std::make_shared<pcl::PointCloud<pcl::PointXYZ>>();
        cloud->points.reserve(reserve_size_);
        return cloud;
    }

    void release(pcl::PointCloud<pcl::PointXYZ>::Ptr cloud)
    {
        if (!cloud) {
            return;
        }
        std::lock_guard<std::mutex> lock(mutex_);
        if (in_pool_.find(cloud.get()) != in_pool_.end()) {
            return;
        }
        cloud->clear();
        if (cloud->points.capacity() < reserve_size_) {
            cloud->points.reserve(reserve_size_);
        }
        in_pool_.insert(cloud.get());
        available_clouds_.push(cloud);
    }

private:
    std::queue<pcl::PointCloud<pcl::PointXYZ>::Ptr> available_clouds_;
    std::unordered_set<const pcl::PointCloud<pcl::PointXYZ>*> in_pool_;
    std::mutex mutex_;
    size_t reserve_size_;
};

struct AxisSpec {
    char name{'x'};
    int dim{0};
    bool enabled{true};
    double limit_min{0.0};
    double limit_max{0.0};
    rclcpp::Publisher<sensor_msgs::msg::PointCloud2>::SharedPtr debug_pub;
};

class CloudPassthroughFilterNode : public rclcpp::Node {
public:
    CloudPassthroughFilterNode()
        : Node("cloud_passthrough_filter_node")
    {
        declare_and_load_parameters();

        memory_pool_ = std::make_unique<PointCloudMemoryPool>(
            static_cast<size_t>(memory_pool_size_),
            static_cast<size_t>(memory_pool_reserve_));

        subscription_ = this->create_subscription<sensor_msgs::msg::PointCloud2>(
            input_topic_, rclcpp::QoS{1}.best_effort(),
            std::bind(&CloudPassthroughFilterNode::cloud_callback, this, std::placeholders::_1));
        publisher_ = this->create_publisher<sensor_msgs::msg::PointCloud2>(
            output_topic_, rclcpp::QoS{1}.best_effort());

        if (debug_mode_) {
            axes_[0].debug_pub = this->create_publisher<sensor_msgs::msg::PointCloud2>(
                debug_topic_x_, rclcpp::QoS{5}.best_effort());
            axes_[1].debug_pub = this->create_publisher<sensor_msgs::msg::PointCloud2>(
                debug_topic_y_, rclcpp::QoS{5}.best_effort());
            axes_[2].debug_pub = this->create_publisher<sensor_msgs::msg::PointCloud2>(
                debug_topic_z_, rclcpp::QoS{5}.best_effort());
        }

        log_startup();
    }

private:
    void declare_and_load_parameters()
    {
        this->declare_parameter("input_topic", std::string("input_cloud"));
        this->declare_parameter("output_topic", std::string("filtered_cloud"));
        this->declare_parameter("filter_order", std::string("xyz"));
        this->declare_parameter("enable_x", true);
        this->declare_parameter("enable_y", true);
        this->declare_parameter("enable_z", true);
        this->declare_parameter("limit_min_x", 0.0);
        this->declare_parameter("limit_max_x", 3.5);
        this->declare_parameter("limit_min_y", -2.0);
        this->declare_parameter("limit_max_y", 2.0);
        this->declare_parameter("limit_min_z", -0.2);
        this->declare_parameter("limit_max_z", 0.4);
        this->declare_parameter("debug_mode", false);
        this->declare_parameter("debug_topic_x", std::string("passthrough_cloud_x"));
        this->declare_parameter("debug_topic_y", std::string("passthrough_cloud_y"));
        this->declare_parameter("debug_topic_z", std::string("passthrough_cloud_z"));
        this->declare_parameter("memory_pool_size", 10);
        this->declare_parameter("memory_pool_reserve", 1000000);
        this->declare_parameter("cloud_log_interval_sec", 1.0);

        input_topic_ = this->get_parameter("input_topic").as_string();
        output_topic_ = this->get_parameter("output_topic").as_string();
        filter_order_raw_ = this->get_parameter("filter_order").as_string();
        axes_[0].enabled = this->get_parameter("enable_x").as_bool();
        axes_[1].enabled = this->get_parameter("enable_y").as_bool();
        axes_[2].enabled = this->get_parameter("enable_z").as_bool();
        axes_[0].limit_min = this->get_parameter("limit_min_x").as_double();
        axes_[0].limit_max = this->get_parameter("limit_max_x").as_double();
        axes_[1].limit_min = this->get_parameter("limit_min_y").as_double();
        axes_[1].limit_max = this->get_parameter("limit_max_y").as_double();
        axes_[2].limit_min = this->get_parameter("limit_min_z").as_double();
        axes_[2].limit_max = this->get_parameter("limit_max_z").as_double();
        debug_mode_ = this->get_parameter("debug_mode").as_bool();
        debug_topic_x_ = this->get_parameter("debug_topic_x").as_string();
        debug_topic_y_ = this->get_parameter("debug_topic_y").as_string();
        debug_topic_z_ = this->get_parameter("debug_topic_z").as_string();
        memory_pool_size_ = this->get_parameter("memory_pool_size").as_int();
        memory_pool_reserve_ = this->get_parameter("memory_pool_reserve").as_int();
        cloud_log_interval_sec_ = this->get_parameter("cloud_log_interval_sec").as_double();

        axes_[0].name = 'x';
        axes_[0].dim = 0;
        axes_[1].name = 'y';
        axes_[1].dim = 1;
        axes_[2].name = 'z';
        axes_[2].dim = 2;

        if (memory_pool_size_ <= 0) {
            memory_pool_size_ = 10;
        }
        if (memory_pool_reserve_ <= 0) {
            memory_pool_reserve_ = 1000000;
        }
        for (auto& axis : axes_) {
            if (axis.limit_min > axis.limit_max) {
                std::swap(axis.limit_min, axis.limit_max);
                RCLCPP_WARN(this->get_logger(),
                            "%c 轴上下限颠倒，已交换为 [%.4f, %.4f]",
                            static_cast<char>(std::toupper(axis.name)),
                            axis.limit_min, axis.limit_max);
            }
        }

        if (!parse_filter_order(filter_order_raw_, filter_order_)) {
            throw std::runtime_error("filter_order 非法: " + filter_order_raw_);
        }
    }

    bool parse_filter_order(const std::string& raw, std::vector<int>& order) const
    {
        order.clear();
        bool seen[3] = {false, false, false};
        for (char c : raw) {
            if (c == ' ' || c == ',' || c == ';' || c == '-') {
                continue;
            }
            const char lower = static_cast<char>(std::tolower(static_cast<unsigned char>(c)));
            int dim = -1;
            if (lower == 'x') {
                dim = 0;
            } else if (lower == 'y') {
                dim = 1;
            } else if (lower == 'z') {
                dim = 2;
            } else {
                RCLCPP_ERROR(this->get_logger(),
                             "filter_order='%s' 含非法字符 '%c'，仅允许 x/y/z",
                             raw.c_str(), c);
                return false;
            }
            if (seen[dim]) {
                RCLCPP_ERROR(this->get_logger(),
                             "filter_order='%s' 中轴 '%c' 重复",
                             raw.c_str(), lower);
                return false;
            }
            seen[dim] = true;
            order.push_back(dim);
        }
        if (order.empty()) {
            RCLCPP_ERROR(this->get_logger(), "filter_order='%s' 解析后为空", raw.c_str());
            return false;
        }
        return true;
    }

    std::string order_string() const
    {
        std::string s;
        s.reserve(filter_order_.size());
        for (int dim : filter_order_) {
            s.push_back(axes_[static_cast<size_t>(dim)].name);
        }
        return s;
    }

    void log_startup() const
    {
        RCLCPP_INFO(this->get_logger(), "========== XYZ 直通滤波节点 (CPU) ==========");
        RCLCPP_INFO(this->get_logger(), "  输入话题: %s", input_topic_.c_str());
        RCLCPP_INFO(this->get_logger(), "  输出话题: %s", output_topic_.c_str());
        RCLCPP_INFO(this->get_logger(), "  滤波顺序: %s", order_string().c_str());
        for (const auto& axis : axes_) {
            RCLCPP_INFO(this->get_logger(),
                        "  %c轴: %s  区间=[%.4f, %.4f]",
                        static_cast<char>(std::toupper(axis.name)),
                        axis.enabled ? "启用" : "关闭",
                        axis.limit_min, axis.limit_max);
        }
        RCLCPP_INFO(this->get_logger(), "  CPU 点云池: %d 块, reserve=%d",
                    memory_pool_size_, memory_pool_reserve_);
        RCLCPP_INFO(this->get_logger(), "  调试模式: %s", debug_mode_ ? "是" : "否");
        if (debug_mode_) {
            RCLCPP_INFO(this->get_logger(), "  X debug 话题: %s", debug_topic_x_.c_str());
            RCLCPP_INFO(this->get_logger(), "  Y debug 话题: %s", debug_topic_y_.c_str());
            RCLCPP_INFO(this->get_logger(), "  Z debug 话题: %s", debug_topic_z_.c_str());
        }
        RCLCPP_INFO(this->get_logger(),
                    "  帧日志间隔: %.2f s (仅 debug_mode=false 时限频)",
                    cloud_log_interval_sec_);
    }

    bool shouldFrameLog() const
    {
        return debug_mode_ || frame_log_enabled_;
    }

    void updateFrameLogGate()
    {
        if (debug_mode_) {
            frame_log_enabled_ = true;
            return;
        }
        const auto now = std::chrono::steady_clock::now();
        const double elapsed_sec =
            std::chrono::duration<double>(now - last_cloud_log_time_).count();
        if (cloud_log_interval_sec_ <= 0.0 || elapsed_sec >= cloud_log_interval_sec_) {
            frame_log_enabled_ = true;
            last_cloud_log_time_ = now;
        } else {
            frame_log_enabled_ = false;
        }
    }

    void removeNonFinitePointsInPlace(pcl::PointCloud<pcl::PointXYZ>::Ptr cloud)
    {
        if (!cloud || cloud->empty()) {
            return;
        }
        const std::size_t before = cloud->points.size();
        auto it = std::remove_if(cloud->points.begin(), cloud->points.end(),
            [](const pcl::PointXYZ& p) {
                return !std::isfinite(p.x) || !std::isfinite(p.y) || !std::isfinite(p.z);
            });
        if (it == cloud->points.end()) {
            cloud->width = static_cast<uint32_t>(cloud->points.size());
            cloud->height = 1;
            cloud->is_dense = true;
            return;
        }
        cloud->points.erase(it, cloud->points.end());
        cloud->width = static_cast<uint32_t>(cloud->points.size());
        cloud->height = 1;
        cloud->is_dense = true;
        PT_WARN("移除非法点(NaN/Inf): %zu → %zu", before, cloud->points.size());
    }

    void publish_cloud(const pcl::PointCloud<pcl::PointXYZ>::Ptr& cloud,
                       const std_msgs::msg::Header& header,
                       const rclcpp::Publisher<sensor_msgs::msg::PointCloud2>::SharedPtr& pub,
                       const char* description)
    {
        if (!pub || !cloud) {
            return;
        }
        sensor_msgs::msg::PointCloud2 msg;
        pcl::toROSMsg(*cloud, msg);
        msg.header = header;
        pub->publish(msg);
        PT_INFO("已发布 %s 点云，点数=%zu", description, cloud->size());
    }

    // 始终从内存池取新块，永不返回输入指针，便于调用方安全 release 输入。
    pcl::PointCloud<pcl::PointXYZ>::Ptr
    passthrough_filter_cpu(const pcl::PointCloud<pcl::PointXYZ>::Ptr& cloud,
                           double lower_limit, double upper_limit, int dimension)
    {
        auto filtered = memory_pool_->acquire();
        filtered->height = 1;
        filtered->is_dense = true;
        if (!cloud || cloud->empty()) {
            filtered->width = 0;
            return filtered;
        }
        filtered->points.reserve(std::max(filtered->points.capacity(), cloud->size()));
        for (const auto& p : cloud->points) {
            const float value = (dimension == 0) ? p.x : ((dimension == 1) ? p.y : p.z);
            if (value >= lower_limit && value <= upper_limit) {
                filtered->points.push_back(p);
            }
        }
        filtered->width = static_cast<uint32_t>(filtered->points.size());
        return filtered;
    }

    void replace_current(pcl::PointCloud<pcl::PointXYZ>::Ptr& current,
                         pcl::PointCloud<pcl::PointXYZ>::Ptr next)
    {
        if (current && current != next) {
            memory_pool_->release(current);
        }
        current = std::move(next);
    }

    void cloud_callback(const sensor_msgs::msg::PointCloud2::SharedPtr msg)
    {
        updateFrameLogGate();
        const auto t0 = std::chrono::high_resolution_clock::now();

        auto current = memory_pool_->acquire();
        pcl::fromROSMsg(*msg, *current);

        if (current->empty()) {
            PT_WARN("收到空点云");
            publish_cloud(current, msg->header, publisher_, "最终输出");
            memory_pool_->release(current);
            return;
        }

        const size_t n_raw = current->size();
        removeNonFinitePointsInPlace(current);
        const size_t n_in = current->size();
        PT_INFO("输入 %zu 点 (原始 %zu)，order=%s", n_in, n_raw, order_string().c_str());

        for (int dim : filter_order_) {
            AxisSpec& axis = axes_[static_cast<size_t>(dim)];
            if (!axis.enabled) {
                PT_INFO("[%c] 跳过 (未启用)", static_cast<char>(std::toupper(axis.name)));
                continue;
            }
            if (!current || current->empty()) {
                PT_WARN("[%c] 输入为空，后续轴跳过", static_cast<char>(std::toupper(axis.name)));
                break;
            }

            const size_t n_before = current->size();
            const auto t_axis = std::chrono::high_resolution_clock::now();
            auto filtered = passthrough_filter_cpu(
                current, axis.limit_min, axis.limit_max, axis.dim);
            const auto t_axis_end = std::chrono::high_resolution_clock::now();
            const auto axis_ms =
                std::chrono::duration_cast<std::chrono::microseconds>(t_axis_end - t_axis).count();
            PT_INFO("[%c] CPU  %zu → %zu  [%.3f, %.3f]  %.2f ms",
                    static_cast<char>(std::toupper(axis.name)),
                    n_before,
                    filtered->size(),
                    axis.limit_min,
                    axis.limit_max,
                    axis_ms / 1000.0);

            if (debug_mode_ && axis.debug_pub) {
                char desc[64];
                std::snprintf(desc, sizeof(desc), "%c直通滤波",
                              static_cast<char>(std::toupper(axis.name)));
                publish_cloud(filtered, msg->header, axis.debug_pub, desc);
            }

            replace_current(current, filtered);
        }

        publish_cloud(current, msg->header, publisher_, "最终输出");
        memory_pool_->release(current);

        const auto t1 = std::chrono::high_resolution_clock::now();
        const auto total_ms =
            std::chrono::duration_cast<std::chrono::milliseconds>(t1 - t0).count();
        PT_INFO("点云处理完成，耗时 %ld ms", total_ms);
    }

    rclcpp::Subscription<sensor_msgs::msg::PointCloud2>::SharedPtr subscription_;
    rclcpp::Publisher<sensor_msgs::msg::PointCloud2>::SharedPtr publisher_;
    std::unique_ptr<PointCloudMemoryPool> memory_pool_;

    std::array<AxisSpec, 3> axes_{};
    std::vector<int> filter_order_;
    std::string filter_order_raw_;
    std::string input_topic_;
    std::string output_topic_;
    std::string debug_topic_x_;
    std::string debug_topic_y_;
    std::string debug_topic_z_;

    int memory_pool_size_{10};
    int memory_pool_reserve_{1000000};
    bool debug_mode_{false};
    double cloud_log_interval_sec_{1.0};

    bool frame_log_enabled_{false};
    std::chrono::steady_clock::time_point last_cloud_log_time_{};
};

int main(int argc, char* argv[])
{
    rclcpp::init(argc, argv);
    try {
        rclcpp::spin(std::make_shared<CloudPassthroughFilterNode>());
    } catch (const std::exception& ex) {
        RCLCPP_FATAL(rclcpp::get_logger("cloud_passthrough_filter"),
                     "节点启动失败: %s", ex.what());
        rclcpp::shutdown();
        return 1;
    }
    rclcpp::shutdown();
    return 0;
}
