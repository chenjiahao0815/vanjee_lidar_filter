#include "vanjee_lidar_filter/cloud_passthrough_filter.hpp"

#include <pcl_conversions/pcl_conversions.h>
#include <geometry_msgs/msg/point.hpp>
#include <sensor_msgs/msg/point_field.hpp>

#include <algorithm>
#include <cctype>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <stdexcept>
#include <unordered_map>
#include <unordered_set>
#include <vector>

#define PT_INFO(...) \
    do { if (shouldFrameLog()) { RCLCPP_INFO(this->get_logger(), __VA_ARGS__); } } while (0)
#define PT_WARN(...) \
    do { if (shouldFrameLog()) { RCLCPP_WARN(this->get_logger(), __VA_ARGS__); } } while (0)

PointCloudMemoryPool::PointCloudMemoryPool(size_t initial_pool_size, size_t reserve_size)
        : reserve_size_(reserve_size)
    {
        for (size_t i = 0; i < initial_pool_size; ++i) {
            auto cloud = std::make_shared<pcl::PointCloud<pcl::PointXYZ>>();
            cloud->points.reserve(reserve_size);
            in_pool_.insert(cloud.get());
            available_clouds_.push(cloud);
        }
    }

pcl::PointCloud<pcl::PointXYZ>::Ptr PointCloudMemoryPool::acquire()
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

void PointCloudMemoryPool::release(pcl::PointCloud<pcl::PointXYZ>::Ptr cloud)
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

CloudPassthroughFilterNode::CloudPassthroughFilterNode()
        : Node("cloud_passthrough_filter_node")
    {
        declare_and_load_parameters();

        memory_pool_ = std::make_unique<PointCloudMemoryPool>(
            static_cast<size_t>(memory_pool_size_),
            static_cast<size_t>(memory_pool_reserve_));

        // 雷达 /vanjee/lidar 是 Reliable；本节点订阅和所有发布都用 Reliable，才能和 RViz 配对
        const auto qos = rclcpp::QoS{5}.reliable();
        subscription_ = this->create_subscription<sensor_msgs::msg::PointCloud2>(
            input_topic_, qos,
            std::bind(&CloudPassthroughFilterNode::cloud_callback, this, std::placeholders::_1));
        publisher_ = this->create_publisher<sensor_msgs::msg::PointCloud2>(
            output_topic_, qos);

        if (debug_mode_) {
            axes_[0].debug_pub = this->create_publisher<sensor_msgs::msg::PointCloud2>(
                debug_topic_x_, qos);
            axes_[1].debug_pub = this->create_publisher<sensor_msgs::msg::PointCloud2>(
                debug_topic_y_, qos);
            axes_[2].debug_pub = this->create_publisher<sensor_msgs::msg::PointCloud2>(
                debug_topic_z_, qos);
        if (!removed_topic_.empty()) {
            removed_publisher_ = this->create_publisher<sensor_msgs::msg::PointCloud2>(
                removed_topic_, qos);
        }
        const auto viz_qos = qos;
        if (!verdict_topic_.empty()) {
            verdict_publisher_ = this->create_publisher<sensor_msgs::msg::PointCloud2>(
                verdict_topic_, viz_qos);
        }
        if (!boxes_topic_.empty()) {
            boxes_publisher_ = this->create_publisher<visualization_msgs::msg::MarkerArray>(
                boxes_topic_, viz_qos);
        }
        if (publish_occupied_voxels_ && !voxels_topic_.empty()) {
            voxels_publisher_ = this->create_publisher<visualization_msgs::msg::MarkerArray>(
                voxels_topic_, viz_qos);
        }
    }

    grid_.reserve(4096);
    fine_grid_.reserve(4096);
        log_startup();
    }

void CloudPassthroughFilterNode::declare_and_load_parameters()
    {
        this->declare_parameter("input_topic", std::string("input_cloud"));
        this->declare_parameter("output_topic", std::string("filtered_cloud"));
    this->declare_parameter("removed_topic", std::string("removed_cloud"));
        this->declare_parameter("filter_order", std::string("xyz"));
        this->declare_parameter("enable_x", true);
        this->declare_parameter("enable_y", true);
        this->declare_parameter("enable_z", true);
    this->declare_parameter("limit_min_x", -2.0);
    this->declare_parameter("limit_max_x", 2.0);
        this->declare_parameter("limit_min_y", -2.0);
        this->declare_parameter("limit_max_y", 2.0);
    this->declare_parameter("limit_min_z", -1.0);
    this->declare_parameter("limit_max_z", 1.0);
    this->declare_parameter("debug_mode", true);
        this->declare_parameter("debug_topic_x", std::string("passthrough_cloud_x"));
        this->declare_parameter("debug_topic_y", std::string("passthrough_cloud_y"));
        this->declare_parameter("debug_topic_z", std::string("passthrough_cloud_z"));
        this->declare_parameter("memory_pool_size", 10);
        this->declare_parameter("memory_pool_reserve", 1000000);
        this->declare_parameter("cloud_log_interval_sec", 1.0);

    this->declare_parameter("enable_voxel_filter", true);
    this->declare_parameter("cube_length", 4.0);
    this->declare_parameter("cube_width", 4.0);
    this->declare_parameter("cube_height", 2.0);
    this->declare_parameter("ang_h", 0.0034906585);   // 约 0.2 deg
    this->declare_parameter("ang_v", 0.0174532925);   // 约 1.0 deg
    this->declare_parameter("r_max", 8.0);
    this->declare_parameter("base_xy", 0.01);
    this->declare_parameter("base_z", 0.01);
    this->declare_parameter("max_xy", 0.5);
    this->declare_parameter("max_z", 0.5);
    this->declare_parameter("size_z_ratio", 4.0);
    this->declare_parameter("size_z_min", 0.1);
    this->declare_parameter("thr_ratio", 0.5);
    this->declare_parameter("thr_z_min", 0.03);
    this->declare_parameter("cluster_link_m", 0.18);
    this->declare_parameter("cluster_link_k", 6.0);
    this->declare_parameter("cluster_plane_k", 10.0);
    this->declare_parameter("min_cluster_rings", 2);
    this->declare_parameter("cluster_min_h", 0.08);
    this->declare_parameter("cluster_aspect_k", 2.0);
    this->declare_parameter("cluster_wipe_ratio", 0.7);
    this->declare_parameter("max_cluster_rings", 8);
    this->declare_parameter("verdict_topic", std::string("/vanjee/filter_verdict"));
    this->declare_parameter("boxes_topic", std::string("/vanjee/filter_boxes"));
    this->declare_parameter("voxels_topic", std::string("/vanjee/voxel_markers"));
    this->declare_parameter("publish_occupied_voxels", true);

        input_topic_ = this->get_parameter("input_topic").as_string();
        output_topic_ = this->get_parameter("output_topic").as_string();
    removed_topic_ = this->get_parameter("removed_topic").as_string();
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

    enable_voxel_filter_ = this->get_parameter("enable_voxel_filter").as_bool();
    cube_length_ = this->get_parameter("cube_length").as_double();
    cube_width_ = this->get_parameter("cube_width").as_double();
    cube_height_ = this->get_parameter("cube_height").as_double();
    ang_h_ = this->get_parameter("ang_h").as_double();
    ang_v_ = this->get_parameter("ang_v").as_double();
    r_max_ = this->get_parameter("r_max").as_double();
    base_xy_ = this->get_parameter("base_xy").as_double();
    base_z_ = this->get_parameter("base_z").as_double();
    max_xy_ = this->get_parameter("max_xy").as_double();
    max_z_ = this->get_parameter("max_z").as_double();
    size_z_ratio_ = this->get_parameter("size_z_ratio").as_double();
    if (size_z_ratio_ <= 0.0) {
        size_z_ratio_ = 4.0;
    }
    size_z_min_ = this->get_parameter("size_z_min").as_double();
    if (size_z_min_ <= 0.0) {
        size_z_min_ = 0.1;
    }
    thr_ratio_ = this->get_parameter("thr_ratio").as_double();
    thr_z_min_ = this->get_parameter("thr_z_min").as_double();
    if (thr_z_min_ < 0.0) {
        thr_z_min_ = 0.0;
    }
    cluster_link_m_ = this->get_parameter("cluster_link_m").as_double();
    cluster_link_k_ = this->get_parameter("cluster_link_k").as_double();
    cluster_plane_k_ = this->get_parameter("cluster_plane_k").as_double();
    if (cluster_link_m_ <= 0.0) {
        cluster_link_m_ = 0.18;
    }
    if (cluster_link_k_ <= 0.0) {
        cluster_link_k_ = 6.0;
    }
    if (cluster_plane_k_ <= 0.0) {
        cluster_plane_k_ = 10.0;
    }
    min_cluster_rings_ = this->get_parameter("min_cluster_rings").as_int();
    if (min_cluster_rings_ < 1) {
        min_cluster_rings_ = 2;
    }
    cluster_min_h_ = this->get_parameter("cluster_min_h").as_double();
    if (cluster_min_h_ <= 0.0) {
        cluster_min_h_ = 0.08;
    }
    cluster_aspect_k_ = this->get_parameter("cluster_aspect_k").as_double();
    if (cluster_aspect_k_ < 1.0) {
        cluster_aspect_k_ = 2.0;
    }
    cluster_wipe_ratio_ = this->get_parameter("cluster_wipe_ratio").as_double();
    if (cluster_wipe_ratio_ <= 0.0 || cluster_wipe_ratio_ > 1.0) {
        cluster_wipe_ratio_ = 0.7;
    }
    max_cluster_rings_ = this->get_parameter("max_cluster_rings").as_int();
    if (max_cluster_rings_ < min_cluster_rings_) {
        max_cluster_rings_ = min_cluster_rings_;
    }
    verdict_topic_ = this->get_parameter("verdict_topic").as_string();
    boxes_topic_ = this->get_parameter("boxes_topic").as_string();
    voxels_topic_ = this->get_parameter("voxels_topic").as_string();
    publish_occupied_voxels_ = this->get_parameter("publish_occupied_voxels").as_bool();

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

bool CloudPassthroughFilterNode::parse_filter_order(
    const std::string& raw, std::vector<int>& order) const
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

std::string CloudPassthroughFilterNode::order_string() const
    {
        std::string s;
        s.reserve(filter_order_.size());
        for (int dim : filter_order_) {
            s.push_back(axes_[static_cast<size_t>(dim)].name);
        }
        return s;
    }

void CloudPassthroughFilterNode::log_startup() const
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
        RCLCPP_INFO(this->get_logger(), "  被删点话题: %s", removed_topic_.c_str());
        RCLCPP_INFO(this->get_logger(), "  判定配色: %s", verdict_topic_.c_str());
        RCLCPP_INFO(this->get_logger(), "  范围线框: %s", boxes_topic_.c_str());
        RCLCPP_INFO(this->get_logger(), "  有点体素: %s (%s)",
                    voxels_topic_.c_str(),
                    publish_occupied_voxels_ ? "开" : "关");
        }
        RCLCPP_INFO(this->get_logger(),
                    "  帧日志间隔: %.2f s (仅 debug_mode=false 时限频)",
                    cloud_log_interval_sec_);
    RCLCPP_INFO(this->get_logger(), "  体素删点: %s", enable_voxel_filter_ ? "开" : "关");
    if (enable_voxel_filter_) {
        RCLCPP_INFO(this->get_logger(),
                    "  我们的雷达中心立方体: L=%.2f W=%.2f H=%.2f",
                    cube_length_, cube_width_, cube_height_);
        RCLCPP_INFO(this->get_logger(),
                    "  角分辨率 ang_h=%.6f ang_v=%.6f, 细格 base_xy=%.3f base_z=%.3f, "
                    "横向=1×线间距 纵向=%.1f×线间距夹[%.3f,%.3f], thr_ratio=%.2f thr_z_min=%.3f, "
                    "连通团 link=%.2f m k3d=%.1f k_xy=%.1f",
                    ang_h_, ang_v_, base_xy_, base_z_, size_z_ratio_, size_z_min_, max_z_,
                    thr_ratio_, thr_z_min_,
                    cluster_link_m_, cluster_link_k_, cluster_plane_k_);
        RCLCPP_INFO(this->get_logger(),
                    "  团保护: 实物最小高度 cluster_min_h=%.3f m, "
                    "团高≥%.1f×团内最大size_z, 坏格≥%.0f%%整团扫, 线数门槛夹在 [%d, %d]",
                    cluster_min_h_, cluster_aspect_k_, 100.0 * cluster_wipe_ratio_,
                    min_cluster_rings_, max_cluster_rings_);
        for (double r : {0.3, 0.5, 1.0, 2.0, 2.83}) {
            RCLCPP_INFO(this->get_logger(),
                        "    r=%.2f 线间距=%.4f m → 至少 %d 根线",
                        r, r * ang_v_, requiredRings(r));
        }
        logVoxelScaleSamples();
    }
}

bool CloudPassthroughFilterNode::shouldFrameLog() const
    {
        return debug_mode_ || frame_log_enabled_;
    }

void CloudPassthroughFilterNode::updateFrameLogGate()
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

void CloudPassthroughFilterNode::removeNonFinitePointsInPlace(
    pcl::PointCloud<pcl::PointXYZ>::Ptr cloud)
    {
        if (!cloud || cloud->empty()) {
            return;
        }
        const std::size_t before = cloud->points.size();
        // 线号数组要和点一起压缩，否则后面按下标取线号会错位
        const bool track_ring = have_ring_ && ring_of_cur_.size() == before;
        std::size_t write = 0;
        for (std::size_t i = 0; i < before; ++i) {
            const auto& p = cloud->points[i];
            if (!std::isfinite(p.x) || !std::isfinite(p.y) || !std::isfinite(p.z)) {
                continue;
            }
            if (write != i) {
                cloud->points[write] = p;
            }
            if (track_ring) {
                ring_of_cur_[write] = ring_of_cur_[i];
            }
            ++write;
        }
        if (write == before) {
            cloud->width = static_cast<uint32_t>(cloud->points.size());
            cloud->height = 1;
            cloud->is_dense = true;
            return;
        }
        cloud->points.resize(write);
        if (track_ring) {
            ring_of_cur_.resize(write);
        }
        cloud->width = static_cast<uint32_t>(cloud->points.size());
        cloud->height = 1;
        cloud->is_dense = true;
        PT_WARN("移除非法点(NaN/Inf): %zu → %zu", before, cloud->points.size());
    }

bool CloudPassthroughFilterNode::loadRingField(const sensor_msgs::msg::PointCloud2& msg)
    {
        ring_of_cur_.clear();
        have_ring_ = false;

        const sensor_msgs::msg::PointField* ring_field = nullptr;
        for (const auto& f : msg.fields) {
            if (f.name == "ring") {
                ring_field = &f;
                break;
            }
        }
        if (ring_field == nullptr || msg.point_step == 0) {
            return false;
        }

        const size_t n = static_cast<size_t>(msg.width) * static_cast<size_t>(msg.height);
        if (msg.data.size() < n * msg.point_step) {
            return false;
        }
        ring_of_cur_.resize(n);
        for (size_t i = 0; i < n; ++i) {
            const uint8_t* src = msg.data.data() + i * msg.point_step + ring_field->offset;
            uint16_t r = 0;
            switch (ring_field->datatype) {
                case sensor_msgs::msg::PointField::UINT8:
                case sensor_msgs::msg::PointField::INT8:
                    r = static_cast<uint16_t>(*src);
                    break;
                case sensor_msgs::msg::PointField::UINT16:
                case sensor_msgs::msg::PointField::INT16: {
                    uint16_t v = 0;
                    std::memcpy(&v, src, sizeof(v));
                    r = v;
                    break;
                }
                case sensor_msgs::msg::PointField::UINT32:
                case sensor_msgs::msg::PointField::INT32: {
                    uint32_t v = 0;
                    std::memcpy(&v, src, sizeof(v));
                    r = static_cast<uint16_t>(v);
                    break;
                }
                case sensor_msgs::msg::PointField::FLOAT32: {
                    float v = 0.0f;
                    std::memcpy(&v, src, sizeof(v));
                    r = static_cast<uint16_t>(std::max(0.0f, v));
                    break;
                }
                default:
                    ring_of_cur_.clear();
                    return false;
            }
            ring_of_cur_[i] = r;
        }
        have_ring_ = true;
        return true;
    }

void CloudPassthroughFilterNode::publish_cloud(
    const pcl::PointCloud<pcl::PointXYZ>::Ptr& cloud,
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

    pcl::PointCloud<pcl::PointXYZ>::Ptr
CloudPassthroughFilterNode::passthrough_filter_cpu(
    const pcl::PointCloud<pcl::PointXYZ>::Ptr& cloud,
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
        const bool track_ring = have_ring_ && ring_of_cur_.size() == cloud->size();
        if (track_ring) {
            ring_scratch_.clear();
            ring_scratch_.reserve(cloud->size());
        }
        const std::size_t n = cloud->points.size();
        for (std::size_t i = 0; i < n; ++i) {
            const auto& p = cloud->points[i];
            const float value = (dimension == 0) ? p.x : ((dimension == 1) ? p.y : p.z);
            if (value >= lower_limit && value <= upper_limit) {
                filtered->points.push_back(p);
                if (track_ring) {
                    ring_scratch_.push_back(ring_of_cur_[i]);
                }
            }
        }
        if (track_ring) {
            ring_of_cur_.swap(ring_scratch_);
        }
        filtered->width = static_cast<uint32_t>(filtered->points.size());
        return filtered;
    }

void CloudPassthroughFilterNode::replace_current(
    pcl::PointCloud<pcl::PointXYZ>::Ptr& current,
                         pcl::PointCloud<pcl::PointXYZ>::Ptr next)
    {
        if (current && current != next) {
            memory_pool_->release(current);
        }
        current = std::move(next);
    }

void CloudPassthroughFilterNode::cloud_callback(
    const sensor_msgs::msg::PointCloud2::SharedPtr msg)
    {
        updateFrameLogGate();
        const auto t0 = std::chrono::high_resolution_clock::now();

        auto current = memory_pool_->acquire();
        pcl::fromROSMsg(*msg, *current);
        // 线号必须从原始消息取，pcl::PointXYZ 会把 ring 字段丢掉
        loadRingField(*msg);

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

    pcl::PointCloud<pcl::PointXYZ>::Ptr raw_viz;
    if (debug_mode_) {
        raw_viz = memory_pool_->acquire();
        *raw_viz = *current;
    }

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

    // 直通滤波到这里结束，current 是直通后的点云

    std::vector<Voxel*> bad_voxels;
    if (enable_voxel_filter_ && current && !current->empty()) {
        grid_.clear();
        fine_grid_.clear();
        buildVoxelGrid(current);
        const size_t n_before_voxel = current->size();
        bad_voxels = collectBadVoxels();
        if (debug_mode_) {
            publishFilterDebug(raw_viz, current, bad_voxels, msg->header);
        }
        removePointsInBadVoxels(current, bad_voxels, msg->header);
        PT_INFO("体素删点结束: 坏格 %zu, 点数 %zu → %zu",
                bad_voxels.size(), n_before_voxel, current->size());
    } else if (debug_mode_) {
        publishFilterDebug(raw_viz, current, bad_voxels, msg->header);
        }

        publish_cloud(current, msg->header, publisher_, "最终输出");
    if (raw_viz) {
        memory_pool_->release(raw_viz);
    }
        memory_pool_->release(current);

        const auto t1 = std::chrono::high_resolution_clock::now();
        const auto total_ms =
            std::chrono::duration_cast<std::chrono::milliseconds>(t1 - t0).count();
        PT_INFO("点云处理完成，耗时 %ld ms", total_ms);
    }

// ---------- 体素删点 ----------

void CloudPassthroughFilterNode::logVoxelScaleSamples() const
{
    const double samples[] = {1.0, 2.0, 4.0, std::max(r_max_, 1.0)};
    for (double r : samples) {
        const VoxelScale s = computeVoxelScale(r);
        RCLCPP_INFO(this->get_logger(),
                    "  样例 r=%.1f line_gap=%.4f mult_xy=%d mult_z=%d "
                    "size_xy=%.3f size_z=%.3f thr_z=%.4f",
                    r, s.line_gap, s.mult_xy, s.mult_z, s.size_xy, s.size_z, s.thr_z);
    }
}

VoxelScale CloudPassthroughFilterNode::computeVoxelScale(double r) const
{
    VoxelScale s;  //输入和基础值保护      根据点云距离雷达的远近来决定用多大的体素来装点云
    const double rr = std::max(r, 1e-3);
    const double base_xy = std::max(base_xy_, 1e-6);
    const double base_z = std::max(base_z_, 1e-6);

    // 用这个间距来决定格子应该多大
    s.line_gap = rr * ang_v_;
    s.pt_gap = s.line_gap;
    s.size_xy = std::min(std::max(s.line_gap, base_xy), max_xy_);
    // 纵向按线间距放大：近处有 size_z_min 托底，远处顶到 max_z
    s.size_z = std::min(std::max(size_z_ratio_ * s.line_gap, size_z_min_), max_z_);
    s.mult_xy = static_cast<int>(std::max(1.0, std::ceil(s.size_xy / base_xy)));
    s.mult_z = static_cast<int>(std::max(1.0, std::ceil(s.size_z / base_z)));
    s.size_xy = std::min(base_xy * static_cast<double>(s.mult_xy), max_xy_);
    s.size_z = std::min(base_z * static_cast<double>(s.mult_z), max_z_);

    s.thr_z = s.line_gap * thr_ratio_;
    if (s.thr_z < thr_z_min_) {
        s.thr_z = thr_z_min_;
    }
    if (s.thr_z >= s.size_z) {
        s.thr_z = 0.5 * s.size_z;
    }

    s.inv_xy = 1.0 / std::max(s.size_xy, 1e-6);
    s.inv_z = 1.0 / std::max(s.size_z, 1e-6);
    return s;
}

bool CloudPassthroughFilterNode::inOurCube(float x, float y, float z) const
{
    // 以雷达原点为中心的立方体
    const double half_l = 0.5 * cube_length_;
    const double half_w = 0.5 * cube_width_;
    const double half_h = 0.5 * cube_height_;
    return std::fabs(static_cast<double>(x)) <= half_l &&
           std::fabs(static_cast<double>(y)) <= half_w &&
           std::fabs(static_cast<double>(z)) <= half_h;
}

bool CloudPassthroughFilterNode::inPassthrough(float x, float y, float z) const
{
    return static_cast<double>(x) >= axes_[0].limit_min &&
           static_cast<double>(x) <= axes_[0].limit_max &&
           static_cast<double>(y) >= axes_[1].limit_min &&
           static_cast<double>(y) <= axes_[1].limit_max &&
           static_cast<double>(z) >= axes_[2].limit_min &&
           static_cast<double>(z) <= axes_[2].limit_max;
}

void CloudPassthroughFilterNode::buildVoxelGrid(
    const pcl::PointCloud<pcl::PointXYZ>::Ptr& cloud)
{
    if (!cloud || cloud->empty()) {
        return;
    }

    // 只把落在立方体里的点装进细格；框外的点不做操作，后面原样留下
    const double inv_xy = 1.0 / std::max(base_xy_, 1e-6);
    const double inv_z = 1.0 / std::max(base_z_, 1e-6);
    const uint32_t n = static_cast<uint32_t>(cloud->points.size());
    uint32_t loaded = 0;
    uint32_t skipped_outside = 0;

    for (uint32_t i = 0; i < n; ++i) {
        const auto& p = cloud->points[i];
        if (!std::isfinite(p.x) || !std::isfinite(p.y) || !std::isfinite(p.z)) {
            continue;
        }

        if (!inOurCube(p.x, p.y, p.z)) {
            ++skipped_outside;
            continue;
        }

        const int ix = static_cast<int>(std::floor(static_cast<double>(p.x) * inv_xy));
        const int iy = static_cast<int>(std::floor(static_cast<double>(p.y) * inv_xy));
        const int iz = static_cast<int>(std::floor(static_cast<double>(p.z) * inv_z));
        const int64_t key = makeKey(ix, iy, iz);

        auto it = fine_grid_.find(key);
        if (it == fine_grid_.end()) {
            Voxel v;
            v.ix = ix;
            v.iy = iy;
            v.iz = iz;
            v.count = 1;
            v.zmin = p.z;
            v.zmax = p.z;
            v.cx = static_cast<float>((static_cast<double>(ix) + 0.5) * base_xy_);
            v.cy = static_cast<float>((static_cast<double>(iy) + 0.5) * base_xy_);
            v.idx.push_back(i);
            v.keep = true;
            addRing(v, i, p.x, p.y, p.z);
            fine_grid_.emplace(key, std::move(v));
        } else {
            Voxel& v = it->second;
            ++v.count;
            if (p.z < v.zmin) {
                v.zmin = p.z;
            }
            if (p.z > v.zmax) {
                v.zmax = p.z;
            }
            v.idx.push_back(i);
            addRing(v, i, p.x, p.y, p.z);
        }
        ++loaded;
    }

    mergeFineVoxels();
    PT_INFO("体素装格: 框内 %u 点, 框外跳过 %u, 细格 %zu, 合成后 %zu 格",
            loaded, skipped_outside, fine_grid_.size(), grid_.size());
}

void CloudPassthroughFilterNode::mergeFineVoxels()
{
    struct ScaleN {
        int nxy{1};
        int nz{1};
    };
    std::unordered_map<int64_t, ScaleN> scale_n;
    scale_n.reserve(fine_grid_.size());

    for (const auto& entry : fine_grid_) {
        const Voxel& v = entry.second;
        const double cx = (static_cast<double>(v.ix) + 0.5) * base_xy_;
        const double cy = (static_cast<double>(v.iy) + 0.5) * base_xy_;
        const double r = std::hypot(cx, cy);
        const VoxelScale s = computeVoxelScale(r);
        scale_n[entry.first] = ScaleN{s.mult_xy, s.mult_z};
    }

    for (auto& entry : fine_grid_) {
        const Voxel& fine = entry.second;
        ScaleN n = scale_n[entry.first];

        for (int dx = -1; dx <= 1; ++dx) {
            for (int dy = -1; dy <= 1; ++dy) {
                for (int dz = -1; dz <= 1; ++dz) {
                    if (dx == 0 && dy == 0 && dz == 0) {
                        continue;
                    }
                    const int64_t nk = makeKey(fine.ix + dx, fine.iy + dy, fine.iz + dz);
                    auto nit = scale_n.find(nk);
                    if (nit == scale_n.end()) {
                        continue;
                    }
                    if (nit->second.nxy > n.nxy) {
                        n.nxy = nit->second.nxy;
                    }
                    if (nit->second.nz > n.nz) {
                        n.nz = nit->second.nz;
                    }
                }
            }
        }

        const int cx = floorDiv(fine.ix, n.nxy);
        const int cy = floorDiv(fine.iy, n.nxy);
        const int cz = floorDiv(fine.iz, n.nz);
        const int64_t ckey = makeKey(cx, cy, cz);

        const double center_x = (static_cast<double>(fine.ix) + 0.5) * base_xy_;
        const double center_y = (static_cast<double>(fine.iy) + 0.5) * base_xy_;
        const float cell_r = static_cast<float>(std::hypot(center_x, center_y));

        auto it = grid_.find(ckey);
        if (it == grid_.end()) {
            Voxel v;
            v.ix = cx;
            v.iy = cy;
            v.iz = cz;
            v.count = fine.count;
            v.zmin = fine.zmin;
            v.zmax = fine.zmax;
            v.r = cell_r;
            v.cx = static_cast<float>(center_x);
            v.cy = static_cast<float>(center_y);
            v.ring_mask = fine.ring_mask;
            v.idx = fine.idx;
            v.keep = true;
            grid_.emplace(ckey, std::move(v));
        } else {
            Voxel& v = it->second;
            const double w_old = static_cast<double>(v.count);
            const double w_new = static_cast<double>(fine.count);
            const double w = std::max(w_old + w_new, 1.0);
            v.cx = static_cast<float>((static_cast<double>(v.cx) * w_old + center_x * w_new) / w);
            v.cy = static_cast<float>((static_cast<double>(v.cy) * w_old + center_y * w_new) / w);
            v.count += fine.count;
            if (fine.zmin < v.zmin) {
                v.zmin = fine.zmin;
            }
            if (fine.zmax > v.zmax) {
                v.zmax = fine.zmax;
            }
            if (cell_r > v.r) {
                v.r = cell_r;
            }
            v.ring_mask |= fine.ring_mask;
            v.idx.insert(v.idx.end(), fine.idx.begin(), fine.idx.end());
        }
    }

    for (auto& entry : grid_) {
        Voxel& v = entry.second;
        const VoxelScale s = computeVoxelScale(static_cast<double>(v.r));
        v.thr_z = static_cast<float>(s.thr_z);
        v.size_xy = static_cast<float>(s.size_xy);
        v.size_z = static_cast<float>(s.size_z);
    }
}

int CloudPassthroughFilterNode::floorDiv(int a, int b) const
{
    const int d = std::max(b, 1);
    if (a >= 0) {
        return a / d;
    }
    return static_cast<int>(std::floor(static_cast<double>(a) / static_cast<double>(d)));
}

int64_t CloudPassthroughFilterNode::makeKey(int ix, int iy, int iz) const
{
    constexpr int64_t kOffset = 1 << 19;
    const int64_t ux = static_cast<int64_t>(ix) + kOffset;
    const int64_t uy = static_cast<int64_t>(iy) + kOffset;
    const int64_t uz = static_cast<int64_t>(iz) + kOffset;
    return (ux << 40) | (uy << 20) | uz;
}

int CloudPassthroughFilterNode::ringId(float x, float y, float z) const
{
    const double rxy = std::hypot(static_cast<double>(x), static_cast<double>(y));
    return static_cast<int>(std::llround(
        std::atan2(static_cast<double>(z), std::max(rxy, 1e-6)) / std::max(ang_v_, 1e-6)));
}

int CloudPassthroughFilterNode::ringOfPoint(uint32_t i, float x, float y, float z) const
{
    if (have_ring_ && static_cast<size_t>(i) < ring_of_cur_.size()) {
        return static_cast<int>(ring_of_cur_[i]);
    }
    // 退路：atan2 只是仰角，近处同一根线扫到平面会被算成好几根，别当准值用
    return ringId(x, y, z);
}

void CloudPassthroughFilterNode::addRing(Voxel& v, uint32_t i, float x, float y, float z) const
{
    constexpr int kBias = 32;
    const int ring = ringOfPoint(i, x, y, z);
    const int bit = have_ring_ ? ring : (ring + kBias);
    if (bit >= 0 && bit < 64) {
        v.ring_mask |= (1ull << bit);
    }
}

int CloudPassthroughFilterNode::requiredRings(double r) const
{
    // 线间距随距离长大：同一个物理高度，近处要占更多根线，远处只占一两根
    const double gap = std::max(std::max(r, 1e-3) * ang_v_, 1e-6);
    const int n = static_cast<int>(std::ceil(cluster_min_h_ / gap));
    const int lo = std::max(1, min_cluster_rings_);
    const int hi = std::max(lo, max_cluster_rings_);
    return std::min(std::max(n, lo), hi);
}

int CloudPassthroughFilterNode::ringBitCount(uint64_t mask)
{
    int n = 0;
    while (mask != 0ull) {
        n += static_cast<int>(mask & 1ull);
        mask >>= 1;
    }
    return n;
}

void CloudPassthroughFilterNode::markMultiRingClusters(std::vector<Voxel*>& all)
{
    const size_t n = all.size();
    for (Voxel* v : all) {
        if (v) {
            v->keep = false;
            v->cluster_id = 0;
            v->cluster_rings = 0;
        }
    }
    if (n == 0) {
        return;
    }

    std::vector<int> parent(static_cast<int>(n));
    for (size_t i = 0; i < n; ++i) {
        parent[i] = static_cast<int>(i);
    }
    auto find = [&](int x) {
        while (parent[static_cast<size_t>(x)] != x) {
            parent[static_cast<size_t>(x)] = parent[static_cast<size_t>(parent[static_cast<size_t>(x)])];
            x = parent[static_cast<size_t>(x)];
        }
        return x;
    };
    auto unite = [&](int a, int b) {
        a = find(a);
        b = find(b);
        if (a != b) {
            parent[static_cast<size_t>(b)] = a;
        }
    };

    const float k3d = static_cast<float>(cluster_link_k_);
    const float kxy = static_cast<float>(cluster_plane_k_);
    const float link_min = static_cast<float>(cluster_link_m_);
    const float ang = static_cast<float>(ang_v_);
    for (size_t i = 0; i < n; ++i) {
        const Voxel* a = all[i];
        const float za = 0.5f * (a->zmin + a->zmax);
        for (size_t j = i + 1; j < n; ++j) {
            const Voxel* b = all[j];
            const float dx = a->cx - b->cx;
            const float dy = a->cy - b->cy;
            const float zb = 0.5f * (b->zmin + b->zmax);
            const float dz = za - zb;
            const float rr = std::max(std::max(a->r, b->r), 1e-3f);
            const float gap = rr * ang;
            const float link3d = std::max(link_min, k3d * gap);
            const float dxy2 = dx * dx + dy * dy;
            const float d3 = dxy2 + dz * dz;
            bool join = (d3 <= link3d * link3d);
            if (!join) {
                const float z_band = std::max(0.04f, 2.0f * gap);
                const float link_xy = std::max(link3d, kxy * gap);
                if (std::fabs(dz) <= z_band && dxy2 <= link_xy * link_xy) {
                    join = true;
                }
            }
            if (join) {
                unite(static_cast<int>(i), static_cast<int>(j));
            }
        }
    }

    std::vector<uint64_t> root_mask(n, 0ull);
    std::vector<int> root_size(n, 0);
    std::vector<float> root_rmin(n, 1e9f);
    std::vector<float> root_zmin(n, 1e9f);
    std::vector<float> root_zmax(n, -1e9f);
    std::vector<float> root_szmax(n, 0.0f);
    for (size_t i = 0; i < n; ++i) {
        const int r = find(static_cast<int>(i));
        const size_t ri = static_cast<size_t>(r);
        root_mask[ri] |= all[i]->ring_mask;
        ++root_size[ri];
        root_rmin[ri] = std::min(root_rmin[ri], all[i]->r);
        root_zmin[ri] = std::min(root_zmin[ri], all[i]->zmin);
        root_zmax[ri] = std::max(root_zmax[ri], all[i]->zmax);
        root_szmax[ri] = std::max(root_szmax[ri], all[i]->size_z);
    }

    // 线数门槛按团里最近的那格算，最近处最严
    std::vector<int> root_need(n, 0);
    std::vector<char> root_ok(n, 0);
    for (size_t i = 0; i < n; ++i) {
        if (find(static_cast<int>(i)) != static_cast<int>(i)) {
            continue;
        }
        root_need[i] = requiredRings(static_cast<double>(root_rmin[i]));
        const float span = root_zmax[i] - root_zmin[i];
        const bool rings_ok = ringBitCount(root_mask[i]) >= root_need[i];
        // 线数够只说明被扫到得多；还得真的立起来 cluster_min_h 才算实物
        const bool span_ok = span >= static_cast<float>(cluster_min_h_);
        // 团高还得 ≥ 团内最高格子 size_z 的 k 倍，否则是横着摊的饼
        const float need_h = static_cast<float>(cluster_aspect_k_) *
                             std::max(root_szmax[i], 1e-4f);
        const bool aspect_ok = span >= need_h;
        root_ok[i] = (rings_ok && span_ok && aspect_ok) ? 1 : 0;
    }

    // 根 -> 团号 1,2,3...；至少两格才算团
    std::vector<int> root_to_id(n, 0);
    int next_id = 1;
    for (size_t i = 0; i < n; ++i) {
        if (find(static_cast<int>(i)) != static_cast<int>(i)) {
            continue;
        }
        if (root_size[i] >= 2) {
            root_to_id[i] = next_id++;
        }
    }

    size_t multi_n = 0;
    size_t protect_voxels = 0;
    size_t protect_pts = 0;
    size_t drop_thin = 0;     // 线数够但绝对高度不够
    size_t drop_pancake = 0;  // 线数够、绝对高度够，但相对格子是饼
    size_t drop_rings = 0;    // 这个距离上线数不够
    for (size_t i = 0; i < n; ++i) {
        if (find(static_cast<int>(i)) != static_cast<int>(i)) {
            continue;
        }
        if (root_ok[i] != 0) {
            ++multi_n;
            continue;
        }
        if (ringBitCount(root_mask[i]) < root_need[i]) {
            ++drop_rings;
            continue;
        }
        const float span = root_zmax[i] - root_zmin[i];
        if (span < static_cast<float>(cluster_min_h_)) {
            ++drop_thin;
        } else {
            ++drop_pancake;
        }
    }
    for (size_t i = 0; i < n; ++i) {
        const size_t r = static_cast<size_t>(find(static_cast<int>(i)));
        all[i]->cluster_id = root_to_id[r];
        all[i]->cluster_rings = ringBitCount(root_mask[r]);
        all[i]->need_rings = root_need[r];
        all[i]->cluster_span = root_zmax[r] - root_zmin[r];
        if (root_ok[r] != 0) {
            all[i]->keep = true;
            ++protect_voxels;
            protect_pts += all[i]->count;
        }
    }
    PT_INFO("连通团保护: 过关团 %zu, 太扁毙 %zu, 饼毙 %zu, 线不够毙 %zu, 编号团 %d, "
            "保护 %zu 格 / %zu 点 (线号=%s, min_h=%.3f, 团高≥%.1f×size_z, 线数 %d~%d, "
            "link=%.2f k3d=%.1f k_xy=%.1f)",
            multi_n, drop_thin, drop_pancake, drop_rings, next_id - 1,
            protect_voxels, protect_pts,
            have_ring_ ? "真实ring" : "atan2估计", cluster_min_h_, cluster_aspect_k_,
            min_cluster_rings_, max_cluster_rings_,
            cluster_link_m_, cluster_link_k_, cluster_plane_k_);
}

std::vector<Voxel*> CloudPassthroughFilterNode::collectBadVoxels()
{
    std::vector<Voxel*> all;
    all.reserve(grid_.size());
    for (auto& entry : grid_) {
        all.push_back(&entry.second);
    }
    markMultiRingClusters(all);

    std::vector<Voxel*> bad;
    bad.reserve(grid_.size());

    size_t cluster_n = 0;
    size_t thick_n = 0;
    float sample_span = 0.0f;
    float sample_thr = 0.0f;
    bool have_sample = false;

    for (Voxel* vp : all) {
        Voxel& v = *vp;
        if (v.keep) {
            ++cluster_n;
            continue;
        }
        // 没团保护：只看本格厚度，够厚就留，不够就删
        const float span = v.zmax - v.zmin;
        if (span >= v.thr_z) {
            ++thick_n;
            continue;
        }
        bad.push_back(&v);
        if (!have_sample) {
            sample_span = span;
            sample_thr = v.thr_z;
            have_sample = true;
        }
    }

    // 编号团里坏格太多：整团扫光，别留几颗孤点
    std::unordered_set<const Voxel*> bad_set;
    bad_set.reserve(bad.size() * 2 + 1);
    for (const Voxel* v : bad) {
        bad_set.insert(v);
    }
    std::unordered_map<int, std::vector<Voxel*>> by_cid;
    by_cid.reserve(64);
    for (Voxel* vp : all) {
        if (vp && vp->cluster_id > 0) {
            by_cid[vp->cluster_id].push_back(vp);
        }
    }
    size_t wipe_clusters = 0;
    size_t wipe_extra = 0;
    for (auto& entry : by_cid) {
        const auto& members = entry.second;
        if (members.empty()) {
            continue;
        }
        size_t n_bad = 0;
        for (const Voxel* v : members) {
            if (bad_set.count(v) != 0) {
                ++n_bad;
            }
        }
        const double ratio = static_cast<double>(n_bad) /
                             static_cast<double>(members.size());
        if (ratio + 1e-12 < cluster_wipe_ratio_) {
            continue;
        }
        ++wipe_clusters;
        for (Voxel* v : members) {
            if (bad_set.count(v) != 0) {
                continue;
            }
            // 团保护过关的不扫；只清「单格厚留」留下来的碎渣
            if (v->keep) {
                continue;
            }
            bad.push_back(v);
            bad_set.insert(v);
            ++wipe_extra;
            if (thick_n > 0) {
                --thick_n;
            }
        }
    }

    PT_INFO("坏体素判定: 总格 %zu, 团保护 %zu, 厚留 %zu, 坏 %zu "
            "(整团扫 %zu 个编号团, 多删 %zu 格, 门槛=%.0f%%)",
            grid_.size(), cluster_n, thick_n, bad.size(),
            wipe_clusters, wipe_extra, 100.0 * cluster_wipe_ratio_);
    if (have_sample) {
        PT_INFO("坏体素样例: span=%.4f < thr_z=%.4f", sample_span, sample_thr);
    }
    return bad;
}

void CloudPassthroughFilterNode::removePointsInBadVoxels(
    pcl::PointCloud<pcl::PointXYZ>::Ptr& cloud,
    const std::vector<Voxel*>& bad_voxels,
    const std_msgs::msg::Header& header)
{
    if (!cloud) {
        PT_WARN("删点: 点云为空, 跳过");
        return;
    }
    if (bad_voxels.empty()) {
        PT_INFO("删点: 无坏体素, 点数保持 %zu", cloud->size());
        return;
    }

    const uint32_t n = static_cast<uint32_t>(cloud->points.size());
    std::vector<char> drop(n, 0);
    size_t marked = 0;
    for (const Voxel* v : bad_voxels) {
        if (!v) {
            continue;
        }
        for (uint32_t idx : v->idx) {
            if (idx < n && drop[idx] == 0) {
                drop[idx] = 1;
                ++marked;
            }
        }
    }

    pcl::PointCloud<pcl::PointXYZ>::Ptr removed;
    if (debug_mode_ && removed_publisher_) {
        removed = memory_pool_->acquire();
        removed->height = 1;
        removed->is_dense = true;
        removed->points.reserve(marked);
    }

    size_t write = 0;
    for (uint32_t i = 0; i < n; ++i) {
        if (drop[i] != 0) {
            if (removed) {
                removed->points.push_back(cloud->points[i]);
            }
            continue;
        }
        if (write != static_cast<size_t>(i)) {
            cloud->points[write] = cloud->points[i];
        }
        ++write;
    }
    cloud->points.resize(write);
    cloud->width = static_cast<uint32_t>(write);
    cloud->height = 1;
    cloud->is_dense = true;

    PT_INFO("删点: 坏格 %zu, 去掉 %zu 点, %u → %zu",
            bad_voxels.size(), marked, n, cloud->size());

    if (removed) {
        removed->width = static_cast<uint32_t>(removed->points.size());
        publish_cloud(removed, header, removed_publisher_, "被删点");
        memory_pool_->release(removed);
    }
}

namespace {

uint32_t packRgb(uint8_t r, uint8_t g, uint8_t b)
{
    return (static_cast<uint32_t>(r) << 16) |
           (static_cast<uint32_t>(g) << 8) |
           static_cast<uint32_t>(b);
}

void appendBoxEdges(
    visualization_msgs::msg::Marker& m,
    double xmin, double xmax,
    double ymin, double ymax,
    double zmin, double zmax)
{
    const double xs[2] = {xmin, xmax};
    const double ys[2] = {ymin, ymax};
    const double zs[2] = {zmin, zmax};
    const int corners[8][3] = {
        {0, 0, 0}, {1, 0, 0}, {1, 1, 0}, {0, 1, 0},
        {0, 0, 1}, {1, 0, 1}, {1, 1, 1}, {0, 1, 1},
    };
    const int edges[12][2] = {
        {0, 1}, {1, 2}, {2, 3}, {3, 0},
        {4, 5}, {5, 6}, {6, 7}, {7, 4},
        {0, 4}, {1, 5}, {2, 6}, {3, 7},
    };
    for (const auto& e : edges) {
        for (int k = 0; k < 2; ++k) {
            const int c = e[k];
            geometry_msgs::msg::Point p;
            p.x = xs[corners[c][0]];
            p.y = ys[corners[c][1]];
            p.z = zs[corners[c][2]];
            m.points.push_back(p);
        }
    }
}

}  // namespace

void CloudPassthroughFilterNode::publishFilterDebug(
    const pcl::PointCloud<pcl::PointXYZ>::Ptr& raw_cloud,
    const pcl::PointCloud<pcl::PointXYZ>::Ptr& pass_cloud,
    const std::vector<Voxel*>& bad_voxels,
    const std_msgs::msg::Header& header)
{
    std::unordered_set<const Voxel*> bad_set;
    bad_set.reserve(bad_voxels.size() * 2 + 1);
    for (const Voxel* v : bad_voxels) {
        if (v) {
            bad_set.insert(v);
        }
    }

    const uint32_t n_pass = (pass_cloud && !pass_cloud->empty())
        ? static_cast<uint32_t>(pass_cloud->points.size()) : 0;
    std::unordered_set<uint64_t> deleted_keys;
    deleted_keys.reserve(256);
    auto pack_key = [](float x, float y, float z) -> uint64_t {
        const int64_t ix = static_cast<int64_t>(std::llround(static_cast<double>(x) * 1000.0));
        const int64_t iy = static_cast<int64_t>(std::llround(static_cast<double>(y) * 1000.0));
        const int64_t iz = static_cast<int64_t>(std::llround(static_cast<double>(z) * 1000.0));
        return (static_cast<uint64_t>(ix + 200000) << 42) |
               (static_cast<uint64_t>(iy + 200000) << 21) |
               static_cast<uint64_t>(iz + 200000);
    };
    if (n_pass > 0) {
        std::vector<char> drop(n_pass, 0);
        for (const Voxel* v : bad_voxels) {
            if (!v) {
                continue;
            }
            for (uint32_t idx : v->idx) {
                if (idx < n_pass) {
                    drop[idx] = 1;
                }
            }
        }
        for (uint32_t i = 0; i < n_pass; ++i) {
            if (drop[i] != 0) {
                const auto& p = pass_cloud->points[i];
                deleted_keys.insert(pack_key(p.x, p.y, p.z));
            }
        }
    }

    // 1) 三种颜色：红=删，绿=留下，蓝=立方体内但没进直通
    size_t n_del = 0;
    size_t n_keep = 0;
    size_t n_blue = 0;
    if (verdict_publisher_ && raw_cloud && !raw_cloud->empty()) {
        const uint32_t n_raw = static_cast<uint32_t>(raw_cloud->points.size());
        std::vector<uint32_t> pick;
        std::vector<uint32_t> color;
        pick.reserve(n_raw);
        color.reserve(n_raw);
        const uint32_t c_del = packRgb(240, 40, 40);
        const uint32_t c_keep = packRgb(40, 220, 80);
        const uint32_t c_blue = packRgb(40, 110, 255);

        for (uint32_t i = 0; i < n_raw; ++i) {
            const auto& p = raw_cloud->points[i];
            const bool cube = inOurCube(p.x, p.y, p.z);
            const bool pass = inPassthrough(p.x, p.y, p.z);
            if (cube && !pass) {
                pick.push_back(i);
                color.push_back(c_blue);
                ++n_blue;
            } else if (cube && pass) {
                if (deleted_keys.count(pack_key(p.x, p.y, p.z)) != 0) {
                    pick.push_back(i);
                    color.push_back(c_del);
                    ++n_del;
                } else {
                    pick.push_back(i);
                    color.push_back(c_keep);
                    ++n_keep;
                }
            }
        }

        const uint32_t n_out = static_cast<uint32_t>(pick.size());
        sensor_msgs::msg::PointCloud2 msg;
        msg.header = header;
        msg.height = 1;
        msg.width = n_out;
        msg.is_bigendian = false;
        msg.is_dense = true;
        msg.fields.resize(4);
        msg.fields[0].name = "x";
        msg.fields[0].offset = 0;
        msg.fields[0].datatype = sensor_msgs::msg::PointField::FLOAT32;
        msg.fields[0].count = 1;
        msg.fields[1].name = "y";
        msg.fields[1].offset = 4;
        msg.fields[1].datatype = sensor_msgs::msg::PointField::FLOAT32;
        msg.fields[1].count = 1;
        msg.fields[2].name = "z";
        msg.fields[2].offset = 8;
        msg.fields[2].datatype = sensor_msgs::msg::PointField::FLOAT32;
        msg.fields[2].count = 1;
        msg.fields[3].name = "rgb";
        msg.fields[3].offset = 12;
        msg.fields[3].datatype = sensor_msgs::msg::PointField::FLOAT32;
        msg.fields[3].count = 1;
        msg.point_step = 16;
        msg.row_step = msg.point_step * n_out;
        msg.data.resize(static_cast<size_t>(msg.row_step));
        for (uint32_t k = 0; k < n_out; ++k) {
            const auto& p = raw_cloud->points[pick[k]];
            uint8_t* dst = msg.data.data() + static_cast<size_t>(k) * msg.point_step;
            float xyz[3] = {p.x, p.y, p.z};
            std::memcpy(dst, xyz, 12);
            std::memcpy(dst + 12, &color[k], 4);
        }
        verdict_publisher_->publish(msg);
    }

    // 2) 范围线框：青=立方体，橙=直通；头顶一行统计
    if (boxes_publisher_) {
        visualization_msgs::msg::MarkerArray arr;
        visualization_msgs::msg::Marker clear;
        clear.header = header;
        clear.ns = "filter_boxes";
        clear.id = 0;
        clear.action = visualization_msgs::msg::Marker::DELETEALL;
        arr.markers.push_back(clear);

        visualization_msgs::msg::Marker cube;
        cube.header = header;
        cube.ns = "filter_boxes";
        cube.id = 1;
        cube.type = visualization_msgs::msg::Marker::LINE_LIST;
        cube.action = visualization_msgs::msg::Marker::ADD;
        cube.pose.orientation.w = 1.0;
        cube.scale.x = 0.02;
        cube.color.r = 0.0f;
        cube.color.g = 0.85f;
        cube.color.b = 0.85f;
        cube.color.a = 0.95f;
        appendBoxEdges(cube,
                       -0.5 * cube_length_, 0.5 * cube_length_,
                       -0.5 * cube_width_, 0.5 * cube_width_,
                       -0.5 * cube_height_, 0.5 * cube_height_);
        arr.markers.push_back(cube);

        visualization_msgs::msg::Marker pass;
        pass.header = header;
        pass.ns = "filter_boxes";
        pass.id = 2;
        pass.type = visualization_msgs::msg::Marker::LINE_LIST;
        pass.action = visualization_msgs::msg::Marker::ADD;
        pass.pose.orientation.w = 1.0;
        pass.scale.x = 0.015;
        pass.color.r = 1.0f;
        pass.color.g = 0.55f;
        pass.color.b = 0.0f;
        pass.color.a = 0.9f;
        appendBoxEdges(pass,
                       axes_[0].limit_min, axes_[0].limit_max,
                       axes_[1].limit_min, axes_[1].limit_max,
                       axes_[2].limit_min, axes_[2].limit_max);
        arr.markers.push_back(pass);

        visualization_msgs::msg::Marker text;
        text.header = header;
        text.ns = "filter_boxes";
        text.id = 3;
        text.type = visualization_msgs::msg::Marker::TEXT_VIEW_FACING;
        text.action = visualization_msgs::msg::Marker::ADD;
        text.pose.orientation.w = 1.0;
        text.pose.position.x = 0.0;
        text.pose.position.y = 0.0;
        text.pose.position.z = 0.5 * cube_height_ + 0.35;
        text.scale.z = 0.12;
        text.color.r = 1.0f;
        text.color.g = 1.0f;
        text.color.b = 1.0f;
        text.color.a = 1.0f;
        char buf[192];
        std::snprintf(buf, sizeof(buf),
                      "del(红)=%zu  keep(绿)=%zu  cube-only(蓝)=%zu  voxels=%zu",
                      n_del, n_keep, n_blue, grid_.size());
        text.text = buf;
        arr.markers.push_back(text);
        boxes_publisher_->publish(arr);
    }

    // 3) 有点的合成体素：半透明红=删格 / 绿=留格；格内画穿过的雷达线
    if (voxels_publisher_ && publish_occupied_voxels_) {
        visualization_msgs::msg::MarkerArray arr;
        visualization_msgs::msg::Marker clear;
        clear.header = header;
        clear.ns = "occupied_voxels";
        clear.id = 0;
        clear.action = visualization_msgs::msg::Marker::DELETEALL;
        arr.markers.push_back(clear);

        constexpr size_t kMaxVoxels = 2500;
        size_t drawn = 0;
        int cube_id = 10;
        int arrow_id = 5000;
        int text_id = 10000;

        for (const auto& entry : grid_) {
            if (drawn >= kMaxVoxels) {
                break;
            }
            const Voxel& v = entry.second;
            const double sx = std::max(static_cast<double>(v.size_xy), base_xy_);
            const double sz = std::max(static_cast<double>(v.size_z), base_z_);
            const double xmin = static_cast<double>(v.ix) * sx;
            const double ymin = static_cast<double>(v.iy) * sx;
            const double zmin = static_cast<double>(v.iz) * sz;

            const double cx = xmin + 0.5 * sx;
            const double cy = ymin + 0.5 * sx;
            const double cz = zmin + 0.5 * sz;
            const bool is_bad = bad_set.count(&v) != 0;

            visualization_msgs::msg::Marker cube;
            cube.header = header;
            cube.ns = "occupied_voxels";
            cube.id = cube_id++;
            cube.type = visualization_msgs::msg::Marker::CUBE;
            cube.action = visualization_msgs::msg::Marker::ADD;
            cube.pose.orientation.w = 1.0;
            cube.pose.position.x = cx;
            cube.pose.position.y = cy;
            cube.pose.position.z = cz;
            cube.scale.x = sx;
            cube.scale.y = sx;
            cube.scale.z = sz;
            if (is_bad) {
                cube.color.r = 0.95f;
                cube.color.g = 0.2f;
                cube.color.b = 0.2f;
                cube.color.a = 0.28f;
            } else {
                cube.color.r = 0.2f;
                cube.color.g = 0.85f;
                cube.color.b = 0.3f;
                cube.color.a = 0.28f;
            }
            arr.markers.push_back(cube);

            // 格内按线号分组：方位从小到大，细箭头表示穿过方向
            if (pass_cloud && !v.idx.empty()) {
                std::unordered_map<int, std::vector<const pcl::PointXYZ*>> by_ring;
                by_ring.reserve(8);
                for (uint32_t idx : v.idx) {
                    if (idx >= n_pass) {
                        continue;
                    }
                    const auto& p = pass_cloud->points[idx];
                    by_ring[ringOfPoint(idx, p.x, p.y, p.z)].push_back(&p);
                }
                for (auto& ring_entry : by_ring) {
                    auto& pts = ring_entry.second;
                    if (pts.size() < 2) {
                        continue;
                    }
                    std::sort(pts.begin(), pts.end(), [](const pcl::PointXYZ* a, const pcl::PointXYZ* b) {
                        return std::atan2(static_cast<double>(a->y), static_cast<double>(a->x)) <
                               std::atan2(static_cast<double>(b->y), static_cast<double>(b->x));
                    });
                    const pcl::PointXYZ* first = pts.front();
                    const pcl::PointXYZ* last = pts.back();
                    visualization_msgs::msg::Marker arrow;
                    arrow.header = header;
                    arrow.ns = "occupied_voxels";
                    arrow.id = arrow_id++;
                    arrow.type = visualization_msgs::msg::Marker::ARROW;
                    arrow.action = visualization_msgs::msg::Marker::ADD;
                    arrow.pose.orientation.w = 1.0;
                    arrow.color.r = 0.75f;
                    arrow.color.g = 0.55f;
                    arrow.color.b = 1.0f;
                    arrow.color.a = 0.95f;
                    // 两点式：杆是细线，箭头只在末尾
                    arrow.scale.x = 0.0015;
                    arrow.scale.y = 0.004;
                    arrow.scale.z = 0.010;
                    geometry_msgs::msg::Point p0;
                    p0.x = first->x;
                    p0.y = first->y;
                    p0.z = first->z;
                    geometry_msgs::msg::Point p1;
                    p1.x = last->x;
                    p1.y = last->y;
                    p1.z = last->z;
                    arrow.points.push_back(p0);
                    arrow.points.push_back(p1);
                    arr.markers.push_back(arrow);
                }
            }

            visualization_msgs::msg::Marker text;
            text.header = header;
            text.ns = "occupied_voxels";
            text.id = text_id++;
            text.type = visualization_msgs::msg::Marker::TEXT_VIEW_FACING;
            text.action = visualization_msgs::msg::Marker::ADD;
            text.pose.orientation.w = 1.0;
            text.pose.position.x = cx;
            text.pose.position.y = cy;
            text.pose.position.z = zmin + sz + 0.03;
            text.scale.z = 0.018;
            text.color.r = 1.0f;
            text.color.g = 1.0f;
            text.color.b = 0.85f;
            text.color.a = 1.0f;
            char buf[80];
            const float zspan = v.zmax - v.zmin;
            if (v.cluster_id > 0) {
                std::snprintf(buf, sizeof(buf), "%.2f,%.2f,%.2f,%.2f,t%d",
                              cx, cy, cz, zspan, v.cluster_id);
            } else {
                std::snprintf(buf, sizeof(buf), "%.2f,%.2f,%.2f,%.2f",
                              cx, cy, cz, zspan);
            }
            text.text = buf;
            arr.markers.push_back(text);
            ++drawn;
        }
        voxels_publisher_->publish(arr);
        PT_INFO("调试可视化: 红删 %zu 绿留 %zu 蓝(立方体无直通) %zu 有点体素 %zu/%zu",
                n_del, n_keep, n_blue, drawn, grid_.size());
    }
}

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
