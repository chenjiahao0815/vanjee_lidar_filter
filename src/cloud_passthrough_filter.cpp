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
#include <string>
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
        if (publish_occupied_voxels_) {
            if (!voxels_topic_.empty()) {
                voxels_publisher_ = this->create_publisher<visualization_msgs::msg::MarkerArray>(
                    voxels_topic_, viz_qos);
            }
            if (!voxels_deleted_topic_.empty()) {
                voxels_deleted_publisher_ =
                    this->create_publisher<visualization_msgs::msg::MarkerArray>(
                        voxels_deleted_topic_, viz_qos);
            }
            if (!voxels_kept_topic_.empty()) {
                voxels_kept_publisher_ =
                    this->create_publisher<visualization_msgs::msg::MarkerArray>(
                        voxels_kept_topic_, viz_qos);
            }
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
    this->declare_parameter("cube_min_x", -2.0);
    this->declare_parameter("cube_max_x", 2.0);
    this->declare_parameter("cube_min_y", -2.0);
    this->declare_parameter("cube_max_y", 2.0);
    this->declare_parameter("cube_min_z", -1.0);
    this->declare_parameter("cube_max_z", 1.0);
    this->declare_parameter("ang_h", 0.0034906585);   // 约 0.2 deg
    this->declare_parameter("ang_v", 0.0174532925);   // 约 1.0 deg
    this->declare_parameter("r_max", 8.0);
    this->declare_parameter("base_x", 0.01);
    this->declare_parameter("base_y", 0.01);
    this->declare_parameter("base_z", 0.01);
    this->declare_parameter("thr_ratio", std::string("(0.0,2.0)"));
    this->declare_parameter("thr_z_min", 0.03);
    this->declare_parameter("enable_merge_z_check", true);
    this->declare_parameter("enable_small_cluster_filter", true);
    this->declare_parameter("small_cluster_max_points", 10);
    this->declare_parameter("small_cluster_link_m", 0.15);
    this->declare_parameter("verdict_topic", std::string("/vanjee/filter_verdict"));
    this->declare_parameter("boxes_topic", std::string("/vanjee/filter_boxes"));
    this->declare_parameter("voxels_topic", std::string("/vanjee/voxel_markers"));
    this->declare_parameter("voxels_deleted_topic", std::string("/vanjee/voxel_markers_deleted"));
    this->declare_parameter("voxels_kept_topic", std::string("/vanjee/voxel_markers_kept"));
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
    cube_min_x_ = this->get_parameter("cube_min_x").as_double();
    cube_max_x_ = this->get_parameter("cube_max_x").as_double();
    cube_min_y_ = this->get_parameter("cube_min_y").as_double();
    cube_max_y_ = this->get_parameter("cube_max_y").as_double();
    cube_min_z_ = this->get_parameter("cube_min_z").as_double();
    cube_max_z_ = this->get_parameter("cube_max_z").as_double();
    if (cube_min_x_ > cube_max_x_) {
        std::swap(cube_min_x_, cube_max_x_);
    }
    if (cube_min_y_ > cube_max_y_) {
        std::swap(cube_min_y_, cube_max_y_);
    }
    if (cube_min_z_ > cube_max_z_) {
        std::swap(cube_min_z_, cube_max_z_);
    }
    ang_h_ = this->get_parameter("ang_h").as_double();
    ang_v_ = this->get_parameter("ang_v").as_double();
    r_max_ = this->get_parameter("r_max").as_double();
    base_x_ = this->get_parameter("base_x").as_double();
    base_y_ = this->get_parameter("base_y").as_double();
    base_z_ = this->get_parameter("base_z").as_double();
    thr_ratio_raw_ = this->get_parameter("thr_ratio").as_string();
    if (!parseSizeXyBands(thr_ratio_raw_, thr_ratio_bands_)) {
        RCLCPP_WARN(this->get_logger(),
                    "thr_ratio 解析失败: '%s'，退回全程倍率 2.0",
                    thr_ratio_raw_.c_str());
        thr_ratio_bands_.clear();
        thr_ratio_bands_.push_back(SizeXyBand{0.0, 2.0});
    }
    thr_z_min_ = this->get_parameter("thr_z_min").as_double();
    if (thr_z_min_ < 0.0) {
        thr_z_min_ = 0.0;
    }
    enable_merge_z_check_ = this->get_parameter("enable_merge_z_check").as_bool();
    enable_small_cluster_filter_ = this->get_parameter("enable_small_cluster_filter").as_bool();
    small_cluster_max_points_ = this->get_parameter("small_cluster_max_points").as_int();
    if (small_cluster_max_points_ < 1) {
        small_cluster_max_points_ = 10;
    }
    small_cluster_link_m_ = this->get_parameter("small_cluster_link_m").as_double();
    if (small_cluster_link_m_ <= 0.0) {
        small_cluster_link_m_ = 0.15;
    }
    verdict_topic_ = this->get_parameter("verdict_topic").as_string();
    boxes_topic_ = this->get_parameter("boxes_topic").as_string();
    voxels_topic_ = this->get_parameter("voxels_topic").as_string();
    voxels_deleted_topic_ = this->get_parameter("voxels_deleted_topic").as_string();
    voxels_kept_topic_ = this->get_parameter("voxels_kept_topic").as_string();
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
        RCLCPP_INFO(this->get_logger(), "  删格体素: %s", voxels_deleted_topic_.c_str());
        RCLCPP_INFO(this->get_logger(), "  留格体素: %s", voxels_kept_topic_.c_str());
        }
        RCLCPP_INFO(this->get_logger(),
                    "  帧日志间隔: %.2f s (仅 debug_mode=false 时限频)",
                    cloud_log_interval_sec_);
    RCLCPP_INFO(this->get_logger(), "  体素删点: %s", enable_voxel_filter_ ? "开" : "关");
    if (enable_voxel_filter_) {
        RCLCPP_INFO(this->get_logger(),
                    "  雷达系立方体: x[%.2f,%.2f] y[%.2f,%.2f] z[%.2f,%.2f]",
                    cube_min_x_, cube_max_x_,
                    cube_min_y_, cube_max_y_,
                    cube_min_z_, cube_max_z_);
        RCLCPP_INFO(this->get_logger(),
                    "  角分辨率 ang_h=%.6f ang_v=%.6f, 细格固定 "
                    "base_x=%.3f base_y=%.3f base_z=%.3f, "
                    "thr_ratio=%s thr_z_min=%.3f",
                    ang_h_, ang_v_, base_x_, base_y_, base_z_,
                    thr_ratio_raw_.c_str(), thr_z_min_);
        RCLCPP_INFO(this->get_logger(),
                    "  删点: 固定细格，26邻接%s并大格，再按厚度删",
                    enable_merge_z_check_ ? "且两端同柱另有z格才" : "直接");
        RCLCPP_INFO(this->get_logger(),
                    "  并边z检查: %s",
                    enable_merge_z_check_ ? "开" : "关");
        RCLCPP_INFO(this->get_logger(), "  --- 门槛样例 ---");
        logVoxelScaleSamples();
    }
    RCLCPP_INFO(this->get_logger(),
                "  小团清扫: %s  团点数≤%d  link=%.3f m",
                enable_small_cluster_filter_ ? "开" : "关",
                small_cluster_max_points_, small_cluster_link_m_);
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
    std::vector<char> drop;
    const size_t n_before_voxel = (current && !current->empty()) ? current->size() : 0;
    if (enable_voxel_filter_ && current && !current->empty()) {
        grid_.clear();
        fine_grid_.clear();
        cells_.clear();
        key_to_idx_.clear();
        nbr_table_.clear();
        nbrs_.clear();
        buildVoxelGrid(current);
        bad_voxels = collectBadVoxels();
        drop.assign(current->points.size(), 0);
        markDropFromBadVoxels(current, bad_voxels, drop);
    }
    if (enable_small_cluster_filter_ && current && !current->empty()) {
        if (drop.size() != current->points.size()) {
            drop.assign(current->points.size(), 0);
        }
        markSmallPointClusters(current, drop);
    }
    if (debug_mode_) {
        publishFilterDebug(raw_viz, current, bad_voxels, msg->header,
                           drop.empty() ? nullptr : &drop);
    }
    if (!drop.empty()) {
        removePointsByMask(current, drop, msg->header);
        PT_INFO("体素删点结束: 坏格 %zu, 点数 %zu → %zu",
                bad_voxels.size(), n_before_voxel, current->size());
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
                    "  样例 r=%.1f thr_ratio=%.2f "
                    "line_gap=%.4f size_x=%.3f size_y=%.3f size_z=%.3f thr_z=%.4f",
                    r, lookupThrRatio(r),
                    s.line_gap, base_x_, base_y_, s.size_z, s.thr_z);
    }
}

VoxelScale CloudPassthroughFilterNode::computeVoxelScale(double r) const
{
    VoxelScale s;
    const double rr = std::max(r, 1e-3);
    const double base_x = std::max(base_x_, 1e-6);
    const double base_y = std::max(base_y_, 1e-6);
    const double base_z = std::max(base_z_, 1e-6);

    s.line_gap = rr * ang_v_;
    s.pt_gap = s.line_gap;

    s.size_xy = std::max(base_x, base_y);
    s.size_z = base_z;
    s.mult_xy = 1;
    s.mult_z = 1;

    s.thr_z = s.line_gap * lookupThrRatio(rr);
    if (s.thr_z < thr_z_min_) {
        s.thr_z = thr_z_min_;
    }

    s.inv_xy = 1.0 / std::max(s.size_xy, 1e-6);
    s.inv_z = 1.0 / std::max(s.size_z, 1e-6);
    return s;
}

bool CloudPassthroughFilterNode::parseSizeXyBands(
    const std::string& text,
    std::vector<SizeXyBand>& out) const
{
    out.clear();
    size_t i = 0;
    const size_t n = text.size();
    auto skipWs = [&]() {
        while (i < n && std::isspace(static_cast<unsigned char>(text[i]))) {
            ++i;
        }
    };
    auto readNumber = [&](double& v) -> bool {
        skipWs();
        if (i >= n) {
            return false;
        }
        const size_t begin = i;
        if (text[i] == '+' || text[i] == '-') {
            ++i;
        }
        bool any_digit = false;
        while (i < n && std::isdigit(static_cast<unsigned char>(text[i]))) {
            any_digit = true;
            ++i;
        }
        if (i < n && text[i] == '.') {
            ++i;
            while (i < n && std::isdigit(static_cast<unsigned char>(text[i]))) {
                any_digit = true;
                ++i;
            }
        }
        if (!any_digit) {
            return false;
        }
        try {
            v = std::stod(text.substr(begin, i - begin));
        } catch (...) {
            return false;
        }
        return true;
    };

    skipWs();
    if (i >= n) {
        return false;
    }

    while (i < n) {
        skipWs();
        if (i >= n) {
            break;
        }
        if (text[i] != '(') {
            return false;
        }
        ++i;
        double split_r = 0.0;
        double ratio = 0.0;
        if (!readNumber(split_r)) {
            return false;
        }
        skipWs();
        if (i >= n || text[i] != ',') {
            return false;
        }
        ++i;
        if (!readNumber(ratio)) {
            return false;
        }
        skipWs();
        if (i >= n || text[i] != ')') {
            return false;
        }
        ++i;
        if (!(split_r >= 0.0) || !(ratio > 0.0)) {
            return false;
        }
        out.push_back(SizeXyBand{split_r, ratio});
        skipWs();
        if (i < n && text[i] == ',') {
            ++i;
            continue;
        }
        if (i < n) {
            return false;
        }
    }

    if (out.empty()) {
        return false;
    }
    std::sort(out.begin(), out.end(),
              [](const SizeXyBand& a, const SizeXyBand& b) {
                  return a.split_r < b.split_r;
              });
    for (size_t k = 1; k < out.size(); ++k) {
        if (!(out[k].split_r > out[k - 1].split_r)) {
            return false;
        }
    }
    return true;
}

double CloudPassthroughFilterNode::lookupThrRatio(double r_xy) const
{
    if (thr_ratio_bands_.empty()) {
        return 2.0;
    }
    const double r = std::max(r_xy, 0.0);
    double ratio = 1.0;
    for (const auto& band : thr_ratio_bands_) {
        if (r >= band.split_r) {
            ratio = band.ratio;
        } else {
            break;
        }
    }
    return ratio;
}

bool CloudPassthroughFilterNode::inOurCube(float x, float y, float z) const
{
    // 与直通相同：按轴 min/max，原点两侧可不对称
    return static_cast<double>(x) >= cube_min_x_ &&
           static_cast<double>(x) <= cube_max_x_ &&
           static_cast<double>(y) >= cube_min_y_ &&
           static_cast<double>(y) <= cube_max_y_ &&
           static_cast<double>(z) >= cube_min_z_ &&
           static_cast<double>(z) <= cube_max_z_;
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
    const double inv_x = 1.0 / std::max(base_x_, 1e-6);
    const double inv_y = 1.0 / std::max(base_y_, 1e-6);
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

        const int ix = static_cast<int>(std::floor(static_cast<double>(p.x) * inv_x));
        const int iy = static_cast<int>(std::floor(static_cast<double>(p.y) * inv_y));
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
            v.cx = static_cast<float>((static_cast<double>(ix) + 0.5) * base_x_);
            v.cy = static_cast<float>((static_cast<double>(iy) + 0.5) * base_y_);
            v.idx.push_back(i);
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

    flattenOccupiedVoxels();
    buildNeighborTable();
    mergeFineVoxels();
    int cell_n = 0;
    for (auto& entry : grid_) {
        entry.second.cell_id = ++cell_n;
    }
    PT_INFO("体素装格: 框内 %u 点, 框外跳过 %u, 有点细格 %zu, 邻接边 %zu, 并后大格 %zu",
            loaded, skipped_outside, cells_.size(), nbr_table_.size(), grid_.size());
}

void CloudPassthroughFilterNode::flattenOccupiedVoxels()
{
    cells_.clear();
    key_to_idx_.clear();
    cells_.reserve(fine_grid_.size());
    key_to_idx_.reserve(fine_grid_.size() * 2 + 1);
    for (auto& entry : fine_grid_) {
        Voxel v = std::move(entry.second);
        const double cx = (static_cast<double>(v.ix) + 0.5) * base_x_;
        const double cy = (static_cast<double>(v.iy) + 0.5) * base_y_;
        const float cell_r = static_cast<float>(std::hypot(cx, cy));
        const VoxelScale s = computeVoxelScale(static_cast<double>(cell_r));
        v.cx = static_cast<float>(cx);
        v.cy = static_cast<float>(cy);
        v.r = cell_r;
        v.cell_sx = static_cast<float>(std::max(base_x_, 1e-6));
        v.cell_sy = static_cast<float>(std::max(base_y_, 1e-6));
        v.cell_sz = static_cast<float>(std::max(base_z_, 1e-6));
        v.merge_nx = 1;
        v.merge_ny = 1;
        v.merge_nz = 1;
        v.thr_z = static_cast<float>(s.thr_z);
        v.size_xy = static_cast<float>(s.size_xy);
        v.size_z = static_cast<float>(s.size_z);
        v.cluster_id = 0;
        const int idx = static_cast<int>(cells_.size());
        key_to_idx_.emplace(entry.first, idx);
        cells_.push_back(std::move(v));
    }
    fine_grid_.clear();
}

void CloudPassthroughFilterNode::buildNeighborTable()
{
    nbr_table_.clear();
    nbrs_.clear();
    const int n = static_cast<int>(cells_.size());
    nbrs_.assign(static_cast<size_t>(n), {});
    if (n == 0) {
        return;
    }

    nbr_table_.reserve(static_cast<size_t>(n) * 13);
    for (int i = 0; i < n; ++i) {
        const Voxel& v = cells_[static_cast<size_t>(i)];
        for (int dx = -1; dx <= 1; ++dx) {
            for (int dy = -1; dy <= 1; ++dy) {
                for (int dz = -1; dz <= 1; ++dz) {
                    if (dx == 0 && dy == 0 && dz == 0) {
                        continue;
                    }
                    const int64_t nk = makeKey(v.ix + dx, v.iy + dy, v.iz + dz);
                    const auto nit = key_to_idx_.find(nk);
                    if (nit == key_to_idx_.end()) {
                        continue;
                    }
                    const int j = nit->second;
                    if (j <= i) {
                        continue;
                    }
                    nbr_table_.push_back(NbrEdge{i, j});
                }
            }
        }
    }
    for (const NbrEdge& e : nbr_table_) {
        nbrs_[static_cast<size_t>(e.a)].push_back(e.b);
        nbrs_[static_cast<size_t>(e.b)].push_back(e.a);
    }
}

void CloudPassthroughFilterNode::mergeFineVoxels()
{
    grid_.clear();
    const int n = static_cast<int>(cells_.size());
    if (n == 0) {
        return;
    }

    std::vector<int> parent(static_cast<size_t>(n));
    for (int i = 0; i < n; ++i) {
        parent[static_cast<size_t>(i)] = i;
    }
    auto find = [&](int x) {
        while (parent[static_cast<size_t>(x)] != x) {
            parent[static_cast<size_t>(x)] =
                parent[static_cast<size_t>(parent[static_cast<size_t>(x)])];
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
    size_t n_merge_edge = 0;
    if (enable_merge_z_check_) {
        auto col_key = [](int ix, int iy) -> int64_t {
            constexpr int64_t kOff = 1 << 20;
            return ((static_cast<int64_t>(ix) + kOff) << 32) |
                   (static_cast<int64_t>(iy) + kOff);
        };
        std::unordered_map<int64_t, int> col_n;
        col_n.reserve(static_cast<size_t>(n) * 2 + 1);
        for (int i = 0; i < n; ++i) {
            const Voxel& v = cells_[static_cast<size_t>(i)];
            ++col_n[col_key(v.ix, v.iy)];
        }
        std::vector<char> z_ok(static_cast<size_t>(n), 0);
        for (int i = 0; i < n; ++i) {
            const Voxel& v = cells_[static_cast<size_t>(i)];
            z_ok[static_cast<size_t>(i)] =
                col_n[col_key(v.ix, v.iy)] >= 2 ? 1 : 0;
        }
        for (const NbrEdge& e : nbr_table_) {
            if (z_ok[static_cast<size_t>(e.a)] == 0 ||
                z_ok[static_cast<size_t>(e.b)] == 0) {
                continue;
            }
            unite(e.a, e.b);
            ++n_merge_edge;
        }
    } else {
        for (const NbrEdge& e : nbr_table_) {
            unite(e.a, e.b);
            ++n_merge_edge;
        }
    }

    std::unordered_map<int, std::vector<int>> comps;
    comps.reserve(static_cast<size_t>(n));
    for (int i = 0; i < n; ++i) {
        comps[find(i)].push_back(i);
    }

    int cluster_n = 0;
    for (auto& ce : comps) {
        ++cluster_n;
        const std::vector<int>& members = ce.second;
        if (members.empty()) {
            continue;
        }

        const Voxel& seed = cells_[static_cast<size_t>(members[0])];
        int min_ix = seed.ix, max_ix = seed.ix;
        int min_iy = seed.iy, max_iy = seed.iy;
        int min_iz = seed.iz, max_iz = seed.iz;
        float zmin = seed.zmin;
        float zmax = seed.zmax;
        uint32_t count = 0;
        uint64_t ring_mask = 0;
        std::vector<uint32_t> idx;
        idx.reserve(members.size() * 4);

        for (int mi : members) {
            Voxel& fine = cells_[static_cast<size_t>(mi)];
            fine.cluster_id = cluster_n;
            if (fine.ix < min_ix) {
                min_ix = fine.ix;
            }
            if (fine.ix > max_ix) {
                max_ix = fine.ix;
            }
            if (fine.iy < min_iy) {
                min_iy = fine.iy;
            }
            if (fine.iy > max_iy) {
                max_iy = fine.iy;
            }
            if (fine.iz < min_iz) {
                min_iz = fine.iz;
            }
            if (fine.iz > max_iz) {
                max_iz = fine.iz;
            }
            if (fine.zmin < zmin) {
                zmin = fine.zmin;
            }
            if (fine.zmax > zmax) {
                zmax = fine.zmax;
            }
            count += fine.count;
            ring_mask |= fine.ring_mask;
            idx.insert(idx.end(), fine.idx.begin(), fine.idx.end());
        }

        const int nx = std::max(1, max_ix - min_ix + 1);
        const int ny = std::max(1, max_iy - min_iy + 1);
        const int nz = std::max(1, max_iz - min_iz + 1);
        const double cell_sx_d = base_x_ * static_cast<double>(nx);
        const double cell_sy_d = base_y_ * static_cast<double>(ny);
        const double cell_sz_d = base_z_ * static_cast<double>(nz);
        const float cell_sx = static_cast<float>(std::max(cell_sx_d, base_x_));
        const float cell_sy = static_cast<float>(std::max(cell_sy_d, base_y_));
        const float cell_sz = static_cast<float>(std::max(cell_sz_d, base_z_));
        const double cube_cx =
            static_cast<double>(min_ix) * base_x_ + 0.5 * cell_sx_d;
        const double cube_cy =
            static_cast<double>(min_iy) * base_y_ + 0.5 * cell_sy_d;
        const float cell_r = static_cast<float>(std::hypot(cube_cx, cube_cy));
        const VoxelScale s = computeVoxelScale(static_cast<double>(cell_r));

        Voxel v;
        v.ix = min_ix;
        v.iy = min_iy;
        v.iz = min_iz;
        v.count = count;
        v.zmin = zmin;
        v.zmax = zmax;
        v.r = cell_r;
        v.cx = static_cast<float>(cube_cx);
        v.cy = static_cast<float>(cube_cy);
        v.cell_sx = cell_sx;
        v.cell_sy = cell_sy;
        v.cell_sz = cell_sz;
        v.merge_nx = nx;
        v.merge_ny = ny;
        v.merge_nz = nz;
        v.ring_mask = ring_mask;
        v.idx = std::move(idx);
        v.cluster_id = cluster_n;
        v.thr_z = static_cast<float>(s.thr_z);
        v.size_xy = static_cast<float>(s.size_xy);
        v.size_z = static_cast<float>(s.size_z);
        grid_.emplace(static_cast<int64_t>(cluster_n), std::move(v));
    }
    PT_INFO("邻接并格: 细格 %d, 邻边 %zu, %s并边 %zu, 大格 %zu",
            n, nbr_table_.size(),
            enable_merge_z_check_ ? "同柱z支撑" : "26邻接",
            n_merge_edge, grid_.size());
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

std::vector<Voxel*> CloudPassthroughFilterNode::collectBadVoxels()
{
    std::vector<Voxel*> all;
    all.reserve(grid_.size());
    for (auto& entry : grid_) {
        all.push_back(&entry.second);
    }

    std::vector<Voxel*> bad;
    bad.reserve(grid_.size());

    size_t thick_n = 0;
    float sample_span = 0.0f;
    float sample_thr = 0.0f;
    bool have_sample = false;

    for (Voxel* vp : all) {
        Voxel& v = *vp;
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

    PT_INFO("坏体素判定: 大格 %zu, 厚留 %zu, 坏 %zu ",
            grid_.size(), thick_n, bad.size());
    if (have_sample) {
        PT_INFO("坏体素样例: span=%.4f < thr_z=%.4f", sample_span, sample_thr);
    }
    return bad;
}

void CloudPassthroughFilterNode::markDropFromBadVoxels(
    const pcl::PointCloud<pcl::PointXYZ>::Ptr& cloud,
    const std::vector<Voxel*>& bad_voxels,
    std::vector<char>& drop)
{
    if (!cloud || drop.size() != cloud->points.size()) {
        PT_WARN("删点: drop 与点云长度不符, 跳过坏格标记");
        return;
    }
    if (bad_voxels.empty()) {
        PT_INFO("删点: 无坏体素, 点数保持 %zu", cloud->size());
        return;
    }

    const uint32_t n = static_cast<uint32_t>(cloud->points.size());
    for (const Voxel* v : bad_voxels) {
        if (!v) {
            continue;
        }
        const float zmid = 0.5f * (v->zmin + v->zmax);
        const float zspan = v->zmax - v->zmin;
        if (v->cluster_id > 0) {
            PT_INFO("删体素: #%d xyz=(%.2f,%.2f,%.2f) zspan=%.3f thr_z=%.3f t%d",
                    v->cell_id, v->cx, v->cy, zmid, zspan, v->thr_z, v->cluster_id);
        } else {
            PT_INFO("删体素: #%d xyz=(%.2f,%.2f,%.2f) zspan=%.3f thr_z=%.3f",
                    v->cell_id, v->cx, v->cy, zmid, zspan, v->thr_z);
        }
        for (uint32_t idx : v->idx) {
            if (idx < n) {
                drop[idx] = 1;
            }
        }
    }
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
    std::vector<char> drop(cloud->points.size(), 0);
    markDropFromBadVoxels(cloud, bad_voxels, drop);
    removePointsByMask(cloud, drop, header);
}

void CloudPassthroughFilterNode::markSmallPointClusters(
    const pcl::PointCloud<pcl::PointXYZ>::Ptr& cloud,
    std::vector<char>& drop)
{
    if (!cloud || cloud->empty()) {
        return;
    }
    const uint32_t n = static_cast<uint32_t>(cloud->points.size());
    if (drop.size() != static_cast<size_t>(n)) {
        PT_WARN("小团清扫: drop 长度与点数不符 (%zu vs %u), 跳过", drop.size(), n);
        return;
    }
    if (small_cluster_max_points_ < 1) {
        return;
    }

    const double link = std::max(small_cluster_link_m_, 1e-4);
    const double link2 = link * link;
    const double inv = 1.0 / link;

    std::vector<uint32_t> ids;
    ids.reserve(n);
    for (uint32_t i = 0; i < n; ++i) {
        if (drop[i] != 0) {
            continue;
        }
        const auto& p = cloud->points[i];
        if (!std::isfinite(p.x) || !std::isfinite(p.y) || !std::isfinite(p.z)) {
            continue;
        }
        if (!inOurCube(p.x, p.y, p.z)) {
            continue;
        }
        ids.push_back(i);
    }
    if (ids.empty()) {
        PT_INFO("小团清扫: 框内无候选点");
        return;
    }

    const int m = static_cast<int>(ids.size());
    std::vector<int> parent(static_cast<size_t>(m));
    std::vector<int> rank(static_cast<size_t>(m), 0);
    std::vector<int> sz(static_cast<size_t>(m), 1);
    for (int i = 0; i < m; ++i) {
        parent[static_cast<size_t>(i)] = i;
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
        if (a == b) {
            return;
        }
        if (rank[static_cast<size_t>(a)] < rank[static_cast<size_t>(b)]) {
            std::swap(a, b);
        }
        parent[static_cast<size_t>(b)] = a;
        sz[static_cast<size_t>(a)] += sz[static_cast<size_t>(b)];
        if (rank[static_cast<size_t>(a)] == rank[static_cast<size_t>(b)]) {
            ++rank[static_cast<size_t>(a)];
        }
    };

    std::vector<int> cx(static_cast<size_t>(m));
    std::vector<int> cy(static_cast<size_t>(m));
    std::vector<int> cz(static_cast<size_t>(m));
    std::unordered_map<int64_t, std::vector<int>> buckets;
    buckets.reserve(static_cast<size_t>(m) * 2 + 1);
    for (int k = 0; k < m; ++k) {
        const auto& p = cloud->points[ids[static_cast<size_t>(k)]];
        cx[static_cast<size_t>(k)] = static_cast<int>(std::floor(static_cast<double>(p.x) * inv));
        cy[static_cast<size_t>(k)] = static_cast<int>(std::floor(static_cast<double>(p.y) * inv));
        cz[static_cast<size_t>(k)] = static_cast<int>(std::floor(static_cast<double>(p.z) * inv));
        buckets[makeKey(cx[static_cast<size_t>(k)], cy[static_cast<size_t>(k)], cz[static_cast<size_t>(k)])]
            .push_back(k);
    }

    for (int k = 0; k < m; ++k) {
        const auto& p = cloud->points[ids[static_cast<size_t>(k)]];
        for (int dx = -1; dx <= 1; ++dx) {
            for (int dy = -1; dy <= 1; ++dy) {
                for (int dz = -1; dz <= 1; ++dz) {
                    const auto it = buckets.find(makeKey(
                        cx[static_cast<size_t>(k)] + dx,
                        cy[static_cast<size_t>(k)] + dy,
                        cz[static_cast<size_t>(k)] + dz));
                    if (it == buckets.end()) {
                        continue;
                    }
                    for (int j : it->second) {
                        if (j <= k) {
                            continue;
                        }
                        const auto& q = cloud->points[ids[static_cast<size_t>(j)]];
                        const double ddx = static_cast<double>(p.x) - static_cast<double>(q.x);
                        const double ddy = static_cast<double>(p.y) - static_cast<double>(q.y);
                        const double ddz = static_cast<double>(p.z) - static_cast<double>(q.z);
                        if (ddx * ddx + ddy * ddy + ddz * ddz <= link2) {
                            unite(k, j);
                        }
                    }
                }
            }
        }
    }

    size_t n_small_clusters = 0;
    size_t n_drop_pts = 0;
    std::vector<char> counted(static_cast<size_t>(m), 0);
    for (int k = 0; k < m; ++k) {
        const int root = find(k);
        if (sz[static_cast<size_t>(root)] > small_cluster_max_points_) {
            continue;
        }
        drop[ids[static_cast<size_t>(k)]] = 1;
        ++n_drop_pts;
        if (counted[static_cast<size_t>(root)] == 0) {
            counted[static_cast<size_t>(root)] = 1;
            ++n_small_clusters;
        }
    }
    PT_INFO("小团清扫: 框内候选 %d, 小团 %zu, 删 %zu 点 (团点数≤%d, link=%.3f)",
            m, n_small_clusters, n_drop_pts, small_cluster_max_points_, link);
}

void CloudPassthroughFilterNode::removePointsByMask(
    pcl::PointCloud<pcl::PointXYZ>::Ptr& cloud,
    const std::vector<char>& drop,
    const std_msgs::msg::Header& header)
{
    if (!cloud) {
        PT_WARN("删点: 点云为空, 跳过");
        return;
    }
    const uint32_t n = static_cast<uint32_t>(cloud->points.size());
    if (drop.size() != static_cast<size_t>(n)) {
        PT_WARN("删点: drop 长度与点数不符 (%zu vs %u), 跳过", drop.size(), n);
        return;
    }

    size_t marked = 0;
    for (uint32_t i = 0; i < n; ++i) {
        if (drop[i] != 0) {
            ++marked;
        }
    }
    if (marked == 0) {
        PT_INFO("删点: 无待删点, 点数保持 %u", n);
        return;
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

    PT_INFO("删点: 去掉 %zu 点, %u → %zu", marked, n, cloud->size());

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
    const std_msgs::msg::Header& header,
    const std::vector<char>* drop_mask)
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
    if (n_pass > 0 && drop_mask != nullptr &&
        drop_mask->size() == static_cast<size_t>(n_pass)) {
        for (uint32_t i = 0; i < n_pass; ++i) {
            if ((*drop_mask)[i] == 0) {
                continue;
            }
            const auto& p = pass_cloud->points[i];
            deleted_keys.insert(pack_key(p.x, p.y, p.z));
        }
    } else if (n_pass > 0) {
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

    // 1) 红=删，绿=留下，蓝=立方体内但没进直通
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
                const uint64_t key = pack_key(p.x, p.y, p.z);
                if (deleted_keys.count(key) != 0) {
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

    // 2) 范围线框：青=立方体，橙=直通
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
                       cube_min_x_, cube_max_x_,
                       cube_min_y_, cube_max_y_,
                       cube_min_z_, cube_max_z_);
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
        boxes_publisher_->publish(arr);
    }

    // 3) 有点的合成体素：合并话题 + 红删/绿留分开话题
    if (publish_occupied_voxels_ &&
        (voxels_publisher_ || voxels_deleted_publisher_ || voxels_kept_publisher_)) {
        visualization_msgs::msg::MarkerArray arr_all;
        visualization_msgs::msg::MarkerArray arr_del;
        visualization_msgs::msg::MarkerArray arr_keep;

        auto push_clear = [&](visualization_msgs::msg::MarkerArray& arr, const char* ns) {
            visualization_msgs::msg::Marker clear;
            clear.header = header;
            clear.ns = ns;
            clear.id = 0;
            clear.action = visualization_msgs::msg::Marker::DELETEALL;
            arr.markers.push_back(clear);
        };
        if (voxels_publisher_) {
            push_clear(arr_all, "occupied_voxels");
        }
        if (voxels_deleted_publisher_) {
            push_clear(arr_del, "deleted_voxels");
        }
        if (voxels_kept_publisher_) {
            push_clear(arr_keep, "kept_voxels");
        }

        size_t drawn = 0;
        size_t drawn_del = 0;
        size_t drawn_keep = 0;
        int marker_id = 1;

        auto append_voxel = [&](visualization_msgs::msg::MarkerArray& arr,
                                const char* ns,
                                bool is_bad,
                                double cx, double cy, double cz,
                                double sx, double sy, double sz, double z_top,
                                const Voxel& v) {
            visualization_msgs::msg::Marker cube;
            cube.header = header;
            cube.ns = ns;
            cube.id = marker_id++;
            cube.type = visualization_msgs::msg::Marker::CUBE;
            cube.action = visualization_msgs::msg::Marker::ADD;
            cube.pose.orientation.w = 1.0;
            cube.pose.position.x = cx;
            cube.pose.position.y = cy;
            cube.pose.position.z = cz;
            cube.scale.x = sx;
            cube.scale.y = sy;
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

            visualization_msgs::msg::Marker text;
            text.header = header;
            text.ns = ns;
            text.id = marker_id++;
            text.type = visualization_msgs::msg::Marker::TEXT_VIEW_FACING;
            text.action = visualization_msgs::msg::Marker::ADD;
            text.pose.orientation.w = 1.0;
            text.pose.position.x = cx;
            text.pose.position.y = cy;
            text.pose.position.z = z_top + 0.03;
            text.scale.z = 0.018;
            text.color.r = 1.0f;
            text.color.g = 1.0f;
            text.color.b = 0.85f;
            text.color.a = 1.0f;
            char buf[160];
            const float zspan = v.zmax - v.zmin;
            const double cell_r = std::hypot(cx, cy);
            const double line_gap = cell_r * ang_v_;
            const double thr_ratio = lookupThrRatio(cell_r);
            if (v.cluster_id > 0) {
                std::snprintf(buf, sizeof(buf),
                              "x:%.2f,y:%.2f,z:%.2f,span:%.2f,thr:%.2f,thr_z:%.2fx%.2f,t%d",
                              cx, cy, cz, zspan, v.thr_z, line_gap, thr_ratio, v.cluster_id);
            } else {
                std::snprintf(buf, sizeof(buf),
                              "x:%.2f,y:%.2f,z:%.2f,span:%.2f,thr:%.2f,thr_z:%.2fx%.2f",
                              cx, cy, cz, zspan, v.thr_z, line_gap, thr_ratio);
            }
            text.text = buf;
            arr.markers.push_back(text);
        };

        for (const auto& entry : grid_) {
            const Voxel& v = entry.second;
            if (v.count == 0 || v.idx.empty()) {
                continue;
            }
            // 画邻接表并成的大格，x/y 按实际跨度，不再撑成正方形
            const double sx = std::max(static_cast<double>(v.cell_sx), base_x_);
            const double sy = std::max(static_cast<double>(v.cell_sy), base_y_);
            const double sz = std::max(static_cast<double>(v.cell_sz), base_z_);
            const double cx = static_cast<double>(v.ix) * base_x_ + 0.5 * sx;
            const double cy = static_cast<double>(v.iy) * base_y_ + 0.5 * sy;
            const double cz = static_cast<double>(v.iz) * base_z_ + 0.5 * sz;
            const double z_top = cz + 0.5 * sz;
            const bool is_bad = bad_set.count(&v) != 0;

            if (voxels_publisher_) {
                append_voxel(arr_all, "occupied_voxels", is_bad,
                             cx, cy, cz, sx, sy, sz, z_top, v);
            }
            // 坏格进删话题，其余进留话题
            if (is_bad && voxels_deleted_publisher_) {
                append_voxel(arr_del, "deleted_voxels", true,
                             cx, cy, cz, sx, sy, sz, z_top, v);
                ++drawn_del;
            }
            if (!is_bad && voxels_kept_publisher_) {
                append_voxel(arr_keep, "kept_voxels", false,
                             cx, cy, cz, sx, sy, sz, z_top, v);
                ++drawn_keep;
            }
            ++drawn;
        }

        if (voxels_publisher_) {
            voxels_publisher_->publish(arr_all);
        }
        if (voxels_deleted_publisher_) {
            voxels_deleted_publisher_->publish(arr_del);
        }
        if (voxels_kept_publisher_) {
            voxels_kept_publisher_->publish(arr_keep);
        }
        PT_INFO("调试可视化: 红删 %zu 绿留 %zu 蓝立方体无直通 %zu "
                "体素合并 %zu 删格话题 %zu 留格话题 %zu / 总格 %zu",
                n_del, n_keep, n_blue, drawn, drawn_del, drawn_keep, grid_.size());
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
