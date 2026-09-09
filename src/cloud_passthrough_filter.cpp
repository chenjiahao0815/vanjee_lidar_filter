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

#include <Eigen/Dense>

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
    this->declare_parameter("base_xy", 0.01);
    this->declare_parameter("base_z", 0.01);
    this->declare_parameter("max_xy", 0.5);
    this->declare_parameter("max_z", 0.5);
    this->declare_parameter(
        "size_xy_bands", std::string("(1.0,4.0),(2.0,6.0),(3.5,8.0)"));
    this->declare_parameter(
        "size_z_bands", std::string("(1.0,3.0),(2.0,4.0),(3.0,5.0)"));
    this->declare_parameter("size_z_ratio", 4.0);
    this->declare_parameter("size_z_min", 0.1);
    this->declare_parameter("thr_ratio", std::string("(0.0,2.0)"));
    this->declare_parameter("thr_z_min", 0.03);
    this->declare_parameter("cluster_link_m", 0.18);
    this->declare_parameter("cluster_link_k", 6.0);
    this->declare_parameter("cluster_plane_k", 10.0);
    this->declare_parameter("min_cluster_points", 10);
    this->declare_parameter("enable_small_cluster_filter", false);
    this->declare_parameter("enable_plane_protect", true);
    this->declare_parameter("plane_protect_min_points", 80);
    this->declare_parameter("plane_protect_rms_m", 0.015);
    this->declare_parameter("plane_protect_intensity", 10.0);
    this->declare_parameter("plane_protect_nz_min", 0.9);
    this->declare_parameter("plane_protect_max_rings", 8);
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
    base_xy_ = this->get_parameter("base_xy").as_double();
    base_z_ = this->get_parameter("base_z").as_double();
    max_xy_ = this->get_parameter("max_xy").as_double();
    max_z_ = this->get_parameter("max_z").as_double();
    size_xy_bands_raw_ = this->get_parameter("size_xy_bands").as_string();
    if (!parseSizeXyBands(size_xy_bands_raw_, size_xy_bands_)) {
        RCLCPP_WARN(this->get_logger(),
                    "size_xy_bands 解析失败: '%s'，退回全程倍率 1.0",
                    size_xy_bands_raw_.c_str());
        size_xy_bands_.clear();
        size_xy_bands_.push_back(SizeXyBand{1.0e9, 1.0});
    }
    size_z_bands_raw_ = this->get_parameter("size_z_bands").as_string();
    size_z_bands_.clear();
    if (!size_z_bands_raw_.empty() &&
        !parseSizeXyBands(size_z_bands_raw_, size_z_bands_)) {
        RCLCPP_WARN(this->get_logger(),
                    "size_z_bands 解析失败: '%s'，退回 size_z_ratio",
                    size_z_bands_raw_.c_str());
        size_z_bands_.clear();
    }
    size_z_ratio_ = this->get_parameter("size_z_ratio").as_double();
    if (size_z_ratio_ <= 0.0) {
        size_z_ratio_ = 4.0;
    }
    size_z_min_ = this->get_parameter("size_z_min").as_double();
    if (size_z_min_ <= 0.0) {
        size_z_min_ = 0.1;
    }
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
    min_cluster_points_ = this->get_parameter("min_cluster_points").as_int();
    if (min_cluster_points_ < 1) {
        min_cluster_points_ = 10;
    }
    enable_small_cluster_filter_ =
        this->get_parameter("enable_small_cluster_filter").as_bool();
    enable_plane_protect_ = this->get_parameter("enable_plane_protect").as_bool();
    plane_protect_min_points_ = this->get_parameter("plane_protect_min_points").as_int();
    if (plane_protect_min_points_ < 3) {
        plane_protect_min_points_ = 80;
    }
    plane_protect_rms_m_ = this->get_parameter("plane_protect_rms_m").as_double();
    if (plane_protect_rms_m_ <= 0.0) {
        plane_protect_rms_m_ = 0.015;
    }
    plane_protect_intensity_ = this->get_parameter("plane_protect_intensity").as_double();
    if (plane_protect_intensity_ < 0.0) {
        plane_protect_intensity_ = 10.0;
    }
    plane_protect_nz_min_ = this->get_parameter("plane_protect_nz_min").as_double();
    if (plane_protect_nz_min_ < 0.0) {
        plane_protect_nz_min_ = 0.0;
    }
    if (plane_protect_nz_min_ > 1.0) {
        plane_protect_nz_min_ = 1.0;
    }
    plane_protect_max_rings_ = this->get_parameter("plane_protect_max_rings").as_int();
    if (plane_protect_max_rings_ < 1) {
        plane_protect_max_rings_ = 8;
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
                    "  角分辨率 ang_h=%.6f ang_v=%.6f, 细格 base_xy=%.3f base_z=%.3f, "
                    "横向 bands=%s 夹[%.3f,%.3f] "
                    "纵向 bands=%s (空则×%.1f) ×线间距夹[%.3f,%.3f], "
                    "thr_ratio=%s thr_z_min=%.3f, "
                    "连通团 link=%.2f m k3d=%.1f k_xy=%.1f",
                    ang_h_, ang_v_, base_xy_, base_z_,
                    size_xy_bands_raw_.c_str(), base_xy_, max_xy_,
                    size_z_bands_raw_.c_str(), size_z_ratio_,
                    size_z_min_, max_z_,
                    thr_ratio_raw_.c_str(), thr_z_min_,
                    cluster_link_m_, cluster_link_k_, cluster_plane_k_);
        RCLCPP_INFO(this->get_logger(),
                    "  删点: 每格看厚度; 平面保护=%s (点数>=%d 且 RMS<=%.3fm 且 Imed>=%.1f; "
                    "若 |nz|>=%.2f 且 线数<=%d 则否决); "
                    "小团清扫=%s (门槛<%d)",
                    enable_plane_protect_ ? "开" : "关",
                    plane_protect_min_points_, plane_protect_rms_m_,
                    plane_protect_intensity_,
                    plane_protect_nz_min_, plane_protect_max_rings_,
                    enable_small_cluster_filter_ ? "开" : "关",
                    min_cluster_points_);
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
        // 线号/强度数组要和点一起压缩，否则后面按下标会错位
        const bool track_ring = have_ring_ && ring_of_cur_.size() == before;
        const bool track_intensity =
            have_intensity_ && intensity_of_cur_.size() == before;
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
            if (track_intensity) {
                intensity_of_cur_[write] = intensity_of_cur_[i];
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
        if (track_intensity) {
            intensity_of_cur_.resize(write);
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

bool CloudPassthroughFilterNode::loadIntensityField(
    const sensor_msgs::msg::PointCloud2& msg)
{
    intensity_of_cur_.clear();
    have_intensity_ = false;

    const sensor_msgs::msg::PointField* intensity_field = nullptr;
    for (const auto& f : msg.fields) {
        if (f.name == "intensity") {
            intensity_field = &f;
            break;
        }
    }
    if (intensity_field == nullptr || msg.point_step == 0) {
        return false;
    }

    const size_t n = static_cast<size_t>(msg.width) * static_cast<size_t>(msg.height);
    if (msg.data.size() < n * msg.point_step) {
        return false;
    }
    intensity_of_cur_.resize(n);
    for (size_t i = 0; i < n; ++i) {
        const uint8_t* src =
            msg.data.data() + i * msg.point_step + intensity_field->offset;
        float v = 0.0f;
        switch (intensity_field->datatype) {
            case sensor_msgs::msg::PointField::UINT8:
            case sensor_msgs::msg::PointField::INT8:
                v = static_cast<float>(*src);
                break;
            case sensor_msgs::msg::PointField::UINT16:
            case sensor_msgs::msg::PointField::INT16: {
                uint16_t t = 0;
                std::memcpy(&t, src, sizeof(t));
                v = static_cast<float>(t);
                break;
            }
            case sensor_msgs::msg::PointField::UINT32:
            case sensor_msgs::msg::PointField::INT32: {
                uint32_t t = 0;
                std::memcpy(&t, src, sizeof(t));
                v = static_cast<float>(t);
                break;
            }
            case sensor_msgs::msg::PointField::FLOAT32: {
                std::memcpy(&v, src, sizeof(v));
                break;
            }
            case sensor_msgs::msg::PointField::FLOAT64: {
                double t = 0.0;
                std::memcpy(&t, src, sizeof(t));
                v = static_cast<float>(t);
                break;
            }
            default:
                intensity_of_cur_.clear();
                return false;
        }
        intensity_of_cur_[i] = v;
    }
    have_intensity_ = true;
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
        const bool track_intensity =
            have_intensity_ && intensity_of_cur_.size() == cloud->size();
        if (track_ring) {
            ring_scratch_.clear();
            ring_scratch_.reserve(cloud->size());
        }
        if (track_intensity) {
            intensity_scratch_.clear();
            intensity_scratch_.reserve(cloud->size());
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
                if (track_intensity) {
                    intensity_scratch_.push_back(intensity_of_cur_[i]);
                }
            }
        }
        if (track_ring) {
            ring_of_cur_.swap(ring_scratch_);
        }
        if (track_intensity) {
            intensity_of_cur_.swap(intensity_scratch_);
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
        // 线号/强度必须从原始消息取，pcl::PointXYZ 会把这些字段丢掉
        loadRingField(*msg);
        loadIntensityField(*msg);

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
        bad_voxels = collectBadVoxels(current);
        if (debug_mode_) {
            publishFilterDebug(raw_viz, current, bad_voxels, msg->header);
        }
        removePointsInBadVoxels(current, bad_voxels, msg->header);
        PT_INFO("体素删点结束: 坏格 %zu, 点数 %zu → %zu",
                bad_voxels.size(), n_before_voxel, current->size());
        if (enable_small_cluster_filter_) {
            const size_t n_before_small = current->size();
            removeSmallPointClusters(current, msg->header);
            PT_INFO("小团清扫结束: 点数 %zu → %zu (门槛<%d)",
                    n_before_small, current->size(), min_cluster_points_);
        } else {
            PT_INFO("小团清扫: 关，跳过");
        }
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
                    "  样例 r=%.1f xy_ratio=%.2f z_ratio=%.2f thr_ratio=%.2f "
                    "line_gap=%.4f mult_xy=%d mult_z=%d size_xy=%.3f size_z=%.3f thr_z=%.4f",
                    r, lookupSizeXyRatio(r), lookupSizeZRatio(r), lookupThrRatio(r),
                    s.line_gap, s.mult_xy, s.mult_z, s.size_xy, s.size_z, s.thr_z);
    }
}

VoxelScale CloudPassthroughFilterNode::computeVoxelScale(double r) const
{
    VoxelScale s;  //输入和基础值保护      根据点云距离雷达的远近来决定用多大的体素来装点云
    const double rr = std::max(r, 1e-3);
    const double base_xy = std::max(base_xy_, 1e-6);
    const double base_z = std::max(base_z_, 1e-6);

    s.line_gap = rr * ang_v_;
    s.pt_gap = s.line_gap;
    const double xy_ratio = lookupSizeXyRatio(rr);
    const double z_ratio = lookupSizeZRatio(rr);
    s.size_xy = std::min(std::max(base_xy * xy_ratio, base_xy), max_xy_);
    s.size_z = std::min(std::max(z_ratio * s.line_gap, size_z_min_), max_z_);
    s.mult_xy = static_cast<int>(std::max(1.0, std::ceil(s.size_xy / base_xy - 1e-12)));
    s.mult_z = static_cast<int>(std::max(1.0, std::ceil(s.size_z / base_z - 1e-12)));
    s.size_xy = std::min(base_xy * static_cast<double>(s.mult_xy), max_xy_);
    s.size_z = std::min(base_z * static_cast<double>(s.mult_z), max_z_);

    s.thr_z = s.line_gap * lookupThrRatio(rr);
    if (s.thr_z < thr_z_min_) {
        s.thr_z = thr_z_min_;
    }
    if (s.thr_z > s.size_z) {
        s.thr_z = s.size_z;
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

double CloudPassthroughFilterNode::lookupSizeXyRatio(double r_xy) const
{
    if (size_xy_bands_.empty()) {
        return 1.0;
    }
    const double r = std::max(r_xy, 0.0);
    // (起始距离, 倍率)：取最后一个 split_r ≤ r 的段；比第一段还近 → 不扩大
    double ratio = 1.0;
    for (const auto& band : size_xy_bands_) {
        if (r >= band.split_r) {
            ratio = band.ratio;
        } else {
            break;
        }
    }
    return ratio;
}

double CloudPassthroughFilterNode::lookupSizeZRatio(double r_xy) const
{
    // bands 空：退回常数 size_z_ratio；有 bands：与 XY 相同分段语义
    if (size_z_bands_.empty()) {
        return size_z_ratio_;
    }
    const double r = std::max(r_xy, 0.0);
    double ratio = 1.0;
    for (const auto& band : size_z_bands_) {
        if (r >= band.split_r) {
            ratio = band.ratio;
        } else {
            break;
        }
    }
    return ratio;
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
    int cell_n = 0;
    for (auto& entry : grid_) {
        entry.second.cell_id = ++cell_n;
    }
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

        const int nxy = std::max(1, n.nxy);
        const int nz = std::max(1, n.nz);
        // 键用细格角点，不是 floor(ix/n)；后者不同倍率会撞成同一格
        const int origin_ix = floorDiv(fine.ix, nxy) * nxy;
        const int origin_iy = floorDiv(fine.iy, nxy) * nxy;
        const int origin_iz = floorDiv(fine.iz, nz) * nz;
        const int64_t ckey = makeCoarseKey(origin_ix, origin_iy, origin_iz, nxy, nz);

        const double cell_sx_d = std::min(base_xy_ * static_cast<double>(nxy), max_xy_);
        const double cell_sz_d = std::min(base_z_ * static_cast<double>(nz), max_z_);
        const float cell_sx = static_cast<float>(std::max(cell_sx_d, base_xy_));
        const float cell_sz = static_cast<float>(std::max(cell_sz_d, base_z_));
        const double cube_cx = static_cast<double>(origin_ix) * base_xy_ + 0.5 * cell_sx_d;
        const double cube_cy = static_cast<double>(origin_iy) * base_xy_ + 0.5 * cell_sx_d;
        const float cell_r = static_cast<float>(std::hypot(cube_cx, cube_cy));

        auto it = grid_.find(ckey);
        if (it == grid_.end()) {
            Voxel v;
            v.ix = origin_ix;
            v.iy = origin_iy;
            v.iz = origin_iz;
            v.count = fine.count;
            v.zmin = fine.zmin;
            v.zmax = fine.zmax;
            v.r = cell_r;
            v.cx = static_cast<float>(cube_cx);
            v.cy = static_cast<float>(cube_cy);
            v.cell_sx = cell_sx;
            v.cell_sz = cell_sz;
            v.ring_mask = fine.ring_mask;
            v.idx = fine.idx;
            v.merge_nxy = nxy;
            v.merge_nz = nz;
            grid_.emplace(ckey, std::move(v));
        } else {
            Voxel& v = it->second;
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

int64_t CloudPassthroughFilterNode::makeCoarseKey(
    int ox, int oy, int oz, int nxy, int nz) const
{
    constexpr int64_t kOff = 1 << 15;
    const int64_t ux = static_cast<int64_t>(ox) + kOff;
    const int64_t uy = static_cast<int64_t>(oy) + kOff;
    const int64_t uz = static_cast<int64_t>(oz) + kOff;
    const int64_t nx = static_cast<int64_t>(std::max(1, std::min(nxy, 255)));
    const int64_t nzv = static_cast<int64_t>(std::max(1, std::min(nz, 255)));
    return (ux << 48) | (uy << 32) | (uz << 16) | (nx << 8) | nzv;
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

void CloudPassthroughFilterNode::assignClusters(std::vector<Voxel*>& all)
{
    const size_t n = all.size();
    for (Voxel* v : all) {
        if (v) {
            v->cluster_id = 0;
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

    std::vector<int> root_size(n, 0);
    for (size_t i = 0; i < n; ++i) {
        ++root_size[static_cast<size_t>(find(static_cast<int>(i)))];
    }

    // 至少两格才编号，给后面整团扫用
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
    for (size_t i = 0; i < n; ++i) {
        all[i]->cluster_id = root_to_id[static_cast<size_t>(find(static_cast<int>(i)))];
    }
    PT_INFO("连通团: 编号团 %d (link=%.2f k3d=%.1f k_xy=%.1f)，仅编号/可视化",
            next_id - 1, cluster_link_m_, cluster_link_k_, cluster_plane_k_);
}

std::vector<Voxel*> CloudPassthroughFilterNode::collectBadVoxels(
    const pcl::PointCloud<pcl::PointXYZ>::Ptr& cloud)
{
    std::vector<Voxel*> all;
    all.reserve(grid_.size());
    for (auto& entry : grid_) {
        entry.second.plane_protected = false;
        all.push_back(&entry.second);
    }
    assignClusters(all);

    // 删前保护：点数够 + 平面拟合残差小 + 强度中位够 → 整团不按厚度删
    size_t protect_clusters = 0;
    size_t protect_voxels = 0;
    if (enable_plane_protect_ && cloud && !cloud->empty()) {
        if (!have_intensity_ || intensity_of_cur_.size() != cloud->size()) {
            PT_WARN("平面保护: 本帧无对齐强度字段，跳过豁免（退回厚度规则）");
        } else {
        const uint32_t n_cloud = static_cast<uint32_t>(cloud->points.size());
        std::unordered_map<int, std::vector<Voxel*>> by_cid;
        by_cid.reserve(all.size());
        for (Voxel* vp : all) {
            if (vp->cluster_id > 0) {
                by_cid[vp->cluster_id].push_back(vp);
            }
        }

        auto try_protect = [&](std::vector<Voxel*>& voxs) {
            std::vector<uint32_t> idxs;
            idxs.reserve(256);
            for (Voxel* v : voxs) {
                for (uint32_t i : v->idx) {
                    if (i < n_cloud) {
                        idxs.push_back(i);
                    }
                }
            }
            if (static_cast<int>(idxs.size()) < plane_protect_min_points_) {
                return;
            }
            float nz = 0.0f;
            const float rms = planeFitRms(cloud, idxs, &nz);
            if (!(rms >= 0.0f) ||
                static_cast<double>(rms) > plane_protect_rms_m_) {
                return;
            }
            const float imed = clusterIntensityMedian(idxs);
            if (!(imed >= 0.0f) ||
                static_cast<double>(imed) < plane_protect_intensity_) {
                PT_INFO("平面保护否决: t%d 点数=%zu rms=%.4fm Imed=%.1f < %.1f",
                        voxs.front()->cluster_id, idxs.size(), rms, imed,
                        plane_protect_intensity_);
                return;
            }
            const int nring = clusterRingCount(cloud, idxs);
            if (static_cast<double>(nz) >= plane_protect_nz_min_ &&
                nring <= plane_protect_max_rings_) {
                PT_INFO("平面保护否决: t%d 点数=%zu rms=%.4fm Imed=%.1f "
                        "|nz|=%.2f 线数=%d (水平少线)",
                        voxs.front()->cluster_id, idxs.size(), rms, imed,
                        nz, nring);
                return;
            }
            ++protect_clusters;
            for (Voxel* v : voxs) {
                if (!v->plane_protected) {
                    v->plane_protected = true;
                    ++protect_voxels;
                }
            }
            PT_INFO("平面保护: t%d 点数=%zu rms=%.4fm Imed=%.1f |nz|=%.2f 线=%d → 整团不删",
                    voxs.front()->cluster_id, idxs.size(), rms, imed, nz, nring);
        };

        for (auto& entry : by_cid) {
            try_protect(entry.second);
        }
        // 单格但点数够：也做拟合保护
        for (Voxel* vp : all) {
            if (vp->cluster_id != 0 || vp->plane_protected) {
                continue;
            }
            std::vector<Voxel*> one{vp};
            try_protect(one);
        }
        }  // have_intensity_
    }

    std::vector<Voxel*> bad;
    bad.reserve(grid_.size());

    size_t thick_n = 0;
    size_t protect_skip = 0;
    float sample_span = 0.0f;
    float sample_thr = 0.0f;
    bool have_sample = false;

    for (Voxel* vp : all) {
        Voxel& v = *vp;
        if (v.plane_protected) {
            ++protect_skip;
            continue;
        }
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

    PT_INFO("坏体素判定: 总格 %zu, 厚留 %zu, 平面保护格 %zu (团 %zu), 坏 %zu",
            grid_.size(), thick_n, protect_skip, protect_clusters, bad.size());
    if (have_sample) {
        PT_INFO("坏体素样例: span=%.4f < thr_z=%.4f", sample_span, sample_thr);
    }
    return bad;
}

float CloudPassthroughFilterNode::planeFitRms(
    const pcl::PointCloud<pcl::PointXYZ>::Ptr& cloud,
    const std::vector<uint32_t>& idxs,
    float* abs_nz_out) const
{
    if (abs_nz_out != nullptr) {
        *abs_nz_out = -1.0f;
    }
    if (!cloud || idxs.size() < 3) {
        return -1.0f;
    }
    const uint32_t n_cloud = static_cast<uint32_t>(cloud->points.size());
    double sx = 0.0;
    double sy = 0.0;
    double sz = 0.0;
    size_t n = 0;
    for (uint32_t i : idxs) {
        if (i >= n_cloud) {
            continue;
        }
        const auto& p = cloud->points[i];
        if (!std::isfinite(p.x) || !std::isfinite(p.y) || !std::isfinite(p.z)) {
            continue;
        }
        sx += static_cast<double>(p.x);
        sy += static_cast<double>(p.y);
        sz += static_cast<double>(p.z);
        ++n;
    }
    if (n < 3) {
        return -1.0f;
    }
    const double inv = 1.0 / static_cast<double>(n);
    const double cx = sx * inv;
    const double cy = sy * inv;
    const double cz = sz * inv;

    Eigen::Matrix3d cov = Eigen::Matrix3d::Zero();
    for (uint32_t i : idxs) {
        if (i >= n_cloud) {
            continue;
        }
        const auto& p = cloud->points[i];
        if (!std::isfinite(p.x) || !std::isfinite(p.y) || !std::isfinite(p.z)) {
            continue;
        }
        const double dx = static_cast<double>(p.x) - cx;
        const double dy = static_cast<double>(p.y) - cy;
        const double dz = static_cast<double>(p.z) - cz;
        cov(0, 0) += dx * dx;
        cov(0, 1) += dx * dy;
        cov(0, 2) += dx * dz;
        cov(1, 1) += dy * dy;
        cov(1, 2) += dy * dz;
        cov(2, 2) += dz * dz;
    }
    cov(1, 0) = cov(0, 1);
    cov(2, 0) = cov(0, 2);
    cov(2, 1) = cov(1, 2);
    cov *= inv;

    Eigen::SelfAdjointEigenSolver<Eigen::Matrix3d> es(cov);
    if (es.info() != Eigen::Success) {
        return -1.0f;
    }
    const double lambda0 = std::max(0.0, es.eigenvalues()(0));
    if (abs_nz_out != nullptr) {
        // 最小特征值对应法向；取 |z| 分量判断是否接近水平面
        *abs_nz_out = static_cast<float>(std::fabs(es.eigenvectors()(2, 0)));
    }
    return static_cast<float>(std::sqrt(lambda0));
}

int CloudPassthroughFilterNode::clusterRingCount(
    const pcl::PointCloud<pcl::PointXYZ>::Ptr& cloud,
    const std::vector<uint32_t>& idxs) const
{
    if (!cloud || idxs.empty()) {
        return 0;
    }
    const uint32_t n_cloud = static_cast<uint32_t>(cloud->points.size());
    std::unordered_set<int> rings;
    rings.reserve(32);
    for (uint32_t i : idxs) {
        if (i >= n_cloud) {
            continue;
        }
        const auto& p = cloud->points[i];
        if (!std::isfinite(p.x) || !std::isfinite(p.y) || !std::isfinite(p.z)) {
            continue;
        }
        rings.insert(ringOfPoint(i, p.x, p.y, p.z));
    }
    return static_cast<int>(rings.size());
}

float CloudPassthroughFilterNode::clusterIntensityMedian(
    const std::vector<uint32_t>& idxs) const
{
    if (!have_intensity_ || idxs.empty()) {
        return -1.0f;
    }
    const size_t n_i = intensity_of_cur_.size();
    std::vector<float> vals;
    vals.reserve(idxs.size());
    for (uint32_t i : idxs) {
        if (static_cast<size_t>(i) >= n_i) {
            continue;
        }
        const float v = intensity_of_cur_[i];
        if (std::isfinite(v)) {
            vals.push_back(v);
        }
    }
    if (vals.empty()) {
        return -1.0f;
    }
    const size_t mid = vals.size() / 2;
    std::nth_element(vals.begin(), vals.begin() + static_cast<std::ptrdiff_t>(mid), vals.end());
    return vals[mid];
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

void CloudPassthroughFilterNode::removeSmallPointClusters(
    pcl::PointCloud<pcl::PointXYZ>::Ptr& cloud,
    const std_msgs::msg::Header& header)
{
    if (!cloud || cloud->empty()) {
        return;
    }
    if (min_cluster_points_ <= 1) {
        return;
    }

    const size_t n = cloud->points.size();
    std::vector<int> parent(n);
    for (size_t i = 0; i < n; ++i) {
        parent[i] = static_cast<int>(i);
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

    // 空间哈希：格子边长取连通下限，邻域按距离自适应搜索
    const float cell = static_cast<float>(std::max(cluster_link_m_, 0.05));
    const float inv_cell = 1.0f / cell;
    const float k3d = static_cast<float>(cluster_link_k_);
    const float kxy = static_cast<float>(cluster_plane_k_);
    const float link_min = static_cast<float>(cluster_link_m_);
    const float ang = static_cast<float>(ang_v_);

    struct CellKey {
        int ix{0};
        int iy{0};
        int iz{0};
        bool operator==(const CellKey& o) const
        {
            return ix == o.ix && iy == o.iy && iz == o.iz;
        }
    };
    struct CellHash {
        size_t operator()(const CellKey& k) const
        {
            return (static_cast<size_t>(k.ix) * 73856093u) ^
                   (static_cast<size_t>(k.iy) * 19349663u) ^
                   (static_cast<size_t>(k.iz) * 83492791u);
        }
    };

    std::unordered_map<CellKey, std::vector<int>, CellHash> buckets;
    buckets.reserve(n * 2 + 1);
    for (size_t i = 0; i < n; ++i) {
        const auto& p = cloud->points[i];
        CellKey key{
            static_cast<int>(std::floor(p.x * inv_cell)),
            static_cast<int>(std::floor(p.y * inv_cell)),
            static_cast<int>(std::floor(p.z * inv_cell))};
        buckets[key].push_back(static_cast<int>(i));
    }

    auto shouldJoin = [&](int ia, int ib) {
        const auto& a = cloud->points[static_cast<size_t>(ia)];
        const auto& b = cloud->points[static_cast<size_t>(ib)];
        const float dx = a.x - b.x;
        const float dy = a.y - b.y;
        const float dz = a.z - b.z;
        const float ra = std::hypot(a.x, a.y);
        const float rb = std::hypot(b.x, b.y);
        const float rr = std::max(std::max(ra, rb), 1e-3f);
        const float gap = rr * ang;
        const float link3d = std::max(link_min, k3d * gap);
        const float dxy2 = dx * dx + dy * dy;
        const float d3 = dxy2 + dz * dz;
        if (d3 <= link3d * link3d) {
            return true;
        }
        const float z_band = std::max(0.04f, 2.0f * gap);
        const float link_xy = std::max(link3d, kxy * gap);
        return std::fabs(dz) <= z_band && dxy2 <= link_xy * link_xy;
    };

    for (size_t i = 0; i < n; ++i) {
        const auto& p = cloud->points[i];
        const float ra = std::hypot(p.x, p.y);
        const float gap = std::max(ra, 1e-3f) * ang;
        const float link = std::max(
            link_min, std::max(k3d * gap, kxy * gap));
        const int rad = std::max(1, static_cast<int>(std::ceil(link * inv_cell)));
        const int ix0 = static_cast<int>(std::floor(p.x * inv_cell));
        const int iy0 = static_cast<int>(std::floor(p.y * inv_cell));
        const int iz0 = static_cast<int>(std::floor(p.z * inv_cell));
        for (int dx = -rad; dx <= rad; ++dx) {
            for (int dy = -rad; dy <= rad; ++dy) {
                for (int dz = -rad; dz <= rad; ++dz) {
                    auto it = buckets.find(CellKey{ix0 + dx, iy0 + dy, iz0 + dz});
                    if (it == buckets.end()) {
                        continue;
                    }
                    for (int j : it->second) {
                        if (j <= static_cast<int>(i)) {
                            continue;
                        }
                        if (shouldJoin(static_cast<int>(i), j)) {
                            unite(static_cast<int>(i), j);
                        }
                    }
                }
            }
        }
    }

    std::vector<int> root_size(n, 0);
    std::vector<int> root_of(n, 0);
    for (size_t i = 0; i < n; ++i) {
        const int r = find(static_cast<int>(i));
        root_of[i] = r;
        ++root_size[static_cast<size_t>(r)];
    }

    size_t small_clusters = 0;
    size_t marked = 0;
    std::vector<char> drop(n, 0);
    for (size_t i = 0; i < n; ++i) {
        const int r = root_of[i];
        if (root_size[static_cast<size_t>(r)] < min_cluster_points_) {
            if (drop[i] == 0) {
                drop[i] = 1;
                ++marked;
            }
            if (r == static_cast<int>(i)) {
                ++small_clusters;
            }
        }
    }

    if (marked == 0) {
        PT_INFO("小团清扫: 团门槛=%d, 无小团, 点数保持 %zu",
                min_cluster_points_, n);
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
    for (size_t i = 0; i < n; ++i) {
        if (drop[i] != 0) {
            if (removed) {
                removed->points.push_back(cloud->points[i]);
            }
            continue;
        }
        if (write != i) {
            cloud->points[write] = cloud->points[i];
        }
        ++write;
    }
    cloud->points.resize(write);
    cloud->width = static_cast<uint32_t>(write);
    cloud->height = 1;
    cloud->is_dense = true;

    PT_INFO("小团清扫: 小团 %zu 个, 去掉 %zu 点, %zu → %zu (门槛<%d)",
            small_clusters, marked, n, cloud->size(), min_cluster_points_);

    if (removed) {
        removed->width = static_cast<uint32_t>(removed->points.size());
        publish_cloud(removed, header, removed_publisher_, "小团被删点");
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

    std::unordered_set<uint64_t> protect_keys;
    protect_keys.reserve(256);
    if (n_pass > 0) {
        for (const auto& entry : grid_) {
            const Voxel& v = entry.second;
            if (!v.plane_protected) {
                continue;
            }
            for (uint32_t idx : v.idx) {
                if (idx < n_pass) {
                    const auto& p = pass_cloud->points[idx];
                    protect_keys.insert(pack_key(p.x, p.y, p.z));
                }
            }
        }
    }

    // 1) 红=删，青=平面保护留下，绿=其它留下，蓝=立方体内但没进直通
    size_t n_del = 0;
    size_t n_keep = 0;
    size_t n_fit = 0;
    size_t n_blue = 0;
    if (verdict_publisher_ && raw_cloud && !raw_cloud->empty()) {
        const uint32_t n_raw = static_cast<uint32_t>(raw_cloud->points.size());
        std::vector<uint32_t> pick;
        std::vector<uint32_t> color;
        pick.reserve(n_raw);
        color.reserve(n_raw);
        const uint32_t c_del = packRgb(240, 40, 40);
        const uint32_t c_keep = packRgb(40, 220, 80);
        const uint32_t c_fit = packRgb(40, 220, 220);
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
                } else if (protect_keys.count(key) != 0) {
                    pick.push_back(i);
                    color.push_back(c_fit);
                    ++n_fit;
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
            } else if (v.plane_protected) {
                cube.color.r = 0.15f;
                cube.color.g = 0.85f;
                cube.color.b = 0.9f;
                cube.color.a = 0.32f;
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
            char buf[144];
            const float zspan = v.zmax - v.zmin;
            const double xy_r = lookupSizeXyRatio(static_cast<double>(v.r));
            const double z_r = lookupSizeZRatio(static_cast<double>(v.r));
            if (v.plane_protected && v.cluster_id > 0) {
                std::snprintf(buf, sizeof(buf),
                              "size_xy_bands:%.1f,size_z_bands:%.1f,span:%.2f,thr:%.2f,t%d,fit",
                              xy_r, z_r, zspan, v.thr_z, v.cluster_id);
            } else if (v.plane_protected) {
                std::snprintf(buf, sizeof(buf),
                              "size_xy_bands:%.1f,size_z_bands:%.1f,span:%.2f,thr:%.2f,fit",
                              xy_r, z_r, zspan, v.thr_z);
            } else if (v.cluster_id > 0) {
                std::snprintf(buf, sizeof(buf),
                              "size_xy_bands:%.1f,size_z_bands:%.1f,span:%.2f,thr:%.2f,t%d",
                              xy_r, z_r, zspan, v.thr_z, v.cluster_id);
            } else {
                std::snprintf(buf, sizeof(buf),
                              "size_xy_bands:%.1f,size_z_bands:%.1f,span:%.2f,thr:%.2f",
                              xy_r, z_r, zspan, v.thr_z);
            }
            text.text = buf;
            arr.markers.push_back(text);
        };

        for (const auto& entry : grid_) {
            const Voxel& v = entry.second;
            if (v.count == 0 || v.idx.empty()) {
                continue;
            }
            // 画合并后的那一格：边长 = n×base，夹在 max_xy/max_z 内
            const double sx = std::min(
                std::max(static_cast<double>(v.cell_sx), base_xy_), max_xy_);
            const double sy = sx;
            const double sz = std::min(
                std::max(static_cast<double>(v.cell_sz), base_z_), max_z_);
            const double cx = static_cast<double>(v.ix) * base_xy_ + 0.5 * sx;
            const double cy = static_cast<double>(v.iy) * base_xy_ + 0.5 * sy;
            const double cz = static_cast<double>(v.iz) * base_z_ + 0.5 * sz;
            const double z_top = cz + 0.5 * sz;
            const bool is_bad = bad_set.count(&v) != 0;

            if (voxels_publisher_) {
                append_voxel(arr_all, "occupied_voxels", is_bad,
                             cx, cy, cz, sx, sy, sz, z_top, v);
            }
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
        PT_INFO("调试可视化: 红删 %zu 青平面保护 %zu 绿留 %zu 蓝(立方体无直通) %zu "
                "体素合并 %zu 删格话题 %zu 留格话题 %zu / 总格 %zu",
                n_del, n_fit, n_keep, n_blue, drawn, drawn_del, drawn_keep, grid_.size());
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
