#include "vanjee_lidar_filter/cloud_passthrough_filter.hpp"

#include <pcl_conversions/pcl_conversions.h>

#include <algorithm>
#include <cctype>
#include <cmath>
#include <cstdio>
#include <stdexcept>

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
        if (!removed_topic_.empty()) {
            removed_publisher_ = this->create_publisher<sensor_msgs::msg::PointCloud2>(
                removed_topic_, rclcpp::QoS{5}.best_effort());
        }
    }

    buildVoxelSizeTable();
    grid_.reserve(4096);
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

    this->declare_parameter("enable_voxel_filter", false);
    this->declare_parameter("ang_h", 0.0034906585);   // 约 0.2 deg
    this->declare_parameter("ang_v", 0.0174532925);   // 约 1.0 deg
    this->declare_parameter("r_max", 8.0);
    this->declare_parameter("base_xy", 0.05);
    this->declare_parameter("base_z", 0.05);
    this->declare_parameter("max_xy", 0.5);
    this->declare_parameter("max_z", 0.5);
    this->declare_parameter("thr_ratio", 0.5);

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
    ang_h_ = this->get_parameter("ang_h").as_double();
    ang_v_ = this->get_parameter("ang_v").as_double();
    r_max_ = this->get_parameter("r_max").as_double();
    base_xy_ = this->get_parameter("base_xy").as_double();
    base_z_ = this->get_parameter("base_z").as_double();
    max_xy_ = this->get_parameter("max_xy").as_double();
    max_z_ = this->get_parameter("max_z").as_double();
    thr_ratio_ = this->get_parameter("thr_ratio").as_double();

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
    }
    RCLCPP_INFO(this->get_logger(),
                "  帧日志间隔: %.2f s (仅 debug_mode=false 时限频)",
                cloud_log_interval_sec_);
    RCLCPP_INFO(this->get_logger(), "  体素删点: %s", enable_voxel_filter_ ? "开" : "关");
    if (enable_voxel_filter_) {
        for (size_t i = 0; i < bands_.size(); ++i) {
            const auto& b = bands_[i];
            RCLCPP_INFO(this->get_logger(),
                        "  档%zu r=%.2f size_xy=%.3f size_z=%.3f thr_z=%.3f",
                        i, b.r, b.size_xy, b.size_z, b.thr_z);
        }
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
    for (const auto& p : cloud->points) {
        const float value = (dimension == 0) ? p.x : ((dimension == 1) ? p.y : p.z);
        if (value >= lower_limit && value <= upper_limit) {
            filtered->points.push_back(p);
        }
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

    // 直通滤波到这里结束。体素删点接在下面，enable_voxel_filter 打开后走。
    if (enable_voxel_filter_ && current && !current->empty()) {
        grid_.clear();
        buildVoxelGrid(current);
        markFlatVoxels();
        auto kept = extractByFlag(current, true);
        if (debug_mode_ && removed_publisher_) {
            auto removed = extractByFlag(current, false);
            publish_cloud(removed, msg->header, removed_publisher_, "被删点");
            memory_pool_->release(removed);
        }
        replace_current(current, kept);
        PT_INFO("体素删点后: %zu → %zu 点", n_in, current->size());
    }

    publish_cloud(current, msg->header, publisher_, "最终输出");
    memory_pool_->release(current);

    const auto t1 = std::chrono::high_resolution_clock::now();
    const auto total_ms =
        std::chrono::duration_cast<std::chrono::milliseconds>(t1 - t0).count();
    PT_INFO("点云处理完成，耗时 %ld ms", total_ms);
}

// ---------- 体素删点 ----------

void CloudPassthroughFilterNode::buildVoxelSizeTable()
{
    // 开机算近、远两套格子尺寸和删点门槛，回调只查表
    const double r_near = std::max(r_max_ * 0.5, 1e-3);
    const double r_far = std::max(r_max_, r_near + 1e-3);
    const double r_list[2] = {r_near, r_far};

    for (size_t i = 0; i < bands_.size(); ++i) {
        VoxelBand& b = bands_[i];
        b.r = r_list[i];
        b.line_gap = b.r * ang_v_;
        b.pt_gap = b.r * ang_h_;

        // 线间距是基础边长的几倍，向上取整；再夹到上限
        const double mult_z = std::max(1.0, std::ceil(b.line_gap / std::max(base_z_, 1e-6)));
        const double mult_xy = std::max(1.0, std::ceil(b.pt_gap / std::max(base_xy_, 1e-6)));
        b.size_z = std::min(base_z_ * mult_z, max_z_);
        b.size_xy = std::min(base_xy_ * mult_xy, max_xy_);
        b.thr_z = b.line_gap * thr_ratio_;

        // 竖直格子至少盖住两条扫描线，否则墙上每格也只有一条线，会全删
        const double min_size_z = 2.0 * b.line_gap;
        if (b.size_z < min_size_z) {
            RCLCPP_WARN(this->get_logger(),
                        "档%zu size_z=%.4f < 2*line_gap=%.4f，抬到 %.4f",
                        i, b.size_z, min_size_z, min_size_z);
            b.size_z = min_size_z;
        }
        // 门槛必须小于格子高度，否则跨度永远够不着，等于不删
        if (b.thr_z >= b.size_z) {
            const double new_thr = 0.5 * b.size_z;
            RCLCPP_WARN(this->get_logger(),
                        "档%zu thr_z=%.4f >= size_z=%.4f，压到 %.4f",
                        i, b.thr_z, b.size_z, new_thr);
            b.thr_z = new_thr;
        }

        b.inv_xy = 1.0 / std::max(b.size_xy, 1e-6);
        b.inv_z = 1.0 / std::max(b.size_z, 1e-6);
    }
}

void CloudPassthroughFilterNode::buildVoxelGrid(
    const pcl::PointCloud<pcl::PointXYZ>::Ptr& cloud)
{
    if (!cloud || cloud->empty()) {
        return;
    }

    const uint32_t n = static_cast<uint32_t>(cloud->points.size());
    uint32_t loaded = 0;

    for (uint32_t i = 0; i < n; ++i) {
        const auto& p = cloud->points[i];
        if (!std::isfinite(p.x) || !std::isfinite(p.y) || !std::isfinite(p.z)) {
            continue;
        }

        const double r = std::hypot(static_cast<double>(p.x), static_cast<double>(p.y));
        if (r > r_max_) {
            continue;
        }

        const int band = pickBand(r);
        const VoxelBand& b = bands_[static_cast<size_t>(band)];

        const int ix = static_cast<int>(std::floor(static_cast<double>(p.x) * b.inv_xy));
        const int iy = static_cast<int>(std::floor(static_cast<double>(p.y) * b.inv_xy));
        const int iz = static_cast<int>(std::floor(static_cast<double>(p.z) * b.inv_z));

        const int64_t key = makeKey(band, ix, iy, iz);
        auto it = grid_.find(key);
        if (it == grid_.end()) {
            Voxel v;
            v.count = 1;
            v.zmin = p.z;
            v.zmax = p.z;
            v.idx.push_back(i);
            v.keep = true;
            grid_.emplace(key, std::move(v));
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
        }
        ++loaded;
    }

    PT_INFO("体素装格: %u 点装入 %zu 格", loaded, grid_.size());
}

void CloudPassthroughFilterNode::markFlatVoxels()
{
    size_t kept = 0;
    size_t dropped = 0;

    for (auto& entry : grid_) {
        const int64_t key = entry.first;
        Voxel& v = entry.second;

        const int band = static_cast<int>((key >> 60) & 0x3);
        const size_t bi = static_cast<size_t>(std::clamp(band, 0, 1));
        const float thr = static_cast<float>(bands_[bi].thr_z);
        const float span = v.zmax - v.zmin;

        v.keep = span >= thr;
        if (v.keep) {
            ++kept;
        } else {
            ++dropped;
        }
    }

    PT_INFO("体素判平: 留 %zu 格, 删 %zu 格", kept, dropped);
}

pcl::PointCloud<pcl::PointXYZ>::Ptr CloudPassthroughFilterNode::extractByFlag(
    const pcl::PointCloud<pcl::PointXYZ>::Ptr& cloud,
    bool want_keep)
{
    auto out = memory_pool_->acquire();
    out->height = 1;
    out->is_dense = true;

    if (!cloud || cloud->empty()) {
        out->width = 0;
        return out;
    }

    const uint32_t n = static_cast<uint32_t>(cloud->points.size());
    std::vector<char> assigned(n, 0);
    for (const auto& entry : grid_) {
        for (uint32_t idx : entry.second.idx) {
            if (idx < n) {
                assigned[idx] = 1;
            }
        }
    }

    size_t out_count = 0;
    for (const auto& entry : grid_) {
        if (entry.second.keep == want_keep) {
            out_count += entry.second.idx.size();
        }
    }
    if (want_keep) {
        for (uint32_t i = 0; i < n; ++i) {
            if (!assigned[i]) {
                ++out_count;
            }
        }
    }

    out->points.reserve(out_count);
    for (const auto& entry : grid_) {
        if (entry.second.keep != want_keep) {
            continue;
        }
        for (uint32_t idx : entry.second.idx) {
            if (idx < n) {
                out->points.push_back(cloud->points[idx]);
            }
        }
    }
    if (want_keep) {
        for (uint32_t i = 0; i < n; ++i) {
            if (!assigned[i]) {
                out->points.push_back(cloud->points[i]);
            }
        }
    }

    out->width = static_cast<uint32_t>(out->points.size());
    PT_INFO("体素抽点(%s): %zu → %zu",
            want_keep ? "保留" : "删除",
            cloud->size(),
            out->size());
    return out;
}

int CloudPassthroughFilterNode::pickBand(double r) const
{
    if (r < r_max_ * 0.5) {
        return 0;
    }
    return 1;
}

int64_t CloudPassthroughFilterNode::makeKey(int band, int ix, int iy, int iz) const
{
    constexpr int64_t kOffset = 1 << 19;
    const int64_t ux = static_cast<int64_t>(ix) + kOffset;
    const int64_t uy = static_cast<int64_t>(iy) + kOffset;
    const int64_t uz = static_cast<int64_t>(iz) + kOffset;
    return (static_cast<int64_t>(band) << 60) | (ux << 40) | (uy << 20) | uz;
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
