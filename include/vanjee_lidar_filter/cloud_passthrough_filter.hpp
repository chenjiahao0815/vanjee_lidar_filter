#ifndef VANJEE_LIDAR_FILTER__CLOUD_PASSTHROUGH_FILTER_HPP_
#define VANJEE_LIDAR_FILTER__CLOUD_PASSTHROUGH_FILTER_HPP_

#include <rclcpp/rclcpp.hpp>
#include <sensor_msgs/msg/point_cloud2.hpp>
#include <std_msgs/msg/header.hpp>

#include <pcl/point_cloud.h>
#include <pcl/point_types.h>

#include <array>
#include <chrono>
#include <cstdint>
#include <memory>
#include <mutex>
#include <queue>
#include <string>
#include <unordered_map>
#include <unordered_set>
#include <vector>

class PointCloudMemoryPool {
public:
    PointCloudMemoryPool(size_t initial_pool_size = 10, size_t reserve_size = 100000);
    pcl::PointCloud<pcl::PointXYZ>::Ptr acquire();
    void release(pcl::PointCloud<pcl::PointXYZ>::Ptr cloud);

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

// 距离档：近档 / 远档，边长和门槛构造时算死
struct VoxelBand {
    double r{0.0};
    double line_gap{0.0};
    double pt_gap{0.0};
    double size_xy{0.0};
    double size_z{0.0};
    double thr_z{0.0};
    double inv_xy{0.0};
    double inv_z{0.0};
};

struct Voxel {
    uint32_t count{0};
    float zmin{0.0f};
    float zmax{0.0f};
    std::vector<uint32_t> idx;
    bool keep{true};
};

class CloudPassthroughFilterNode : public rclcpp::Node {
public:
    CloudPassthroughFilterNode();

private:
    void declare_and_load_parameters();
    bool parse_filter_order(const std::string& raw, std::vector<int>& order) const;
    std::string order_string() const;
    void log_startup() const;

    bool shouldFrameLog() const;
    void updateFrameLogGate();

    void removeNonFinitePointsInPlace(pcl::PointCloud<pcl::PointXYZ>::Ptr cloud);
    void publish_cloud(
        const pcl::PointCloud<pcl::PointXYZ>::Ptr& cloud,
        const std_msgs::msg::Header& header,
        const rclcpp::Publisher<sensor_msgs::msg::PointCloud2>::SharedPtr& pub,
        const char* description);

    pcl::PointCloud<pcl::PointXYZ>::Ptr passthrough_filter_cpu(
        const pcl::PointCloud<pcl::PointXYZ>::Ptr& cloud,
        double lower_limit,
        double upper_limit,
        int dimension);

    void replace_current(
        pcl::PointCloud<pcl::PointXYZ>::Ptr& current,
        pcl::PointCloud<pcl::PointXYZ>::Ptr next);

    void cloud_callback(const sensor_msgs::msg::PointCloud2::SharedPtr msg);

    // 体素删点，接在直通滤波之后写
    void buildVoxelSizeTable();
    void buildVoxelGrid(const pcl::PointCloud<pcl::PointXYZ>::Ptr& cloud);
    void markFlatVoxels();
    pcl::PointCloud<pcl::PointXYZ>::Ptr extractByFlag(
        const pcl::PointCloud<pcl::PointXYZ>::Ptr& cloud,
        bool want_keep);
    int pickBand(double r) const;
    int64_t makeKey(int band, int ix, int iy, int iz) const;

    rclcpp::Subscription<sensor_msgs::msg::PointCloud2>::SharedPtr subscription_;
    rclcpp::Publisher<sensor_msgs::msg::PointCloud2>::SharedPtr publisher_;
    rclcpp::Publisher<sensor_msgs::msg::PointCloud2>::SharedPtr removed_publisher_;
    std::unique_ptr<PointCloudMemoryPool> memory_pool_;

    std::array<AxisSpec, 3> axes_{};
    std::vector<int> filter_order_;
    std::string filter_order_raw_;
    std::string input_topic_;
    std::string output_topic_;
    std::string removed_topic_;
    std::string debug_topic_x_;
    std::string debug_topic_y_;
    std::string debug_topic_z_;

    int memory_pool_size_{10};
    int memory_pool_reserve_{1000000};
    bool debug_mode_{false};
    double cloud_log_interval_sec_{1.0};

    bool frame_log_enabled_{false};
    std::chrono::steady_clock::time_point last_cloud_log_time_{};

    // 体素参数
    double ang_h_{0.0};
    double ang_v_{0.0};
    double r_max_{8.0};
    double base_xy_{0.05};
    double base_z_{0.05};
    double max_xy_{0.5};
    double max_z_{0.5};
    double thr_ratio_{0.5};
    bool enable_voxel_filter_{false};

    std::array<VoxelBand, 2> bands_{};
    std::unordered_map<int64_t, Voxel> grid_;
};

#endif  // VANJEE_LIDAR_FILTER__CLOUD_PASSTHROUGH_FILTER_HPP_
