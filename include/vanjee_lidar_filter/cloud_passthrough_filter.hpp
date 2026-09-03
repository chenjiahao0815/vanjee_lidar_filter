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

// 按当前距离和角分辨率算出来的一格尺寸与删点门槛
struct VoxelScale {
    int mult_xy{1};
    int mult_z{1};
    double line_gap{0.0};
    double pt_gap{0.0};
    double size_xy{0.0};
    double size_z{0.0};
    double thr_z{0.0};
    double inv_xy{0.0};
    double inv_z{0.0};
};

struct Voxel {
    int ix{0};
    int iy{0};
    int iz{0};
    uint32_t count{0};
    float zmin{0.0f};
    float zmax{0.0f};
    float thr_z{0.0f};
    float r{0.0f};
    float cx{0.0f};
    float cy{0.0f};
    float size_xy{0.0f};
    float size_z{0.0f};
    uint64_t ring_mask{0};
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

    void logVoxelScaleSamples() const;
    VoxelScale computeVoxelScale(double r) const;
    bool inOurCube(float x, float y, float z) const;
    void buildVoxelGrid(const pcl::PointCloud<pcl::PointXYZ>::Ptr& cloud);
    void mergeFineVoxels();
    // 接收有点的体素列表，返回需要删除的体素；真正删点另写
    std::vector<Voxel*> collectBadVoxels();
    void markMultiRingClusters(std::vector<Voxel*>& all);
    void removePointsInBadVoxels(
        pcl::PointCloud<pcl::PointXYZ>::Ptr& cloud,
        const std::vector<Voxel*>& bad_voxels,
        const std_msgs::msg::Header& header);
    int64_t makeKey(int ix, int iy, int iz) const;
    int floorDiv(int a, int b) const;
    int ringId(float x, float y, float z) const;
    void addRing(Voxel& v, float x, float y, float z) const;
    bool hasNeighborSupport(const Voxel& v, const std::vector<Voxel*>& all) const;
    static int ringBitCount(uint64_t mask);

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
    bool debug_mode_{true};
    double cloud_log_interval_sec_{1.0};

    bool frame_log_enabled_{false};
    std::chrono::steady_clock::time_point last_cloud_log_time_{};

    // 我们自己的立方体，和直通滤波无关；以雷达原点为中心，长宽高是参数
    double cube_length_{4.0};
    double cube_width_{4.0};
    double cube_height_{2.0};

    // 体素参数：边长 = 默认值 × 整数倍率；倍率由距离×角分辨率相对默认值向上取整
    double ang_h_{0.0};
    double ang_v_{0.0};
    double r_max_{8.0};
    double base_xy_{0.01};
    double base_z_{0.01};
    double max_xy_{0.5};
    double max_z_{0.5};
    double thr_ratio_{0.5};
    double cluster_link_m_{0.18};
    double cluster_link_k_{6.0};
    double cluster_plane_k_{10.0};
    bool enable_voxel_filter_{true};

    std::unordered_map<int64_t, Voxel> grid_;
    std::unordered_map<int64_t, Voxel> fine_grid_;
};

#endif  // VANJEE_LIDAR_FILTER__CLOUD_PASSTHROUGH_FILTER_HPP_
