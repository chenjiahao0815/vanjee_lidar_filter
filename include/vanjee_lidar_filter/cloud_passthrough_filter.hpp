#ifndef VANJEE_LIDAR_FILTER__CLOUD_PASSTHROUGH_FILTER_HPP_
#define VANJEE_LIDAR_FILTER__CLOUD_PASSTHROUGH_FILTER_HPP_

#include <rclcpp/rclcpp.hpp>
#include <sensor_msgs/msg/point_cloud2.hpp>
#include <std_msgs/msg/header.hpp>
#include <visualization_msgs/msg/marker_array.hpp>

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

// XY 分段倍率
struct SizeXyBand {
    double split_r{0.0};
    double ratio{1.0};
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
    float cell_sx{0.0f};
    float cell_sz{0.0f};
    int merge_nxy{1};  // 合进本格时用的横向倍率
    int merge_nz{1};
    uint64_t ring_mask{0};
    std::vector<uint32_t> idx;
    int cluster_id{0};  // 连通团编号，0=未成团/单格；可视化/日志用
    int cell_id{0};     // 本帧体素格子编号，可视化/删点日志对照用
    bool restored{false};  // 小格本判删、大格捞回：青框可视化
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
    // 从原始消息里取真实线号；没有 ring 字段时返回 false，退回几何估计
    bool loadRingField(const sensor_msgs::msg::PointCloud2& msg);
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
    bool parseSizeXyBands(const std::string& text, std::vector<SizeXyBand>& out) const;
    double lookupSizeXyRatio(double r_xy) const;
    double lookupSizeZRatio(double r_xy) const;
    double lookupThrRatio(double r_xy) const;
    bool inOurCube(float x, float y, float z) const;
    bool inPassthrough(float x, float y, float z) const;
    void buildVoxelGrid(const pcl::PointCloud<pcl::PointXYZ>::Ptr& cloud);
    void mergeFineVoxels();
    // 接收有点的体素列表，返回需要删除的体素；真正删点另写
    std::vector<Voxel*> collectBadVoxels();
    void assignClusters(std::vector<Voxel*>& all);
    void useSizeXyBands(int which);  // 1=小格 2=大格
    // 大格捞回门闩：ring 滑窗种子 + 向两侧生长；无真实 ring 则跳过
    // drop 入参=小格已标删；出参=最终仍删。restored 写出捞回点。
    void gateRestoreByScanLines(
        const pcl::PointCloud<pcl::PointXYZ>::Ptr& cloud,
        const std::vector<Voxel*>& point_large,
        const std::unordered_set<const Voxel*>& bad_large_set,
        std::vector<char>& drop,
        std::vector<char>& restored,
        size_t* n_restored_out) const;
    static float lineFitRms3d(
        const pcl::PointCloud<pcl::PointXYZ>::Ptr& cloud,
        const std::vector<uint32_t>& idxs);
    // 拟合直线：成功返回 RMS，并写出中心与单位方向；失败返回 <0
    static float lineFit3d(
        const pcl::PointCloud<pcl::PointXYZ>::Ptr& cloud,
        const std::vector<uint32_t>& idxs,
        double* cx, double* cy, double* cz,
        double* ux, double* uy, double* uz);
    void removePointsInBadVoxels(
        pcl::PointCloud<pcl::PointXYZ>::Ptr& cloud,
        const std::vector<Voxel*>& bad_voxels,
        const std_msgs::msg::Header& header);
    void removePointsByMask(
        pcl::PointCloud<pcl::PointXYZ>::Ptr& cloud,
        const std::vector<char>& drop,
        const std_msgs::msg::Header& header);
    void publishFilterDebug(
        const pcl::PointCloud<pcl::PointXYZ>::Ptr& raw_cloud,
        const pcl::PointCloud<pcl::PointXYZ>::Ptr& pass_cloud,
        const std::vector<Voxel*>& bad_voxels,
        const std_msgs::msg::Header& header,
        const std::vector<char>* drop_mask = nullptr,
        const std::vector<char>* restored_mask = nullptr);
    // 团级诊断日志：把 RViz 标签上的 tN 团的几何特征打进日志，
    // 便于回答"这一团为什么被删/被捞回"
    void logClusterStats(
        const pcl::PointCloud<pcl::PointXYZ>::Ptr& cloud,
        const std::vector<char>& drop,
        const std::vector<char>& restored) const;
    // 模式1：团级自证。grid_ 必须是"小格"且已算好 cluster_id。
    // 团覆盖线数 >= cluster_min_rings_ 时，把该团被删的点还回来（点自身径向
    // 小于 cluster_min_radius_ 的不还）。
    void restoreByCluster(
        const pcl::PointCloud<pcl::PointXYZ>::Ptr& cloud,
        std::vector<char>& drop,
        std::vector<char>& restored,
        size_t* n_restored_out) const;
    int64_t makeKey(int ix, int iy, int iz) const;
    int64_t makeCoarseKey(int ox, int oy, int oz, int nxy, int nz) const;
    int floorDiv(int a, int b) const;
    int ringId(float x, float y, float z) const;
    // 优先用 ring_of_cur_[i] 里的真实线号，缺失时退回 atan2 几何估计
    int ringOfPoint(uint32_t i, float x, float y, float z) const;
    void addRing(Voxel& v, uint32_t i, float x, float y, float z) const;

    rclcpp::Subscription<sensor_msgs::msg::PointCloud2>::SharedPtr subscription_;
    rclcpp::Publisher<sensor_msgs::msg::PointCloud2>::SharedPtr publisher_;
    rclcpp::Publisher<sensor_msgs::msg::PointCloud2>::SharedPtr removed_publisher_;
    rclcpp::Publisher<sensor_msgs::msg::PointCloud2>::SharedPtr verdict_publisher_;
    rclcpp::Publisher<visualization_msgs::msg::MarkerArray>::SharedPtr boxes_publisher_;
    rclcpp::Publisher<visualization_msgs::msg::MarkerArray>::SharedPtr voxels_publisher_;
    rclcpp::Publisher<visualization_msgs::msg::MarkerArray>::SharedPtr voxels_deleted_publisher_;
    rclcpp::Publisher<visualization_msgs::msg::MarkerArray>::SharedPtr voxels_kept_publisher_;
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
    std::string verdict_topic_;
    std::string boxes_topic_;
    std::string voxels_topic_;
    std::string voxels_deleted_topic_;
    std::string voxels_kept_topic_;
    bool publish_occupied_voxels_{true};

    int memory_pool_size_{10};
    int memory_pool_reserve_{1000000};
    bool debug_mode_{true};
    double cloud_log_interval_sec_{1.0};

    bool frame_log_enabled_{false};
    std::chrono::steady_clock::time_point last_cloud_log_time_{};

    double cube_min_x_{-2.0};
    double cube_max_x_{2.0};
    double cube_min_y_{-2.0};
    double cube_max_y_{2.0};
    double cube_min_z_{-1.0};
    double cube_max_z_{1.0};

    // 体素参数：边长 = 默认值 × 整数倍率；倍率由距离×角分辨率相对默认值向上取整
    double ang_h_{0.0};
    double ang_v_{0.0};
    double r_max_{8.0};
    double base_xy_{0.01};
    double base_z_{0.01};
    double max_xy_{0.5};
    double max_z_{0.5};
    // 横向边长 = base_xy × 分段倍率，再夹在 [base_xy, max_xy]
    // bands1=小格先删；bands2=大格用整云再判，捞回小格误删点
    std::vector<SizeXyBand> size_xy_bands_;       // 当前生效（build/lookup 用）
    std::string size_xy_bands_raw_;
    std::vector<SizeXyBand> size_xy_bands1_;
    std::string size_xy_bands1_raw_;
    std::vector<SizeXyBand> size_xy_bands2_;
    std::string size_xy_bands2_raw_;
    // 纵向边长 = size_z_bands(r) × 线间距；bands 空则退回 size_z_ratio
    std::vector<SizeXyBand> size_z_bands_;
    std::string size_z_bands_raw_;
    double size_z_ratio_{4.0};
    double size_z_min_{0.1};
    // thr_z = thr_ratio(r) × 线间距；参数名仍叫 thr_ratio，格式同 bands
    std::vector<SizeXyBand> thr_ratio_bands_;
    std::string thr_ratio_raw_;
    double thr_z_min_{0.03};
    double cluster_link_m_{0.18};
    double cluster_link_k_{6.0};
    double cluster_plane_k_{10.0};
    bool enable_voxel_filter_{true};

    // 大格捞回：ring 上滑窗种子 + 向两侧生长
    bool enable_line_restore_gate_{true};
    double line_gap_k_{2.0};          // 邻点距 < k * r * ang_h
    int seed_win_{4};                 // 种子窗口点数
    double seed_rms_m_{0.02};         // 种子局部直线 RMS
    double grow_pred_m_{0.05};        // 生长：到当前直线预测偏差
    int grow_min_points_{3};          // 长完至少多少点才捞回
    // 豁免发放模式：
    //   0 = 大格判留 + ring 门闩（原行为）
    //   1 = 团级自证：小格连通团自己覆盖的线数够，就把整团被小格删的点全还给它
    int restore_mode_{0};
    int cluster_min_rings_{4};         // 模式1：团覆盖 ring 数下限
    // 模式1：点自身径向 < 该值一律不豁免（近场伪影区）；0 = 关闭
    double cluster_min_radius_{1.3};
    // 无 ring / 恢复 ring 各只打一次，避免刷屏
    mutable bool line_gate_warned_no_ring_{false};

    std::unordered_map<int64_t, Voxel> grid_;
    std::unordered_map<int64_t, Voxel> fine_grid_;

    // 与当前工作点云逐点对齐的真实线号；空表示这帧没有 ring 字段
    std::vector<uint16_t> ring_of_cur_;
    std::vector<uint16_t> ring_scratch_;
    bool have_ring_{false};
};

#endif  // VANJEE_LIDAR_FILTER__CLOUD_PASSTHROUGH_FILTER_HPP_
